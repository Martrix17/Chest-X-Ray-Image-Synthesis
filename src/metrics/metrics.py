"""
Metrics computation class.

Example:
    >>> metrics_computer = MetricsComputer(
    ...     use_fd_dino=True,
    ...     use_fd_dino=True,
    ...     use_lpips=True,
    ...     device=torch.device('cuda')
    ... )
    >>> metrics = metrics_computer.compute_all(
    ...     real_samples=real_images,
    ...     fake_sampels=generated_images,
    ... )
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import AutoImageProcessor, AutoModel


class MetricsComputer:
    """Persistent metrics computer class."""

    def __init__(
        self,
        use_fd_dino: bool,
        use_fd_inception: bool,
        use_lpips: bool,
        device: torch.device,
    ) -> None:
        """
        Args:
            use_fd_dino: Whether to use Fréchet Distance DINOv2.
            use_fd_inception: Whether to use Fréchet Distance Inception.
            use_lpips: Whether to use Learned Perceptual Image Patch Similarity.
            device: Torch device to calculate metrics on.
        """
        self.device = device
        self._dino_model = None
        self._dino_processor = None
        self._inception_model = None
        self._lpips_model = None

        self._initialize_models(use_fd_dino, use_fd_inception, use_lpips)

    def _initialize_models(
        self,
        use_fd_dino: bool,
        use_fd_inception: bool,
        use_lpips: bool,
    ) -> None:
        """
        Lazy load models as needed to attribute models.

        Args:
            use_fd_dino: Whether to use Fréchet Distance DINOv2.
            use_fd_inception: Whether to use Fréchet Distance Inception.
            use_lpips: Whether to use Learned Perceptual Image Patch Similarity.
        """
        if use_fd_dino and self._dino_model is None:
            model_name = "facebook/dinov2-small"
            self._dino_model = AutoModel.from_pretrained(model_name).to(self.device)
            self._dino_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            self._dino_model.eval()

            for param in self._dino_model.parameters():
                param.requires_grad = False

        if use_fd_inception and self._inception_model is None:
            self._inception_model = FrechetInceptionDistance(feature=2048, normalize=True).to(
                self.device
            )

        if use_lpips and self._lpips_model is None:
            self._lpips_model = LearnedPerceptualImagePatchSimilarity(
                net_type="alex",
                normalize=False,
            ).to(self.device)

    def compute_all(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        max_samples: int = 500,
        max_pairs: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute all enabled metrics at once.

        Args:
            real_samples: Real images (N, C, H, W)
            fake_samples: Generated images (N, C, H, W)
            max_samples: Max samples for FD metrics
            max_pairs: Max pairs for LPIPS diversity

        Returns:
            Dictionary of computed metrics.
        """
        metrics = {}

        if real_samples.shape[0] < 2 or fake_samples.shape[0] < 2:
            print(
                f"Warning: Not enough samples for metrics "
                f"(real={real_samples.shape[0]}, fake={fake_samples.shape[0]})"
            )
            return metrics

        try:
            if self._dino_model is not None:
                metrics["fd_dino"] = self._compute_fd_dino(real_samples, fake_samples, max_samples)
        except Exception as e:
            print(f"Error computing FD-DINOv2: {e}")
            metrics["fd_dino"] = float("nan")

        try:
            if self._inception_model is not None:
                metrics["fd_inception"] = self._compute_fd_inception(
                    real_samples, fake_samples, max_samples
                )
                self._inception_model.reset()
        except Exception as e:
            print(f"Error computing FD-Inception: {e}")
            metrics["fd_inception"] = float("nan")

        try:
            if self._lpips_model is not None:
                metrics["lpips_diversity"] = self._compute_lpips_diversity(fake_samples, max_pairs)
        except Exception as e:
            print(f"Error computing LPIPS diversity: {e}")
            metrics["lpips_diversity"] = float("nan")

        return metrics

    def _compute_fd_dino(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        max_samples: int = 500,
    ) -> float:
        """
        Computes Fréchet Distance using DINOv2 embeddings between real images
        and generated images.

        Args:
            real_samples: Tensor with values are (N, C, H, W).
            fake_samples: Tensor with values are (N, C, H, W).
            device: Torch device.
            max_samples: Cap for real/fake samples.

        Returns:
            FD with DINO score.
        """
        real_images = self._prepare_for_dino(real_samples[:max_samples].to(self.device))
        fake_images = self._prepare_for_dino(fake_samples[:max_samples].to(self.device))

        with torch.no_grad():
            real_inputs = self._dino_processor(
                images=real_images, return_tensors="pt", do_rescale=False
            )
            fake_inputs = self._dino_processor(
                images=fake_images, return_tensors="pt", do_rescale=False
            )

            real_inputs = {k: v.to(self.device) for k, v in real_inputs.items()}
            fake_inputs = {k: v.to(self.device) for k, v in fake_inputs.items()}

            real_out = self._dino_model(**real_inputs)
            fake_out = self._dino_model(**fake_inputs)

            real_features = real_out.last_hidden_state[:, 0].cpu().numpy()
            fake_features = fake_out.last_hidden_state[:, 0].cpu().numpy()

        mu_real = real_features.mean(axis=0)
        mu_fake = fake_features.mean(axis=0)

        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)

        eps = 1e-6
        sigma_real += np.eye(sigma_real.shape[0]) * eps
        sigma_fake += np.eye(sigma_fake.shape[0]) * eps

        return float(self._frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake))

    def _compute_fd_inception(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        max_samples: int = 500,
    ) -> float:
        """
        Computes Fréchet Distance with Inception (FID) between real images
        and generated images.

        Args:
            real_samples: Tensor with values are (N, C, H, W).
            fake_samples: Tensor with values are (N, C, H, W).
            device: Torch device.
            max_samples: Cap for real/fake samples.

        Returns:
            FID score.
        """
        real = self._prepare_for_inception(real_samples[:max_samples].to(self.device))
        fake = self._prepare_for_inception(fake_samples[:max_samples].to(self.device))

        self._inception_model.update(real, real=True)
        self._inception_model.update(fake, real=False)

        return self._inception_model.compute().item()

    def _compute_lpips_diversity(
        self,
        fake_samples: torch.Tensor,
        max_pairs: Optional[int] = None,
    ) -> float:
        """
        Computes LPIPS-based perceptual diversity for generated samples.

        Args:
            fake_samples: Tensor (N, C, H, W), diffusion outputs in [-1, 1].
            device: Torch device.
            max_pairs: Optional cap on number of pairwise comparisons.

        Returns:
            LPIPS diversity score.
        """
        fake_samples = fake_samples.to(self.device)
        fake_samples = (fake_samples + 1) / 2
        fake_samples = torch.clamp(fake_samples, 0, 1)

        if fake_samples.shape[1] == 1:
            fake_samples = fake_samples.repeat(1, 3, 1, 1)

        distances = []
        n = fake_samples.shape[0]

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if max_pairs is not None and len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]

        for i, j in pairs:
            d = self._lpips_model(
                fake_samples[i].unsqueeze(0),
                fake_samples[j].unsqueeze(0),
            )
            distances.append(d.item())

        return float(sum(distances) / len(distances)) if distances else 0.0

    def _prepare_for_dino(self, images: torch.Tensor) -> torch.Tensor:
        """
        Converts diffusion outputs [-1, 1], 1-channel -> [0, 1], 3-channel for Dino.

        Args:
            images: Tensor images to convert.

        Returns:
            Converted tensor images.
        """
        if images.ndim != 4:
            raise ValueError(f"Expected 4D tensor (N, C, H, W), got shape {images.shape}")

        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)

        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        if images.shape[2] != 224 or images.shape[3] != 224:
            images = F.interpolate(
                images,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        return images

    def _frechet_distance(
        self,
        mu_real: float,
        sigma_real: float,
        mu_fake: float,
        sigma_fake: float,
    ) -> float:
        """
        Computes the Frechet Distance between two multivariate Gaussians.

        Args:
            mu_real: Mean of real samples.
            sigma_real: Covariance of real samples.
            mu_fake: Mean of fake samples.
            sigma_fake: Covariance of fake samples.

        Returns:
            Frechet Distance.
        """
        covmean = sqrtm(sigma_real @ sigma_fake)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return np.sum((mu_real - mu_fake) ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)

    def _prepare_for_inception(self, images: torch.Tensor) -> torch.Tensor:
        """
        Converts diffusion outputs [-1, 1], 1-channel -> [0, 1], 3-channel for Inception.

        Args:
            images: Tensor images to convert.

        Returns:
            Converted tensor images.
        """
        if images.ndim != 4:
            raise ValueError(f"Expected 4D tensor (N, C, H, W), got shape {images.shape}")

        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)

        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

        images = (images * 255).to(torch.uint8)
        return images

    def cleanup(self):
        """Free GPU memory when done."""
        if self._dino_model is not None:
            del self._dino_model
            del self._dino_processor
        if self._inception_model is not None:
            del self._inception_model
        if self._lpips_model is not None:
            del self._lpips_model

        torch.cuda.empty_cache()
