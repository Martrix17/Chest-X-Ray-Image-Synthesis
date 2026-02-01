"""
Functions for single training/evaluation step and image generation.
"""

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel

from models.label_conditioner import MultiLabelConditioner


def single_step(
    images: torch.Tensor,
    labels: torch.Tensor,
    model: UNet2DConditionModel,
    conditioner: MultiLabelConditioner,
    scheduler: DDPMScheduler,
    cfg_drop_prob: float = 0.1,
    is_training: bool = True,
) -> torch.Tensor:
    """
    Single training/evaluation step.

    Args:
        images: Image tensor.
        labels: Label tensor.
        model: U-Net model.
        conditioner: Label conditioner.
        scheduler: Noise scheduler.
        cfg_drop_prob: CFG dropout probability.
        is_training: Whether step is for training or evaluation.

    Returns:
        Average loss.
    """
    device = next(model.parameters()).device
    batch_size = images.shape[0]

    if is_training and cfg_drop_prob > 0:
        keep_mask = torch.rand(batch_size, device=device) > cfg_drop_prob
        labels = labels * keep_mask.unsqueeze(1)

    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (batch_size,), device=device
    ).long()

    noise = torch.randn_like(images)
    noisy_images = scheduler.add_noise(images, noise, timesteps)

    class_embeds = conditioner(labels)

    noise_pred = model(
        noisy_images,
        timesteps,
        encoder_hidden_states=class_embeds,
        return_dict=False,
    )[0]

    return F.mse_loss(noise_pred, noise, reduction="mean")


@torch.inference_mode()
def generate_samples(
    model: UNet2DConditionModel,
    conditioner: MultiLabelConditioner,
    inference_scheduler: Any,
    labels: torch.Tensor,
    guidance_scale: float,
    image_shape: Sequence[int],
) -> torch.Tensor:
    """
    Generate images from labels using CFG.

    Args:
        model: U-Net model.
        conditioner: Label conditioner.
        inference_scheduler: Noise scheduler for inference.
        labels: Label tensor [B, num_classes].
        guidance_scale: CFG guidance scale.
        image_shape: Shape of output (B, C, H, W).

    Returns:
        Generated images [B, C, H, W] in range [-1, 1].
    """
    param = next(model.parameters())
    device, dtype = param.device, param.dtype

    cond_embeds = conditioner(labels)

    if guidance_scale > 1.0:
        uncond_labels = torch.zeros_like(labels)
        uncond_embeds = conditioner(uncond_labels)
        encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds], dim=0)
    else:
        encoder_hidden_states = cond_embeds

    sample = torch.randn(image_shape, device=device, dtype=dtype)
    sample = sample * inference_scheduler.init_noise_sigma

    for timestep in inference_scheduler.timesteps:
        latent_input = torch.cat([sample, sample], dim=0) if guidance_scale > 1.0 else sample
        latent_input = inference_scheduler.scale_model_input(latent_input, timestep)

        noise_pred = model(
            sample=latent_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0:
            noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        sample = inference_scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=sample,
            return_dict=False,
        )[0]

    return sample
