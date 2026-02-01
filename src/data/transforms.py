"""
Torchvision image augmentations for gray-scale and RGB images.
"""

import torch
from torchvision.transforms import v2


def base_gray_transform(
    image_size: int = 256,
) -> v2.Compose:
    """
    Base transform for 1-channel images (grayscale) with [-1, 1] normalization.

    Args:
        image_size: Size to resize to.

    Returns:
        v2.Compose image transform.
    """
    return v2.Compose(
        [
            v2.Grayscale(),
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def augment_gray_transform(
    image_size: int = 256,
) -> v2.Compose:
    """
    Training transform for diffusion models with mild, anatomically-valid augmentation.

    Args:
        image_size: Size to resize to.

    Returns:
        v2.Compose image transform.
    """
    return v2.Compose(
        [
            # v2.RandomAffine(
            #     degrees=0,
            #     scale=(0.9, 1.1),
            #     interpolation=v2.InterpolationMode.BILINEAR,
            # ),
            # v2.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2,
            # ),
            # v2.RandomApply(
            #     [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            #     p=0.3,
            # ),
            # v2.RandomHorizontalFlip(p=0.5),
            *base_gray_transform(image_size).transforms,
        ]
    )
