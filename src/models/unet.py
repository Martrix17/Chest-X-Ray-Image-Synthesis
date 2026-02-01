"""
Factory functions for UNet diffusion model and label conditioner.
"""

from typing import Tuple

from diffusers import UNet2DConditionModel
from omegaconf import DictConfig

from models.label_conditioner import MultiLabelConditioner


def create_unet(
    config: DictConfig,
    checkpoint_gradients: bool = False,
) -> Tuple[UNet2DConditionModel, MultiLabelConditioner]:
    """
    Create conditional 2DUNet model and label conditioner (embedder).

    Args:
        config: Hydra config dictionary.
        checkpoint_gradients: Whether to enable gradient checkpointing.

    Returns:
        UNet2DConditionModel and MultiLabelConditioner.
    """

    model_config = config.model.model

    model = UNet2DConditionModel(
        sample_size=int(config.data.transforms.image_size),
        in_channels=int(model_config.in_channels),
        out_channels=int(model_config.out_channels),
        down_block_types=list(model_config.down_block_types),
        up_block_types=list(model_config.up_block_types),
        block_out_channels=list(model_config.block_out_channels),
        layers_per_block=int(model_config.layers_per_block),
        attention_head_dim=int(model_config.attention_head_dim),
        cross_attention_dim=int(model_config.cross_attention_dim),
    )

    conditioner = MultiLabelConditioner(
        num_classes=len(config.data.label_columns),
        embed_dim=int(config.model.conditioner.embed_dim),
        dropout=float(config.model.conditioner.dropout),
    )

    if checkpoint_gradients:
        model.enable_gradient_checkpointing()

    return model, conditioner
