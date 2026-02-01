"""
Factory functions for noise scheduler initialization.
"""

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from omegaconf import DictConfig


def create_noise_scheduler(config: DictConfig) -> DDPMScheduler | DDIMScheduler:
    """
    Create and return a noise scheduler based on the specified type and configuration.

    Args:
        config: Configuration dictionary for the scheduler.

    Returns:
        Noise scheduler DDPM or DIMM.
    """
    type = config.type.lower()
    if type == "ddpm":
        scheduler = DDPMScheduler
    elif type == "ddim":
        scheduler = DDIMScheduler
    else:
        raise ValueError(f"Unsupported scheduler type: {type}")

    return scheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule=config.beta_schedule,
        prediction_type=config.prediction_type,
        clip_sample=config.clip_sample,
    )


def create_inference_scheduler(
    scheduler: DDPMScheduler | DDIMScheduler,
    num_inference_steps: int,
    device: torch.device,
) -> DDPMScheduler | DDIMScheduler:
    """
    Create and return an inference scheduler based on the provided scheduler configuration.

    Args:
        scheduler: The original scheduler used during training.
        num_inference_steps: Number of inference steps for generation.

    Returns:
        Noise scheduler DDPM or DIMM.
    """

    if isinstance(scheduler, DDPMScheduler):
        inference_scheduler = DDPMScheduler.from_config(scheduler.config)
    elif isinstance(scheduler, DDIMScheduler):
        inference_scheduler = DDIMScheduler.from_config(scheduler.config)
    else:
        raise ValueError("Unsupported scheduler type for inference.")

    inference_scheduler.set_timesteps(num_inference_steps, device=device)

    return inference_scheduler
