"""
Factory inference runner function.
"""

from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler
from omegaconf import DictConfig

from models.unet import create_unet
from src.training.loops import generate_from_labels
from training.ema import EMA
from utils.checkpointing import load_ema_state, load_model_state
from utils.io import save_samples, save_synthetic_metadata


def run_sampling(config: DictConfig):
    """
    Runs sampling.

    Args:
        config: Hydra configuration
    """
    # ------------------------------------------------------------------
    # Accelerator
    # ------------------------------------------------------------------
    accelerator = Accelerator(mixed_precision=config.sample.mixed_precision)

    # ------------------------------------------------------------
    # Models and components
    # ------------------------------------------------------------
    accelerator.print("Loading model, conditioner, scheduler..")
    model, conditioner = create_unet(config, checkpoint_gradients=False)

    scheduler = DDPMScheduler(
        num_train_timesteps=int(config.model.scheduler.num_train_timesteps),
        beta_schedule=str(config.model.scheduler.beta_schedule),
        prediction_type=str(config.model.scheduler.prediction_type),
        clip_sample=bool(config.model.scheduler.clip_sample),
    )

    final_path = Path(config.final.path)
    load_model_state(
        load_path=final_path,
        model=model,
        conditioner=conditioner,
        scheduler=scheduler,
    )

    # ------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------
    ema = None
    if config.ema.enabled:
        ema = EMA(model, decay=float(config.ema.ema_decay))

        load_ema_state(
            load_path=final_path,
            ema=ema,
            device=accelerator.device,
        )

    model = ema.get_model() if ema is not None else model

    # ------------------------------------------------------------
    # Accelerate prepare
    # ------------------------------------------------------------
    model, conditioner = accelerator.prepare(model, conditioner)

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------
    accelerator.print("Generating images from sample labels.")
    sample_labels = torch.tensor(
        config.generation.sample_labels,
        dtype=torch.float32,
    )
    image_shape = (
        sample_labels.shape[0],
        config.model.model.in_channels,
        config.data.transforms.image_size,
        config.data.transforms.image_size,
    )

    generated_images = generate_from_labels(
        labels=sample_labels,
        model=model,
        conditioner=conditioner,
        scheduler=scheduler,
        guidance_scale=config.cfg.guidance_scale,
        num_inference_steps=config.generation.num_inference_steps,
        image_shape=image_shape,
    )

    # ------------------------------------------------------------
    # Output saving
    # ------------------------------------------------------------
    results_path = Path(config.results.path)
    accelerator.print(f"Saving outputs to {results_path}..")

    save_samples(images=generated_images, out_dir=results_path)
    save_synthetic_metadata(
        images=generated_images,
        labels=sample_labels,
        label_names=config.data.label_columns,
        out_dir=results_path,
    )

    accelerator.print("Sampling complete!")
    accelerator.end_training()
