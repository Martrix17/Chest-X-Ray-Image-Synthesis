"""
Convenience functions for training, evaluation and generation loops.
"""

import itertools
from typing import Any, Dict, Optional, Sequence

import torch
from accelerate import Accelerator
from diffusers import UNet2DConditionModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.label_conditioner import MultiLabelConditioner
from models.noise_scheduler import create_inference_scheduler
from src.training.ema import EMA
from training.steps import generate_samples, single_step


def train_steps(
    global_step: int,
    loader: DataLoader,
    model: UNet2DConditionModel,
    ema: EMA,
    conditioner: MultiLabelConditioner,
    scheduler: Any,
    optimizer: Optimizer,
    accelerator: Accelerator,
    lr_scheduler: Optional[LRScheduler],
    max_steps: int,
    cfg_drop_prob: float = 0.1,
    log_every: int = 100,
    val_steps: int = 5000,
) -> int:
    """
    Train for for max steps or until next logging step.

    Args:
        global_step: Current global step number.
        loader: Training data loader.
        model: U-Net model.
        ema: EMA instance.
        conditioner: Label conditioner.
        scheduler: Noise scheduler.
        optimizer: Optimizer instance.
        accelerator: Accelerator instance.
        lr_scheduler: Learning rate scheduler.
        max_steps: Maximum training steps.
        cfg_drop_prob: CFG dropout probability.
        log_every: Log metrics every N steps.
        val_steps: Validate every N steps.

    Returns:
        New global_step.
    """
    model.train()
    conditioner.train()

    running_loss = 0.0
    steps_since_log = 0

    loader_iter = iter(loader)

    next_val_step = ((global_step // val_steps) + 1) * val_steps
    next_val_step = min(next_val_step, max_steps)

    steps_to_train = next_val_step - global_step

    pbar = tqdm(
        total=steps_to_train,
        desc=f"Training steps: {global_step} -> {next_val_step}",
        leave=False,
        disable=not accelerator.is_main_process,
    )

    while global_step < next_val_step:
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, labels = next(loader_iter)

        with accelerator.accumulate(model):
            loss = single_step(
                images=images,
                labels=labels,
                model=model,
                conditioner=conditioner,
                scheduler=scheduler,
                cfg_drop_prob=cfg_drop_prob,
                is_training=True,
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    itertools.chain(model.parameters(), conditioner.parameters()),
                    max_norm=1.0,
                )

            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                if lr_scheduler is not None:
                    lr_scheduler.step()

                global_step += 1
                pbar.update(1)

                if ema is not None:
                    ema.update(model)

                running_loss += loss.item()
                steps_since_log += 1

                if global_step % log_every == 0:
                    avg_loss = running_loss / max(steps_since_log, 1)

                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train_loss": avg_loss,
                                "lr": optimizer.param_groups[0]["lr"],
                            },
                            step=global_step,
                        )

                        pbar.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        )

                    running_loss = 0.0
                    steps_since_log = 0

    return global_step


@torch.no_grad()
def evaluate(
    global_step: int,
    loader: DataLoader,
    model: UNet2DConditionModel,
    conditioner: MultiLabelConditioner,
    scheduler: Any,
    accelerator: Accelerator,
    max_batches: Optional[int] = None,
    desc: str = "Val",
) -> float:
    """
    Evaluate a model.

    Args:
        global_step: Current global step number.
        loader: Evaluation data loader.
        model: U-Net model.
        conditioner: Label conditioner.
        scheduler: Noise scheduler.
        accelerator: Accelerator instance.
        max_batches: Number of batches to evaluate (optional).
        decs: Description for progress bar.

    Returns:
        Average evaluation loss.
    """
    model.eval()
    conditioner.eval()

    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        loader,
        desc=f"{desc}",
        leave=False,
        disable=not accelerator.is_main_process,
    )

    for images, labels in pbar:
        loss = single_step(
            images=images,
            labels=labels,
            model=model,
            conditioner=conditioner,
            scheduler=scheduler,
            cfg_drop_prob=0.0,
            is_training=False,
        )

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())

        if max_batches is not None and num_batches >= max_batches:
            break

    avg_loss = running_loss / max(num_batches, 1)

    if accelerator.is_main_process:
        accelerator.log({f"{desc}_loss": avg_loss}, step=global_step)

    return avg_loss


def generate_from_loader(
    loader: DataLoader,
    model: UNet2DConditionModel,
    conditioner: MultiLabelConditioner,
    scheduler: Any,
    guidance_scale: float,
    num_inference_steps: int,
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate sample images for visualization from data loader.

    Args:
        loader: DataLoader.
        model: U-Net model.
        conditioner: Label conditioner.
        scheduler: Noise scheduler.
        guidance_scale: CFG guidance scale.
        num_inference_steps: Number of denoising steps.
        max_samples: Maximum number of samples to generate (Optional).

    Returns:
        Dictionary with 'generated_images', 'real_images', and 'labels'.
    """
    model.eval()
    conditioner.eval()
    device = next(model.parameters()).device

    inference_scheduler = create_inference_scheduler(scheduler, num_inference_steps, device)

    generated_images = []
    real_images = []
    all_labels = []

    num_processed = 0

    pbar = tqdm(loader, desc="Generating samples", leave=False)
    for images, labels in pbar:
        if max_samples and num_processed >= max_samples:
            break

        batch_size = images.shape[0]
        image_shape = images.shape

        samples = generate_samples(
            model=model,
            conditioner=conditioner,
            inference_scheduler=inference_scheduler,
            labels=labels,
            guidance_scale=guidance_scale,
            image_shape=image_shape,
        )

        generated_images.append(samples)
        real_images.append(images)
        all_labels.append(labels)

        num_processed += batch_size
        pbar.set_postfix(num_generated=num_processed)

    generated_images = torch.cat(generated_images, dim=0)
    real_images = torch.cat(real_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "generated_images": generated_images,
        "real_images": real_images,
        "labels": all_labels,
    }


def generate_from_labels(
    model: UNet2DConditionModel,
    conditioner: MultiLabelConditioner,
    scheduler: Any,
    labels: torch.Tensor,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    image_shape: Sequence[int] = [1, 1, 128, 128],
) -> torch.Tensor:
    """
    Generate sample images for visualization.

    Args:
        model: U-Net model.
        conditioner: Label conditioner.
        scheduler: Noise scheduler.
        labels: Label tensor [B, num_classes].
        guidance_scale: CFG guidance scale.
        num_inference_steps: Number of denoising steps.
        image_shape: Output shape (B, C, H, W).

    Returns:
        Generated images [B, C, H, W] in range [-1, 1].
    """
    model.eval()
    conditioner.eval()

    device = next(model.parameters()).device
    labels = labels.to(device)

    inference_scheduler = create_inference_scheduler(scheduler, num_inference_steps, device)

    samples = generate_samples(
        model=model,
        conditioner=conditioner,
        inference_scheduler=inference_scheduler,
        labels=labels,
        guidance_scale=guidance_scale,
        image_shape=image_shape,
    )

    return samples
