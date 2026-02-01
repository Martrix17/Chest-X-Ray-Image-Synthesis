"""
Factory training runner function.
"""

import json
import shutil
import tempfile
from pathlib import Path

import torch
import torchvision.utils as vutils
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from data.dataloader import create_dataloaders
from metrics.metrics import MetricsComputer
from models.noise_scheduler import create_noise_scheduler
from models.unet import create_unet
from training.ema import EMA
from training.loops import evaluate, generate_from_loader, train_steps
from utils.checkpointing import (
    load_ema_state,
    load_model_state,
    load_training_state,
    save_model,
    save_training_state,
)


def run_training(config: DictConfig):
    """
    Main training runner.

    Args:
        config: Hydra configuration (DictConfig)
    """

    # ------------------------------------------------------------------
    # Accelerator
    # ------------------------------------------------------------------
    accelerator = Accelerator(
        mixed_precision=config.train.training.mixed_precision,
        gradient_accumulation_steps=config.train.training.gradient_accumulation_steps,
        log_with="mlflow",
    )

    # ------------------------------------------------------------------
    # Tracking / logging (convert only for tracker)
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.mlflow.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={
                "mlflow": {
                    "logging_dir": config.mlflow.tracking_uri,
                    "run_name": config.mlflow.run_name,
                }
            },
        )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed = config.train.training.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    accelerator.print("Creating model...")
    model, conditioner = create_unet(
        config=config,
        checkpoint_gradients=bool(config.train.training.gradient_checkpointing),
    )

    scheduler = create_noise_scheduler(config=config.model.scheduler)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    accelerator.print("Loading data...")
    train_loader, val_loader = create_dataloaders(config=config)

    # ------------------------------------------------------------------
    # Optimizer & LR scheduler
    # ------------------------------------------------------------------#
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()),
        lr=config.train.optimizer.lr,
        betas=config.train.optimizer.betas,
        weight_decay=config.train.optimizer.weight_decay,
    )

    num_training_steps = config.train.training.num_train_steps

    lr_scheduler = None
    if not config.train.training.fine_tune:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.train.lr_scheduler.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # ------------------------------------------------------------------
    # Resume (optional)
    # ------------------------------------------------------------------
    start_step = 0
    resume_path = config.checkpoint.get("resume_from")
    ckpt_registry = []

    if resume_path is not None:
        resume_path = Path(resume_path)
        accelerator.print(f"Resuming from checkpoint: {resume_path}")

        load_model_state(
            load_path=resume_path,
            model=model,
            conditioner=conditioner,
            scheduler=scheduler,
            device="cpu",
        )

        start_step = load_training_state(
            load_path=resume_path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device="cpu",
        )

        registry_file = resume_path.parent / "checkpoint_registry.json"
        if registry_file.exists():
            with open(registry_file, "r") as f:
                ckpt_registry = json.load(f)

    # ------------------------------------------------------------------
    # Accelerate prepare
    # ------------------------------------------------------------------
    model, conditioner, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, conditioner, optimizer, train_loader, val_loader, lr_scheduler
    )

    # ------------------------------------------------------------------
    # EMA (optional)
    # ------------------------------------------------------------------
    ema = None
    if config.train.ema.enabled:
        accelerator.print("Creating EMA...")
        ema = EMA(model, decay=config.train.ema.ema_decay)

    if resume_path is not None and ema is not None:
        load_ema_state(
            load_path=resume_path,
            ema=ema,
            device=accelerator.device,
        )

    # ------------------------------------------------------------------
    # Metrics computer
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        metrics_computer = MetricsComputer(
            use_fd_dino=bool(config.train.metrics.fd.get("dino", False)),
            use_fd_inception=bool(config.train.metrics.fd.get("inception", False)),
            use_lpips=bool(config.train.metrics.lpips.get("diversity", False)),
            device=accelerator.device,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    tmp_dir = tempfile.gettempdir()
    ckpt_root = Path(config.checkpoint.path)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    registry_path = ckpt_root / "checkpoint_registry.json"

    val_steps = config.train.validation.val_steps
    sample_steps = config.train.generation.sample_steps
    metrics_steps = config.train.metrics.metrics_steps
    save_steps = config.checkpoint.save_steps

    if config.train.training.fine_tune:
        num_training_steps = start_step + config.train.training.fine_tune_steps
        accelerator.print(f"Fine tuning from step {start_step} to {num_training_steps}...")
    else:
        accelerator.print(f"Training from step {start_step} to {num_training_steps}...")

    global_step = start_step
    while global_step < num_training_steps:
        metrics = {}
        output = {}
        val_loss = None

        # ---------------- Training ----------------
        global_step = train_steps(
            global_step=global_step,
            loader=train_loader,
            model=model,
            ema=ema,
            conditioner=conditioner,
            scheduler=scheduler,
            optimizer=optimizer,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            max_steps=num_training_steps,
            cfg_drop_prob=config.train.cfg.cfg_drop_prob,
            log_every=config.train.training.log_every,
            val_steps=val_steps,
        )

        # ---------------- Validation ----------------
        if global_step % val_steps == 0:
            val_loss = evaluate(
                global_step=global_step,
                loader=val_loader,
                model=ema.get_model() if ema else model,
                conditioner=conditioner,
                scheduler=scheduler,
                accelerator=accelerator,
                max_batches=config.train.validation.get("max_val_batches", None),
                desc="Val",
            )

        # ---------------- Sampling ----------------
        if accelerator.is_main_process and global_step % sample_steps == 0:
            accelerator.print(f"Generating monitoring samples at step {global_step}...")

            output = generate_from_loader(
                loader=val_loader,
                model=ema.get_model() if ema else model,
                conditioner=conditioner,
                scheduler=scheduler,
                guidance_scale=config.train.cfg.guidance_scale,
                num_inference_steps=config.model.scheduler.num_inference_steps,
                max_samples=config.train.generation.get("max_samples"),
            )

            # Save grid images
            num_save_image = config.train.generation.get("num_save_image", 16)
            grid = vutils.make_grid(
                output["generated_images"][:num_save_image].cpu(),
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )
            grid_array = (grid.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")

            image_path = Path(tmp_dir) / f"step_{global_step:07d}.jpg"
            img = Image.fromarray(grid_array)
            with open(image_path, "wb") as f:
                img.save(f, format="JPEG", quality=95)

            accelerator.get_tracker("mlflow").log_artifact(
                image_path, artifact_path="monitoring_samples"
            )

            # Save labels
            val_labels = {}
            for idx, row in enumerate(output["labels"][:num_save_image].cpu()):
                indices = torch.where(row == 1)[0]
                val_labels[f"sample_{idx}"] = [
                    config.data.label_columns[i] for i in indices.tolist()
                ]

            label_path = Path(tmp_dir) / f"labels_{global_step:07d}.json"
            with open(label_path, "w") as f:
                json.dump(val_labels, f, indent=4)

            accelerator.get_tracker("mlflow").log_artifact(
                label_path, artifact_path="monitoring_labels"
            )

            # ---------------- Metrics calculation ----------------
            if global_step % metrics_steps == 0:
                accelerator.print("Calculating metrics...")

                metrics = metrics_computer.compute_all(
                    real_samples=output["real_images"],
                    fake_samples=output["generated_images"],
                    max_samples=config.train.metrics.fd.get("max_samples", 500),
                )

                accelerator.log(metrics, step=global_step)

                torch.cuda.empty_cache()
                del output

        # ---------------- Checkpointing ----------------
        if accelerator.is_main_process and global_step % save_steps == 0:
            ckpt_path = ckpt_root / f"step_{global_step:07d}"

            save_model(
                model=model,
                conditioner=conditioner,
                scheduler=scheduler,
                save_path=ckpt_path,
                accelerator=accelerator,
                ema=ema,
            )

            save_training_state(
                save_path=ckpt_path,
                step=global_step,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

            sort_by = config.checkpoint.get("sort_by", "val_loss")

            ckpt_registry.append(
                {
                    "step": global_step,
                    "fd_dino": metrics.get("fd_dino"),
                    "fd_inception": metrics.get("fd_inception"),
                    "lpips_diversity": metrics.get("lpips_diversity"),
                    "val_loss": val_loss,
                    "path": str(ckpt_path),
                }
            )
            ckpt_registry.sort(key=lambda x: x.get(sort_by, float("inf")))

            # ---------------- Retention policy ----------------
            while len(ckpt_registry) > config.checkpoint.keep_top_k:
                worst = ckpt_registry.pop(-1)
                shutil.rmtree(worst["path"], ignore_errors=True)

            # ---------------- Persist registry ----------------
            with open(registry_path, "w") as f:
                json.dump(ckpt_registry, f, indent=2)

            accelerator.print(f"Checkpoint saved at step {global_step} with best {sort_by}.")

    accelerator.print("Training complete!")

    if metrics_computer is not None:
        metrics_computer.cleanup()

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    if accelerator.is_main_process and len(ckpt_registry) > 0:
        best_ckpt = ckpt_registry[0]
        best_path = Path(best_ckpt["path"])

        final_path = Path(config.final.path)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(best_path, final_path, dirs_exist_ok=True)

        accelerator.print(f"Final model copied from step {best_ckpt['step']}.")

    accelerator.end_training()
