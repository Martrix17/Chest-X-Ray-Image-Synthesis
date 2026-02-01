"""
Factory test runner function.
"""

from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler
from omegaconf import DictConfig, OmegaConf

from data.dataloader import create_dataloaders
from metrics.metrics import MetricsComputer
from models.unet import create_unet
from training.ema import EMA
from training.loops import evaluate, generate_from_loader
from utils.checkpointing import load_ema_state, load_model_state
from utils.io import save_grid, save_metrics, save_samples, save_synthetic_metadata


def run_test(config: DictConfig):
    """
    Runs test evaluation.

    Args:
        config: Hydra configuration.
    """
    # ------------------------------------------------------------------
    # Accelerator
    # ------------------------------------------------------------------
    accelerator = Accelerator(
        mixed_precision=config.test.testing.mixed_precision,
        log_with="mlflow",
    )

    # ------------------------------------------------------------
    # Tracking / logging
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------
    seed = config.test.testing.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    # ------------------------------------------------------------
    # Models and components
    # ------------------------------------------------------------
    accelerator.print("Creating model...")
    model, conditioner = create_unet(config=config, checkpoint_gradients=False)

    scheduler = DDPMScheduler(
        num_train_timesteps=config.model.scheduler.num_train_timesteps,
        beta_schedule=config.model.scheduler.beta_schedule,
        prediction_type=config.model.scheduler.prediction_type,
        clip_sample=config.model.scheduler.clip_sample,
    )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    accelerator.print("Loading data..")
    test_loader = create_dataloaders(config, train_mode=False)

    # ------------------------------------------------------------
    # Load Model State
    # ------------------------------------------------------------
    final_path = Path(config.final.path)

    accelerator.print(f"Loading model states from {final_path}...")
    load_model_state(
        load_path=final_path,
        model=model,
        conditioner=conditioner,
        scheduler=scheduler,
    )

    ema = None
    if config.test.ema.enabled:
        ema = EMA(model, decay=float(config.test.ema.ema_decay))

        accelerator.print("Loading EMA state...")
        load_ema_state(
            load_path=final_path,
            ema=ema,
            device=accelerator.device,
        )

    model = ema.get_model() if ema is not None else model

    # ------------------------------------------------------------
    # Accelerate prepare
    # ------------------------------------------------------------
    model, conditioner, test_loader = accelerator.prepare(model, conditioner, test_loader)

    # ------------------------------------------------------------------
    # Metrics computer
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        metrics_computer = MetricsComputer(
            use_fd_dino=bool(config.test.metrics.fd.get("dino", False)),
            use_fd_inception=bool(config.test.metrics.fd.get("inception", False)),
            use_lpips=bool(config.test.metrics.lpips.get("diversity", False)),
            device=accelerator.device,
        )

    # ------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------
    accelerator.print("Starting test and generating samples..")
    test_loss = evaluate(
        global_step=0,
        loader=test_loader,
        model=model,
        conditioner=conditioner,
        scheduler=scheduler,
        accelerator=accelerator,
        desc="Test",
    )

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------
    accelerator.print("Generating images from loader.")
    dataset_size = len(test_loader.dataset)
    max_samples = config.test.generation.get("max_samples", 20)
    max_samples = max_samples if max_samples <= dataset_size else dataset_size

    output = generate_from_loader(
        loader=test_loader,
        model=model,
        conditioner=conditioner,
        scheduler=scheduler,
        guidance_scale=config.test.cfg.guidance_scale,
        num_inference_steps=config.model.scheduler.num_inference_steps,
        max_samples=max_samples,
    )
    real_images = output["real_images"]
    generated_images = output["generated_images"]

    # ------------------------------------------------------------
    # Metrics calculation
    # ------------------------------------------------------------
    accelerator.print("Computing metrics..")
    metrics = metrics_computer.compute_all(
        real_samples=real_images,
        fake_samples=generated_images,
        max_samples=config.test.metrics.get("max_samples", 500),
        max_pairs=config.test.metrics.get("max_pairs", 1000),
    )

    metrics.update({"test_loss": test_loss})

    # ------------------------------------------------------------
    # Output saving
    # ------------------------------------------------------------
    results_path = Path(config.results.path)
    accelerator.print(f"Saving outputs to {results_path}..")

    num_save_image = config.test.generation.get("num_save_image", 16)
    save_metrics(metrics=metrics, out_dir=results_path)
    save_samples(images=generated_images[:num_save_image], out_dir=results_path)
    save_grid(images=generated_images[:num_save_image], out_dir=results_path)
    save_synthetic_metadata(
        images=generated_images[:num_save_image],
        labels=output["labels"][:num_save_image],
        label_names=config.data.label_columns,
        out_dir=results_path,
    )

    if accelerator.is_main_process:
        accelerator.log(metrics, step=0)

    metrics_computer.cleanup()

    accelerator.print("Test complete!")
    accelerator.end_training()
