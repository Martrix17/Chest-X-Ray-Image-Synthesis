"""
Training script entry point for multilabel diffusion model.

Usage:
    python train.py
    python train.py training.num_epochs=20
    python train.py generation.sample_size=128 training.batch_size=8
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from runner.train import run_training


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig):
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80 + "\n")

    try:
        run_training(config)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
