"""
Inference script entry point for multilabel diffusion model.

Usage:
    python sample.py final.load_path=model/final_dir
"""

import hydra
from omegaconf import DictConfig

from src.runner.sample import run_sampling


@hydra.main(config_path="../configs", config_name="sample", version_base="1.3")
def main(config: DictConfig):
    try:
        run_sampling(config)

    except KeyboardInterrupt:
        print("\n\nSampling interrupted by user!")

    except Exception as e:
        print(f"\n\nSampling failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
