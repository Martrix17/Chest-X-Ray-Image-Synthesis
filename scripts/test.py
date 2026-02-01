"""
Test script entry point for multilabel diffusion model.

Usage:
    python test.py final.load_path=model/final_dir
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from runner.test import run_test


@hydra.main(config_path="../configs", config_name="test", version_base="1.3")
def main(config: DictConfig):
    print("=" * 80)
    print("Test Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80 + "\n")

    try:
        run_test(config)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user!")

    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
