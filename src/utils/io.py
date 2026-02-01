"""
Utility functions for writing outputs to local directories.
"""

import json
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
import torchvision.utils as vutils


def save_synthetic_metadata(
    images: torch.Tensor,
    labels: torch.Tensor,
    label_names: Sequence[str],
    out_dir: Path,
) -> None:
    """
    Save generated images metadata as csv-file.

    Args:
        images: Image Tensors.
        labels: Tensor containing multi-hot encoded labels.
        label_names: Names of classes in labels.
        out_dir: Directory to save output to.
    """
    rows = []

    for i in range(len(images)):
        row = {
            "filename": f"sample_{i:03d}.jpg",
            "split": "synth",
        }
        for j, name in enumerate(label_names):
            row[name] = float(labels[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "synthetic_metadata.csv", index=False)


def save_samples(images: torch.Tensor, out_dir: Path) -> None:
    """
    Save image samples.

    Args:
        images: Image Tensors.
        out_dir: Directory to save image samples to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples"
    out_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        vutils.save_image(
            img,
            out_path / f"sample_{i:03d}.jpg",
            normalize=True,
            value_range=(-1, 1),
        )


def save_grid(images: torch.Tensor, out_dir: Path, nrow: int = 4) -> None:
    """
    Save multiple images into one grid.

    Args:
        images: Image Tensors to save in a grid.
        out_dir: Directory to save grid images to.
        nrow: Number of images per row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid.jpg"
    grid = vutils.make_grid(
        images,
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )
    vutils.save_image(grid, out_path)


def save_metrics(metrics: Dict[str, float], out_dir: Path) -> None:
    """
    Save metrics to json file.

    Args:
        metrics: Dictionary containing metrics.
        output_dir: Directory to save metrics to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=4)
