"""
Dataset class for MIMI-CXR dataset.

Example:
    >>> train_dataset = MimicCXRDataset(
    ...   root_dir=root_dir,
    ...   csv_path=csv_path.csv,
    ...   label_columns=[Atelectasis, Cardiomegaly, Consolidation, ...],
    ...   transform=transform,
    ...   subset_fraction=0.5,
    ...   normal_fraction=0.3,
    ...   seed=0,
    ... )
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class MimicCXRDataset(Dataset):
    """
    MIMIC-CXR dataset using pre-split directories (train/valid/test).

    Expected CSV columns:
        - filename
        - split
        - pathology label columns (0.0 or 1.0)

    Images are expected under:
        root_dir/{filename}
    """

    def __init__(
        self,
        root_dir: str | Path,
        csv_path: str | Path,
        label_columns: Sequence[str],
        transform: v2.Compose,
        subset_fraction: Optional[float] = None,
        normal_fraction: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            root_dir: Path to dataset directory
            csv_path: Path to metadat csv-file.
            label_columns: Columns to consider using from csv Dataframe.
            transform: Image transform.
            subset_fraction: Fraction of dataset to use (optional).
            normal_fraction: Fraction of dataset belonging to normal class (optinal).
            seed: Float for reproducability, when creating dataset fractions (optional).
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"csv_path not found: {self.csv_path}")

        self.label_columns = label_columns
        if label_columns is None:
            raise ValueError("label_columns must be provided explicitly.")

        self.transform = transform
        if transform is None:
            raise ValueError("transform must be provided explicitly.")

        split = self.root_dir.name
        df = pd.read_csv(self.csv_path, skipinitialspace=True)
        df.drop(columns="label")

        missing_cols = set(label_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing label columns in CSV: {missing_cols}")

        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split '{split}'. "
                f"Available splits: {df['split'].unique().tolist()}"
            )

        if subset_fraction is not None and normal_fraction is not None:
            self.df = self._make_subset(
                df=self.df,
                subset_fraction=subset_fraction,
                normal_fraction=normal_fraction,
                seed=seed,
            )

        print(f"Loaded {len(self.df)} samples for split '{split}'.")

    def _make_subset(
        self,
        df: pd.DataFrame,
        subset_fraction: float,
        normal_fraction: float,
        seed: int,
    ) -> pd.DataFrame:
        """
        Create a semantic subset with explicit Normal vs Pathological control.

        Args:
            df: Full dataframe for split.
            subset_fraction: Fraction of total samples to keep (0 < f <= 1).
            normal_fraction: Fraction of subset that should be normal class.
            seed: RNG seed.

        Returns:
            Subset dataframe.
        """
        assert 0 < subset_fraction <= 1.0
        assert 0 < normal_fraction < 1.0

        rng = torch.Generator().manual_seed(seed)

        labels = df[self.label_columns]

        is_normal = (labels.sum(axis=1) == 1) & (labels["Normal"] == 1)
        df_normal = df[is_normal]
        df_path = df[~is_normal]

        total_target = int(len(df) * subset_fraction)
        n_normal = int(total_target * normal_fraction)
        n_patho = total_target - n_normal

        if n_normal > len(df_normal):
            raise ValueError("Requested more normal samples than available.")
        if n_patho > len(df_path):
            raise ValueError("Requested more pathological samples than available.")

        normal_idx = torch.randperm(len(df_normal), generator=rng)[:n_normal]
        patho_idx = torch.randperm(len(df_path), generator=rng)[:n_patho]

        subset_df = (
            pd.concat(
                [
                    df_normal.iloc[normal_idx.tolist()],
                    df_path.iloc[patho_idx.tolist()],
                ],
                axis=0,
            )
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )

        label_counts = subset_df[self.label_columns].sum()
        missing = label_counts[label_counts == 0]
        if len(missing) > 0:
            raise RuntimeError(f"Subset dropped labels entirely: {missing.index.tolist()}")

        return subset_df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Current index to image.

        Returns:
            Tuple containing image and label tensors.
        """
        for _ in range(3):  # bounded retry
            row = self.df.iloc[idx]
            image_path = self.root_dir / row["filename"]

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                break
            except (OSError, ValueError):
                idx = (idx + 1) % len(self.df)
        else:
            raise RuntimeError("Repeated image loading failure in dataset")

        image = self.transform(image)

        labels = self.df.loc[idx, self.label_columns].astype("float32").to_numpy()
        labels = torch.from_numpy(labels)

        return image, labels
