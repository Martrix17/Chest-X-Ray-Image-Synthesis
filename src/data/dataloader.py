"""
Factory function for DataLoader instantiation.

Example:
    >>> train_loader, val_loader = create_dataloaders(config)
    >>> test_loader = create_dataloaders(config, train_mode=False)
"""

from typing import Sequence

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.dataset import MimicCXRDataset
from data.transforms import base_gray_transform


def create_dataloaders(
    config: DictConfig, train_mode: bool = True
) -> Sequence[DataLoader] | DataLoader:
    """
    Create dataloaders from a config dict.

    Args:
        config: Hydra config dictionary.
        train_mode: Whether to prepare train/val loaders or test loader.

    Returns:
        Train/val DataLoaders or a single test DataLoader.
    """

    image_size = int(config.data.transforms.image_size)

    transform = base_gray_transform(image_size=image_size)

    root_dir = str(config.data.root_dir)
    csv_path = str(config.data.csv_path)
    label_columns = list(config.data.label_columns)

    num_workers = int(config.data.dataloader.num_workers)
    pin_memory = bool(config.data.dataloader.pin_memory)
    persistent_workers = bool(config.data.dataloader.persistent_workers)

    if train_mode:
        train_dataset = MimicCXRDataset(
            root_dir=f"{root_dir}/train",
            csv_path=csv_path,
            label_columns=label_columns,
            transform=transform,
            subset_fraction=config.train.training.subset_fraction,
            normal_fraction=config.train.training.normal_fraction,
            seed=config.train.training.get("seed"),
        )

        val_dataset = MimicCXRDataset(
            root_dir=f"{root_dir}/valid",
            csv_path=csv_path,
            label_columns=label_columns,
            transform=transform,
        )

        batch_size = int(config.train.training.batch_size)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            drop_last=False,
        )

        return train_loader, val_loader

    else:
        test_dataset = MimicCXRDataset(
            root_dir=f"{root_dir}/test",
            csv_path=csv_path,
            label_columns=label_columns,
            transform=transform,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=int(config.test.testing.batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            drop_last=False,
        )

        return test_loader
