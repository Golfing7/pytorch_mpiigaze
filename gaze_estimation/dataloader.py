from typing import Tuple, Union

from .dataset import create_dataset
from torch.utils.data import DataLoader


def create_dataloader(
        config,
        is_train: bool) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_train:
        train_dataset, val_dataset = create_dataset(config, is_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=config.train.train_dataloader.pin_memory,
            drop_last=config.train.train_dataloader.drop_last,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=config.train.val_dataloader.pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader
    else:
        test_dataset = create_dataset(config, is_train)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.test.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=config.test.dataloader.pin_memory,
            drop_last=False,
        )
        return test_loader
