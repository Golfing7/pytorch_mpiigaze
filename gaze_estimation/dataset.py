import pathlib
from typing import Callable, Tuple

import h5py
import torch
import torchvision
import numpy as np
from .utils import load_and_process_dataset
from typing import List, Union
from torch.utils.data import Dataset


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable):
        self.transform = transform

        # In case of the MPIIGaze dataset, each image is so small that
        # reading image will become a bottleneck even with HDF5.
        # So, first load them all into memory.
        with h5py.File(dataset_path, 'r') as f:
            images = f.get(f'{person_id_str}/image')[()]
            poses = f.get(f'{person_id_str}/pose')[()]
            gazes = f.get(f'{person_id_str}/gaze')[()]
        assert len(images) == 3000
        assert len(poses) == 3000
        assert len(gazes) == 3000
        self.images = images
        self.poses = poses
        self.gazes = gazes

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        pose = torch.from_numpy(self.poses[index])
        gaze = torch.from_numpy(self.gazes[index])
        return image, pose, gaze

    def __len__(self) -> int:
        return len(self.images)


def create_dataset(config,
                   is_train: bool = True) -> Union[List[Dataset], Dataset]:

    dataset_dir = pathlib.Path(config.dataset.processed_dataset_path)

    if not dataset_dir.exists():
        load_and_process_dataset(config)

    assert dataset_dir.exists()
    assert config.train.test_id in range(-1, 15)
    assert config.test.test_id in range(15)
    person_ids = [f'p{index:02}' for index in range(15)]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255),
        torch.from_numpy,
        torchvision.transforms.Lambda(lambda x: x[None, :, :]),
    ])

    if is_train:
        if config.train.test_id == -1:
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform)
                for person_id in person_ids
            ])
            assert len(train_dataset) == 45000
        else:
            test_person_id = person_ids[config.train.test_id]
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform)
                for person_id in person_ids if person_id != test_person_id
            ])
            assert len(train_dataset) == 42000

        val_ratio = config.train.val_ratio
        assert val_ratio < 1
        val_num = int(len(train_dataset) * val_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        test_person_id = person_ids[config.test.test_id]
        test_dataset = OnePersonDataset(test_person_id, dataset_dir, transform)
        assert len(test_dataset) == 3000
        return test_dataset

