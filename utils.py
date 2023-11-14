import pathlib
import random
from typing import Tuple

import requests
import numpy as np
import torch
import os
import tqdm
import tarfile
from preprocess import process_dataset


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_and_process_dataset(config) -> None:
    if not os.path.isdir(config.dataset.dataset_dir):
        os.makedirs(config.dataset.dataset_dir)

    path = os.path.join(config.dataset.dataset_dir, 'data.tar.gz')
    if not os.path.exists(path):
        url = "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz"
        print("Downloading dataset...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            print(r.headers)
            with open(path, 'wb') as f:
                with tqdm.tqdm(total=float(r.headers['Content-Length'])) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        # if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        f.close()

    # Next, we need to unzip the file.
    if not os.path.exists(os.path.join(config.dataset.dataset_dir, 'MPIIGaze')):
        print("Extracting zip file...")
        file = tarfile.open(path, mode='r')
        all_members = file.getmembers()
        total_members = len(file.getmembers())
        with tqdm.tqdm(total=total_members) as pbar:
            for member in all_members:
                file.extract(member, config.dataset.dataset_dir)
                pbar.update(1)

    # Finally, pre-process the dataset.
    if not os.path.exists(config.dataset.processed_dataset_path):
        print("Pre-Processing the dataset...")
        process_dataset(config)


def setup_cudnn(config) -> None:
    torch.backends.cudnn.benchmark = config.cudnn.benchmark
    torch.backends.cudnn.deterministic = config.cudnn.deterministic


def create_train_output_dir(config) -> pathlib.Path:
    output_root_dir = pathlib.Path(config.train.output_dir)
    if config.train.test_id != -1:
        output_dir = output_root_dir / f'{config.train.test_id:02}'
    else:
        output_dir = output_root_dir / 'all'
    if output_dir.exists():
        os.removedirs(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def convert_to_unit_vector(
        angles: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pitches = angles[:, 0]
    yaws = angles[:, 1]
    x = -torch.cos(pitches) * torch.sin(yaws)
    y = -torch.sin(pitches)
    z = -torch.cos(pitches) * torch.cos(yaws)
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(predictions: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y, pred_z = convert_to_unit_vector(predictions)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
