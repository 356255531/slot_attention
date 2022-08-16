import os
from typing import Callable
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ComMnistDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        transforms: Callable,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"

    def __getitem__(self, index: int):
        img = Image.open(self.data_root + f"images/{self.split}/{index}.png")
        img = img.convert("RGB")
        return self.transforms(img)

    def __len__(self):
        return 100000 if self.split == "train" else 5000


class ComMnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        transforms: Callable,
        max_n_objects: int,
        num_workers: int,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.transforms = transforms
        self.max_n_objects = max_n_objects
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images

        self.train_dataset = ComMnistDataset(
            data_root=self.data_root,
            transforms=self.transforms,
            split="train",
        )
        self.val_dataset = ComMnistDataset(
            data_root=self.data_root,
            transforms=self.transforms,
            split="val",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ComMnistTransforms(object):
    def __init__(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
