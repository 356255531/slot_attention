import os
from typing import Callable
from typing import Tuple

from PIL import Image
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
        self.len = len([name for name in os.listdir(self.data_root + f"images/{self.split}/")])
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"

    def __getitem__(self, index: int):
        img = Image.open(self.data_root + f"images/{self.split}/{index}.png")
        img = img.convert("RGB")
        return self.transforms(img)

    def __len__(self):
        return self.len


class ComMnistTransforms(object):
    def __init__(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2 * x - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
