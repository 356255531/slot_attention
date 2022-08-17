from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
from pytorch_lightning import Callback
import kornia.geometry as tgm
import torch.nn.functional as F
import wandb

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


def conv_transpose_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def group_transformation(images, param):
    bb, s, c, h, w = images.shape
    b = bb * s
    images = images.reshape(bb * s, c, h, w)
    param = param.reshape(bb * s, 4)

    rot_param, trans_param = param[:, :2], param[:, 2: 4]

    # # Rotate image
    # rot = F.normalize(rot_param, p=2, dim=1)
    # rot_ortho = torch.stack([-rot[:, 1], rot[:, 0]], dim=-1)
    # rot = torch.stack([rot, rot_ortho], dim=-1)
    # center = torch.ones((b, 2, 1), device=images.device) * (h - 1) / 2
    # eye_mat = torch.eye(2, device=images.device).reshape((1, 2, 2)).expand(b, -1, -1)
    # offset = torch.bmm(eye_mat - rot, center)
    # affine_mat = torch.cat([rot, offset], dim=-1)
    # images = tgm.warp_affine(images, affine_mat, (h, w), padding_mode='border')

    # translate the image
    trans = torch.sigmoid(trans_param) - 0.5
    trans[:, 0] *= 0.5 * h
    trans[:, 1] *= 0.5 * w
    images = tgm.translate(images, trans, padding_mode='border')

    images = images.reshape(bb, s, c, h, w)

    return images


def rescale(x: Tensor) -> Tensor:
    return x * 2 - 1


def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.log({"images": [wandb.Image(images)]}, commit=False)


def to_rgb_from_tensor(x: Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)
