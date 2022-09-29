import os.path
from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class DataParams:
    img_dim: Tuple[int, int] = (3, 60, 60)
    data_root: str = os.path.join(os.path.abspath(__file__ + "/../../../"), "data/color_multi_mnist/")
    num_objects: int = 3
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    num_workers: int = 8

@attr.s(auto_attribs=True)
class ModelParams:
    lr: float = 0.0004
    encoder: str = "small"
    input_channels: int = 3
    hidden_dim: int = 512
    embedding_dim: int = 16


@attr.s(auto_attribs=True)
class TrainParams:
    seed: int = 2023
    lr: float = 0.0004
    batch_size: int = 16
    val_batch_size: int = 128
    gpus: list = [0]
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
    n_samples: int = 40


@attr.s(auto_attribs=True)
class Params:
    data = DataParams()
    model = ModelParams()
    train = TrainParams()
    max_epochs: int = 150
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0

    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
