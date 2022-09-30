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


@attr.s(auto_attribs=True)
class ModelParams:
    lr: float = 0.001
    encoder: str = "small"
    input_channels: int = 3
    hidden_dim: int = 512
    embedding_dim: int = 128


@attr.s(auto_attribs=True)
class DeviceParams:
    device_type: str = "gpu"
    device:


@attr.s(auto_attribs=True)
class TrainParams:
    seed: int = 2023
    gpus: list = [0]
    lr: float = 0.0004
    max_epochs: int = 150
    batch_size: int = 16
    val_batch_size: int = 128
    num_workers: int = 8
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
    n_samples: int = 40


@attr.s(auto_attribs=True)
class LoggerParams:
    logger: bool = True
    project: str = "unsupervised-multi-object-disentanglement"
    logger_name: str = "conv_encoder_perinvariant_decoder"


@attr.s(auto_attribs=True)
class Params:
    device = DeviceParams()
    data = DataParams()
    model = ModelParams()
    train = TrainParams()
    logger = LoggerParams()

    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0

    empty_cache: bool = True
    is_logger_enabled: bool = False
    is_verbose: bool = True
