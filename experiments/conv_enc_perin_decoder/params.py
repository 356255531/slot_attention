from os.path import dirname, abspath, join
from typing import List, Tuple, Optional
import attr


@attr.s(auto_attribs=True)
class DeviceParams:
    device_type: str = "gpu"
    devices: List[int] = [0]
    precision: int = 32


@attr.s(auto_attribs=True)
class DataParams:
    img_dim: Tuple[int, int] = (3, 60, 60)
    data_root: str = join(abspath(__file__ + "/../../../"), "data/color_multi_mnist/")
    num_objects: int = 3
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None


@attr.s(auto_attribs=True)
class ModelParams:
    encoder: str = "small"
    decoder: str = "small"
    hidden_dim: int = 512
    embedding_dim: int = 128
    checkpoint_path: str = None
    every_n_epochs: int = 5
    save_top_k: int = 5


@attr.s(auto_attribs=True)
class TrainParams:
    seed: int = 2023
    lr: float = 0.0004
    max_epochs: int = 150
    batch_size: int = 16
    val_batch_size: int = 128
    visual_batch_size: int = 64
    test_batch_size: int = 128
    num_workers: int = 8
    early_stop_patience: int = 10
    early_stop_mode: bool = "min"
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
    n_samples: int = 40


@attr.s(auto_attribs=True)
class LoggerParams:
    project: str = "unsupervised-multi-object-disentanglement"
    name: str = "conv_encoder_perinvariant_decoder"
    id: str = None
    save_dir: str = dirname(abspath(__file__))
    offline: bool = True
    log_every_n_steps: int = 50


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