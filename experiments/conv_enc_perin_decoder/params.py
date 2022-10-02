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
    num_objects: int = 4


@attr.s(auto_attribs=True)
class ModelParams:
    encoder: str = "medium"
    decoder: str = "medium"
    hidden_dim: int = 512
    embedding_dim: int = 64
    checkpoint_path: str = None
    every_n_epochs: int = 10
    save_top_k: int = 5


@attr.s(auto_attribs=True)
class TrainParams:
    seed: int = 2023
    max_epochs: int = 1000
    # LR
    lr: float = 10e-3
    factor: float = 0.67
    patience: int = 5
    min_lr: float = 10e-7
    # Batch size
    train_batch_size: int = 128
    val_batch_size: int = 128
    visual_batch_size: int = 20
    test_batch_size: int = 128
    # Dataloader
    num_workers: int = 8
    # Early stop
    early_stop_patience: int = 10
    early_stop_mode: bool = "min"


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
