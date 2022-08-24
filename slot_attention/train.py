from typing import Optional
import sys
import os
import numpy as np
import random as rd

import torch
from torch.utils.data import DataLoader

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

sys.path.append(os.getcwd())

from slot_attention.data import ComMnistDataset, ComMnistTransforms
from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ImageLogCallback


def main(logger_name, params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        print(f"INFO: limiting the dataset to only images with `num_slots - 1` ({params.num_slots - 1}) objects.")
        if params.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}")

    # Define dataset
    commnist_transforms = ComMnistTransforms(params.resolution)

    train_dataset = ComMnistDataset(
            data_root=params.data_root,
            transforms=commnist_transforms,
            split="train",
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
    )

    val_dataset = ComMnistDataset(
            data_root=params.data_root,
            transforms=commnist_transforms,
            split="val",
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params.val_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
    )

    print(f"Training set size (images must have {params.num_slots - 1} objects):", len(train_dataset))

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        empty_cache=params.empty_cache,
        params=params,
    )

    logger = pl_loggers.WandbLogger(project="slot_attention", name=logger_name)

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="gpu" if len(params.gpus) > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus if len(params.gpus) > 0 else None,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if params.is_logger_enabled else [],
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)
    rd.seed(seed)
    main(f"commnist96pixel_nonsymmetric_scale_rot_tsla_{seed}")
