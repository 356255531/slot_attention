from typing import Optional
import sys
import os

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

sys.path.append(os.getcwd())

from slot_attention.data import ComMnistDataModule
from slot_attention.method import SlotAttentionMethod
from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ImageLogCallback
from slot_attention.utils import rescale


def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        print(f"INFO: limiting the dataset to only images with `num_slots - 1` ({params.num_slots - 1}) objects.")
        if params.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}")

    commnist_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),  # rescale between -1 and 1
            transforms.Resize(params.resolution),
        ]
    )

    commnist_datamodule = ComMnistDataModule(
        data_root=params.data_root,
        max_n_objects=params.num_slots - 1,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        transforms=commnist_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
    )

    print(f"Training set size (images must have {params.num_slots - 1} objects):", len(commnist_datamodule.train_dataset))

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
    )

    method = SlotAttentionMethod(model=model, datamodule=commnist_datamodule, params=params)

    logger_name = "commnist_rot_tsla"
    logger = pl_loggers.WandbLogger(project="slot_attention", name=logger_name)

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="ddp" if len(params.gpus) > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus if len(params.gpus) > 0 else None,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if params.is_logger_enabled else [],
    )
    trainer.fit(method)


if __name__ == "__main__":
    main()
