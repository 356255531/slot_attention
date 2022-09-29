import torch
import pytorch_lightning as pl

from udl.data import ColorMultiMinist, ColorMultiMinistTransforms
from udl.models import ObjDisentangleAE
from udl.utils import ImageLogCallback

from experiments.conv_enc_perin_decoder.params import Params


def main(params):
    # Define dataset
    train_dataset = ColorMultiMinist(
            data_root=params.data.data_root,
            transforms=ColorMultiMinistTransforms(params.data.img_dim[1:]),
            split="train",
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.train.batch_size,
        shuffle=True,
        num_workers=params.data.num_workers,
        pin_memory=True,
    )

    val_dataset = ColorMultiMinist(
            data_root=params.data.data_root,
            transforms=ColorMultiMinistTransforms(params.data.img_dim[1:]),
            split="val",
        )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params.train.val_batch_size,
        shuffle=False,
        num_workers=params.data.num_workers,
        pin_memory=True,
    )

    model = ObjDisentangleAE(params)


    trainer = pl.Trainer(
        accelerator="gpu" if len(params.train.gpus) > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.train.gpus if len(params.train.gpus) > 0 else None,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        callbacks=[pl.callbacks.LearningRateMonitor("step"), ImageLogCallback(),] if params.is_logger_enabled else [],
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    params = Params()
    pl.seed_everything(params.train.seed)
    main(params)
