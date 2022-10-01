from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from udl.data import ColorMultiMinist, ColorMultiMinistTransforms
from udl.models import ObjDisentangleAE as model_class
from udl.utils import ImageLogCallback

from experiments.conv_enc_perin_decoder.params import Params


def main(params):
    # define dataset
    train_dataset = ColorMultiMinist(
            data_root=params.data.data_root,
            transforms=ColorMultiMinistTransforms(params.data.img_dim[1:]),
            split="train",
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.train.batch_size,
        shuffle=True,
        num_workers=params.train.num_workers,
        pin_memory=True,
    )

    val_dataset = ColorMultiMinist(
            data_root=params.data.data_root,
            transforms=ColorMultiMinistTransforms(params.data.img_dim[1:]),
            split="val",
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params.train.val_batch_size,
        shuffle=False,  # For the purpose of plotting different images
        num_workers=params.train.num_workers,
        pin_memory=True,
    )

    # init the callback
    callbacks = [ImageLogCallback()]
    if params.logger.logger:
        callbacks += [LearningRateMonitor("step")]

    # set up the trainer
    trainer = pl.Trainer(
        # device parameters
        accelerator=params.device.device_type,
        devices=params.device.devices,
        precision=params.device.precision,
        # train parameters
        max_epochs=params.train.max_epochs,
        benchmark=True,  # accelerate the speed when input size does not change
        resume_from_checkpoint=params.model.checkpoint_path,
        gradient_clip_val=None,
        # logger
        logger=WandbLogger(
            project=params.logger.project,
            name=params.logger.logger_name
        ) if params.logger.logger else None,
        # LEAVE_ME_ALONE: usually you don't need to touch here
        enable_progress_bar=True,
        num_sanity_val_steps=-1,
        callbacks=callbacks,
        track_grad_norm=2,
        log_every_n_steps=params.logger.log_every_n_steps,
        enable_model_summary=True,
        move_metrics_to_cpu=False,  # True when the GPU memory is critical
    )
    trainer.fit(model_class(params), train_dataloader, val_dataloader)


if __name__ == "__main__":
    params = Params()
    pl.seed_everything(params.train.seed)
    main(params)
