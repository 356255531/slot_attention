import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from params import Params
from udl.data import ColorMultiMinist, ColorMultiMinistTransforms
from udl.models import ObjDisentangleAE as model_class


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

    test_dataset = ColorMultiMinist(
        data_root=params.data.data_root,
        transforms=ColorMultiMinistTransforms(params.data.img_dim[1:]),
        split="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params.train.test_batch_size,
        shuffle=False,  # For the purpose of plotting different images
        num_workers=params.train.num_workers,
        pin_memory=True,
    )

    # init the callback
    callbacks = [
        ModelCheckpoint(
            None,  # save to wandb logger dir
            "{epoch}-{train_loss:.2f}-{val_loss:.2f}",
            save_last=True,
            monitor="val_loss",
            save_top_k=params.model.save_top_k,
            mode="min",
            every_n_epochs=params.model.every_n_epochs
        ),
        EarlyStopping(
            "val_loss",
            patience=params.train.early_stop_patience,
            mode=params.train.early_stop_mode,
            strict=True,
            check_finite=True
        ),
        LearningRateMonitor("step")
    ]

    # init model
    model = model_class(params)

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
            name=params.logger.name,
            id=params.logger.id,
            save_dir=params.logger.save_dir,
            offline=params.logger.offline,
        ),
        # LEAVE_ME_ALONE: usually you don't need to touch here
        enable_progress_bar=True,
        num_sanity_val_steps=-1,
        callbacks=callbacks,
        track_grad_norm=2,
        log_every_n_steps=params.logger.log_every_n_steps,
        enable_model_summary=True,
        move_metrics_to_cpu=False,  # True when the GPU memory is critical
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    params = Params()
    pl.seed_everything(params.train.seed)
    main(params)
