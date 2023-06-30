import argparse
import os
import shutil

import box
import pytorch_lightning as pl
import torch
import yaml

import data
import fine_recon
import utils


def load_config(config_fname):
    with open(config_fname, "r") as f:
        config = box.Box(yaml.safe_load(f))

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        config.accelerator = "gpu"
        config.n_devices = n_gpus
    else:
        config.accelerator = "cpu"
        config.n_devices = 1

    return config


@pl.utilities.rank_zero_only
def zip_code(save_dir):
    os.system(f"zip {save_dir}/code.zip *.py config.yml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument(
        "--task", default="train", choices=["train", "predict", "find_lr"]
    )
    parser.add_argument("--ckpt")
    args = parser.parse_args()

    if args.ckpt is not None:
        shutil.copy(args.ckpt, args.ckpt + ".bak")

    config = load_config(args.config)
    if args.task == "predict":
        config.n_devices = 1

    model = fine_recon.FineRecon(config)

    logger = pl.loggers.TensorBoardLogger(save_dir=".", version=config.run_name)
    logger.experiment

    zip_code(logger.experiment.log_dir)

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.accelerator,
        devices=config.n_devices,
        max_steps=config.steps,
        log_every_n_steps=50,
        precision=16,
        strategy="ddp" if config.n_devices > 1 else None,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="loss_val/loss", save_top_k=10)
        ],
    )

    if args.task == "train":
        trainer.fit(model, ckpt_path=args.ckpt)

    elif args.task == "find_lr":
        tuner = pl.tuner.Tuner(trainer)
        model.lr = model.config.initial_lr
        lr_finder = tuner.lr_find(
            model, train_dataloaders=model.train_dataloader(), val_dataloaders=[]
        )
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr.png")

    elif args.task == "predict":
        trainer.predict(model, ckpt_path=args.ckpt)
    else:
        raise NotImplementedError
