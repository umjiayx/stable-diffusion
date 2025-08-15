# train.py
import omegaconf
import pytorch_lightning as pl
from model import MNISTClassifier
from dataset import MNISTDataModule

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    config = omegaconf.OmegaConf.load("config.yaml")
    logger = TensorBoardLogger("my_logs", name="mnist_run")

    dm = MNISTDataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    model = MNISTClassifier(config=config)

    # Optional: fix a deterministic directory for checkpoints
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{logger.save_dir}/{logger.name}/version_{logger.version}/checkpoints",
        filename="best-checkpoint",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        gpus=1,  # using PL 1.x here; if you upgrade, switch to accelerator/devices
        check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCH,
        logger=logger,
        callbacks=[ckpt_cb],
    )

    trainer.fit(model, dm)

    print("Best checkpoint:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
