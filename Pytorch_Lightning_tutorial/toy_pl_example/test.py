# test.py

import omegaconf
import pytorch_lightning as pl
from model import MNISTClassifier
from dataset import MNISTDataModule


def main():
    config = omegaconf.OmegaConf.load("config.yaml")

    # Path to the best checkpoint from training
    ckpt_path = "lightning_logs/version_0/checkpoints/best-checkpoint.ckpt"

    # Load model from checkpoint
    model = MNISTClassifier.load_from_checkpoint(ckpt_path)

    # DataModule for test data
    dm = MNISTDataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    trainer = pl.Trainer(accelerator="auto", devices="auto")

    # This will call model.test_step()
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
