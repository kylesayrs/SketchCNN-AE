import os
import shutil
from src.train import TrainingConfig, train_model


if __name__ == "__main__":
    config = TrainingConfig(
        #latent_size=16,
        wandb_mode="disabled",
        save_checkpoints=False,
        save_samples=True
    )
    train_model(config)
