import os
import shutil
from src.train import TrainingConfig, train_model


if __name__ == "__main__":
    config = TrainingConfig(
        wandb_mode="disabled",
        save_checkpoints=True,
        save_samples=True
    )
    train_model(config)
