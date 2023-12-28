import os
import shutil
from src.train import TrainingConfig, train_model


if __name__ == "__main__":
    print("Starting")
    config = TrainingConfig(
        wandb_mode="disabled",
        save_checkpoints=True,
        device="cuda",
        save_samples=True
    )
    print("Created config")
    train_model(config)
