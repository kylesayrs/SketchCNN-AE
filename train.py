from src.train import TrainingConfig, train_model


if __name__ == "__main__":
    config = TrainingConfig(wandb_mode="disabled", samples_dir="samples")
    train_model(config)
