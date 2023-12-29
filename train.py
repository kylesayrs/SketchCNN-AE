from src.train import TrainingConfig, train_model


if __name__ == "__main__":
    config = TrainingConfig(
        wandb_mode="disabled",
        save_checkpoints=True,
        device="cpu",
        save_samples=True,
        latent_size=256,
    )
    train_model(config)
