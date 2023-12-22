import os
import wandb
import torch
import matplotlib.pyplot as plt

from src.train import TrainingConfig


def create_optimizer(
    model: torch.nn.Module,
    optimizer: str,
    **optimizer_kwargs
) -> torch.optim.Optimizer:
    match optimizer.lower():
        case "adam":
            return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        
    raise ValueError(f"Unknown optimizer {optimizer}")


def log_training(
    config: TrainingConfig,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_batches: int,
    criterion: torch.nn.Module,
    train_loss: float,
    epoch_index: int,
    batch_index: int
):
    if (epoch_index * num_batches + batch_index) % config.logging_rate == 0:
        test_loss = test_model(config, model, test_loader, criterion)
        log_metrics(config, epoch_index, batch_index, num_batches, train_loss, test_loss)

        if config.save_samples:
            save_sample(config, epoch_index, batch_index, model, test_loader)


def save_sample(
    config: TrainingConfig,
    epoch_index: int,
    batch_index: int,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
):
    image = next(iter(test_loader))[0: 1]
    image = image.to(config.device)

    with torch.no_grad():
        reconstruction, _latent = model(image)

    _figure, axes = plt.subplots(1, 2)
    axes[0].imshow(image.cpu().numpy().squeeze(0).squeeze(0))
    axes[1].imshow(reconstruction.cpu().numpy().squeeze(0).squeeze(0))

    dir_path = os.path.join("samples", f"{wandb.run.id}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = f"sample_{epoch_index}_{batch_index}.png"
    save_path = os.path.join(dir_path, file_name)
    plt.savefig(save_path)


def log_metrics(
    config: TrainingConfig,
    epoch_index: int,
    batch_index: int,
    num_batches: int,
    train_loss: float,
    test_loss: float
):  
    train_loss_normed = train_loss / config.batch_size
    test_loss_normed = test_loss / config.batch_size

    wandb.log({
        "train_loss": train_loss_normed,
        "test_loss": test_loss_normed,
    })

    print(
        f"[{epoch_index} / {config.num_epochs}] "
        f"[{batch_index} / {num_batches}] "
        f"train_loss: {train_loss_normed:.5f} "
        f"test_loss: {test_loss_normed:.5f} "
    )


def test_model(
    config: TrainingConfig,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module
) -> float:
    test_images = next(iter(test_loader))
    test_images = test_images.to(config.device)

    with torch.no_grad():
        test_reconstructions, _latents = model(test_images)
        test_loss = criterion(test_images, test_reconstructions)

    return test_loss.item()
