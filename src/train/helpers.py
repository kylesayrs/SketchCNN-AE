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
    criterion: torch.nn.Module,
    train_loss: float,
    epoch_index: int,
    batch_index: int
):
    test_loss = test_model(config, model, test_loader, criterion)
    log_metrics(config, epoch_index, batch_index, train_loss, test_loss)

    if config.samples_dir is not None:
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

    figure, axes = plt.subplots(1, 2)
    axes[0].imshow(image.cpu().numpy().squeeze(0).squeeze(0))
    axes[1].imshow(reconstruction.cpu().numpy().squeeze(0).squeeze(0))

    os.makedirs(config.samples_dir, exist_ok=True)
    file_name = f"sample_{epoch_index}_{batch_index}.png"
    save_path = os.path.join(config.samples_dir, file_name)
    plt.savefig(save_path)


def log_metrics(
    config: TrainingConfig,
    epoch_index: int,
    batch_index: int,
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
        f"[{epoch_index} / {config.num_epochs}] [{batch_index}] "
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
