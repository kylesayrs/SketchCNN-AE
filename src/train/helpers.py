import wandb
import torch

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
