from typing import Dict, Any

import torch
import wandb
from sklearn.model_selection import train_test_split

from src.train.config import TrainingConfig
from src.model import AutoEncoder
from src.data import get_all_local_labels, load_data, QuickDrawImageDataset
from src.train import create_optimizer, log_training


def train_model(config: TrainingConfig):
    run = wandb.init(
        project="SketchCNN-AE",
        entity="kylesayrs",
        name=None,
        reinit=True,
        mode=config.wandb_mode,
        config=config.dict()
    )

    # load data
    all_local_labels = get_all_local_labels("images")
    all_images = load_data(
        config.data_dir,
        config.image_shape,
        class_names=all_local_labels
    )

    # create split
    x_train, x_test = train_test_split(
        all_images,
        test_size=config.test_size,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        QuickDrawImageDataset(x_train, augmentations=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        QuickDrawImageDataset(x_test, augmentations=False),
        batch_size=config.test_batch_size,
        shuffle=True,
        num_workers=0
    )
    print("created datasets")

    # create model, optimizer, and loss
    model = AutoEncoder(config.image_shape, config.latent_size).to(config.device)
    optimizer = create_optimizer(model, config.optimizer, lr=config.lr)
    criterion = torch.nn.MSELoss().to(config.device)

    # train model
    for epoch_index in range(config.num_epochs):
        for batch_index, images in enumerate(train_loader):
            # load data to device
            images = images.to(config.device)

            # forward
            optimizer.zero_grad()
            reconstructions, _latents = model(images)

            # backwards
            loss = criterion(images, reconstructions)
            loss.backward()

            # optimize
            optimizer.step()


            # log
            log_training(
                config,
                model,
                test_loader,
                len(train_loader),
                criterion,
                loss.item(),
                epoch_index,
                batch_index
            )
