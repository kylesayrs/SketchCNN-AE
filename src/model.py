from typing import Tuple, List

import math
import torch


def conv_down(in_filters: int, out_filters: int, bn: bool = True) -> List[torch.nn.Module]:
    block = [
        torch.nn.BatchNorm2d(in_filters) if bn else None,
        torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.2)
    ]
    block = [layer for layer in block if layer is not None]
    
    return block


def conv_up(in_filters: int, out_filters: int, bn: bool = True) -> List[torch.nn.Module]:
    block = [
        torch.nn.BatchNorm2d(in_filters) if bn else None,
        torch.nn.ConvTranspose2d(in_filters, out_filters, 3, 2, 1),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.2)
    ]
    block = [layer for layer in block if layer is not None]
    
    return block


def conv_up_final(in_filters: int, out_filters: int) -> List[torch.nn.Module]:
    return [
        torch.nn.BatchNorm2d(in_filters),
        torch.nn.ConvTranspose2d(in_filters, out_filters, 3, 2, 1),
        torch.nn.Sigmoid()
    ]


class AutoEncoder(torch.nn.Module):
    def __init__(self, image_shape: Tuple[int, int], latent_size: int):
        super().__init__()
        self.image_shape = image_shape

        # get sqrt of latent size
        latent_size_sqrt = math.sqrt(latent_size)
        if int(latent_size_sqrt) != latent_size_sqrt:
            raise ValueError("latent size must be a square number for resizing purposes")
        latent_size_sqrt = int(latent_size_sqrt)

        # define encoder
        self.encoder = torch.nn.Sequential(
            *conv_down(1, 32, bn=False),
            *conv_down(32, 32),
            *conv_down(32, 64),
            *conv_down(64, 64),
            *conv_down(64, 64),
            torch.nn.Flatten(),
            torch.nn.Linear(256, latent_size),
        )

        # define decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (1, latent_size_sqrt, latent_size_sqrt)),
            *conv_up(1, 32, bn=False),
            *conv_up(32, 16),
            *conv_up(16, 1),
            torch.nn.Flatten(),
            torch.nn.Linear(14641, image_shape[0] * image_shape[1]),
            torch.nn.Unflatten(1, (1, *image_shape)),
        )

        # validate model shapes are correct
        self.validate()

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        z = self.encoder(x)
        y = self.decoder(z)

        return y, z


    def validate(self):
        test_input = torch.rand((1, 1, *self.image_shape))
        self.forward(test_input)
