from typing import Tuple, List

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
    def __init__(self, image_shape: Tuple[int, int]):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            *conv_down(1, 32, bn=False),
            *conv_down(32, 32),
            *conv_down(32, 64),
            *conv_down(64, 64),
            *conv_down(64, 64),
        )

        self.decoder = torch.nn.Sequential(
            *conv_up(64, 64, bn=False),
            *conv_up(64, 64),
            *conv_up(64, 32),
            *conv_up(32, 32),
            *conv_up(32, 32),
            *conv_up_final(32, 1),
            torch.nn.Flatten(),
            torch.nn.Linear(65 * 65, image_shape[0] * image_shape[1]),
            torch.nn.Unflatten(1, (1, *image_shape)),
        )

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        z = self.encoder(x)
        y = self.decoder(z)

        return y, z

        
