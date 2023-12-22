import torch
import numpy
from torchvision import transforms

from competitive_drawing.train.utils.RandomResizePad import RandomResizePad


class QuickDrawImageDataset(torch.utils.data.Dataset):
    def __init__(self, images: numpy.ndarray, augmentations: bool = False):
        self.images = images
        if augmentations:
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(5, shear=5),
                transforms.ToTensor(),
                RandomResizePad(scale=(0.3, 1.0), value=0)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.images[idx]

        return self.transform(image)
