from typing import Tuple, Union
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    data_dir: str = Field(default="images")
    image_shape: Tuple[int, int] = Field(default=(50, 50))
    test_size: float = Field(default=0.2)

    num_epochs: int = Field(default=2)
    batch_size: int = Field(default=128)
    lr: float = Field(default=1e-2)
    optimizer: str = Field(default="adam")
    momentum: float = Field(default=0.9)

    logging_rate: int = Field(default=100_000)
    test_batch_size: int = Field(default=128)
    save_checkpoints: bool = Field(default=True)
    device: str = Field(default="cpu")
    wandb_mode: str = Field(default="online")

    samples_dir: Union[str, None] = Field(default=None)
