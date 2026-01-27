from pydantic_settings import BaseSettings
from torch.optim import Optimizer
from torch.nn import Module
from typing import ClassVar


class TrainingParams(BaseSettings):
    DEFAULT_BATCH_SIZE: ClassVar[int] = 32
    DEFAULT_CHECKPOINTS_DIR: ClassVar[str] = "./checkpoints"

    learning_rate: float
    num_epochs: int
    batch_size: int = DEFAULT_BATCH_SIZE
    criterion: Module
    optimizer: Optimizer
    cats_dogs_checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR
