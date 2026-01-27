import torch
import torch.nn as nn
from torch.optim import AdamW

from dataloaders.dataloaders import train_set_dataloader
from models.cats_dogs_resnet import ResNetCatsDogsClassifier
from configs.training_params import TrainingParams
from training.training import NNTrainer
from utils.device import get_device
from loguru import logger

if __name__ == "__main__":
    resnet_model = ResNetCatsDogsClassifier(pretrained=True).to(get_device())

    lr: float = 0.0001
    training_params = TrainingParams(
        learning_rate=lr,
        num_epochs=5,
        batch_size=32,
        criterion=nn.CrossEntropyLoss(),
        optimizer=AdamW(resnet_model.parameters(), lr=lr),
    )

    trainer = NNTrainer(
        model=resnet_model,
        optimizer=training_params.optimizer,
        criterion=training_params.criterion,
        train_loader=train_set_dataloader,
        device=get_device(),
    )
    trainer.train(n_epochs=training_params.num_epochs)

    # Save trained weights
    torch.save(
        resnet_model.state_dict(),
        f"{training_params.cats_dogs_checkpoints_dir}/cats_dogs_resnet18_weights.pt",
    )
    logger.info("ResNet weights saved")
