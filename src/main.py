import os

import torch
import torchmetrics
from dotenv import load_dotenv
from dataloaders.dataloaders import (
    valid_set_dataloader,
    train_set_dataloader,
    transform_config,
)
from dataloaders.visualise_batch import visualize_batch
from models.cats_dogs_alex_net import (
    ConvNNCatsDogsClassifier,
)
from models.cats_dogs_resnet import ResNetCatsDogsClassifier
from models.predict import (
    process_image,
    predict_image,
)
from model_evaluation.model_evaluator import evaluate_on_valid_set
from training.load_checkpoints import load_model_checkpoint
from utils.device import get_device
from loguru import logger

CATS_DOGS_CHECKPOINTS_DIR = os.getenv("CATS_DOGS_CHECKPOINTS_DIR")


def main() -> None:
    load_dotenv()
    hermes_img = os.getenv("HERMES_JPG_DIR")
    visualize_batch(train_set_dataloader, n_images=4)

    # AlexNet
    # Model hyperparameters used for training are in README
    alex_net: ConvNNCatsDogsClassifier = ConvNNCatsDogsClassifier().to(
        device=get_device()
    )
    loaded_alex_net = load_model_checkpoint(
        alex_net, checkpoints_file_path="cats_dogs_alexNet_weights.pt"
    )

    # ResNet - pretrained ResNet18 version
    resnet: ResNetCatsDogsClassifier = ResNetCatsDogsClassifier(pretrained=False).to(
        device=get_device()
    )
    loaded_resnet = load_model_checkpoint(
        resnet, checkpoints_file_path="cats_dogs_resnet18_weights.pt"
    )

    # Metric function
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(get_device())

    # Compute acc on validation set for alexnet
    alex_net_acc = evaluate_on_valid_set(
        loaded_alex_net, valid_set_dataloader, accuracy
    )
    logger.info(f"Accuracy on validation set with AlexNet: {alex_net_acc}")
    # Compute acc on validation set for resnet
    resnet_acc = evaluate_on_valid_set(loaded_resnet, valid_set_dataloader, accuracy)
    logger.info(f"Accuracy on validation set with ResNet18: {resnet_acc}")

    class_map: dict[int, str] = {0: "Cat", 1: "Dog"}

    hermes_img_tensor: torch.Tensor = process_image(
        hermes_img, size=transform_config.img_size
    )
    pred, proba = predict_image(loaded_resnet, hermes_img_tensor)
    confidence = proba[0][pred].item()

    logger.info(f"Przewidywana klasa: {class_map[pred.item()]}, pewność: {confidence}")


if __name__ == "__main__":
    main()
