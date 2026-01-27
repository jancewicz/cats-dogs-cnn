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
from models.predict import (
    process_image,
    predict_image,
)
from model_evaluation.model_evaluator import NNEvaluator
from configs.training_params import TrainingParams
from utils.device import get_device
from loguru import logger

CATS_DOGS_CHECKPOINTS_DIR = os.getenv("CATS_DOGS_CHECKPOINTS_DIR")


def main() -> None:
    load_dotenv()
    hermes_img = os.getenv("HERMES_JPG_DIR")
    visualize_batch(train_set_dataloader, n_images=4)

    # Model hyperparameters used for training are in README
    # AlexNet model
    loaded_weights_alex_net = torch.load(
        f=f"{TrainingParams.DEFAULT_CHECKPOINTS_DIR}/cats_dogs_alexNet_weights.pt",
        weights_only=True,
    )
    loaded_conv_nn = ConvNNCatsDogsClassifier().to(device=get_device())
    loaded_conv_nn.load_state_dict(loaded_weights_alex_net)

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(get_device())
    nn_evaluator = NNEvaluator(
        model=loaded_conv_nn,
        data_loader=valid_set_dataloader,
        metric_fn=accuracy,
        device=get_device(),
    )
    evaluation = nn_evaluator.evaluate()
    logger.info(f"Accuracy on validation set with AlexNet: {evaluation}")

    class_map = {0: "Cat", 1: "Dog"}

    hermes_img_tensor = process_image(hermes_img, size=transform_config.img_size)
    pred, proba = predict_image(loaded_conv_nn, hermes_img_tensor)
    confidence = proba[0][pred].item()

    logger.info(f"Przewidywana klasa: {class_map[pred.item()]}, pewność: {confidence}")


if __name__ == "__main__":
    main()
