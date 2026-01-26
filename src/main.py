import os

import torch
import torchmetrics
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloaders.dataloaders import (
    STD_NORMALIZATION_VALUES,
    MEAN_NORMALIZATION_VALUES,
    split_and_transform_trainset,
    cats_dogs_training_set_dir,
    SIZE,
)
from models.cats_dogs_conv_nn import (
    ConvNNCatsDogsClassifier,
)
from models.predict import (
    process_image,
    predict_image,
)
from model_evaluation.model_evaluator import NNEvaluator
from utils.device import get_device
from loguru import logger

CATS_DOGS_CHECKPOINTS_DIR = os.getenv("CATS_DOGS_CHECKPOINTS_DIR")


def visualize_batch(data_loader):
    # batch has shape (images, labes)
    batch = next(iter(data_loader))
    # images_batch has shape [4, 3, 224, 224]
    images_batch, labels_batch = batch[:4]

    # convert std and mean lists to tensor + apply broadcasting for new tensor to match shape
    std = torch.tensor(STD_NORMALIZATION_VALUES).view(1, 3, 1, 1)
    mean = torch.tensor(MEAN_NORMALIZATION_VALUES).view(1, 3, 1, 1)

    denormalized_imgs = (images_batch * std) + mean
    denormalized_imgs = torch.clamp(denormalized_imgs, 0, 1)

    # move height and width to the middle, channel goes last, batch position stays the same
    permuted_images = denormalized_imgs.permute(0, 2, 3, 1)
    class_names = data_loader.dataset.data_subset.dataset.classes

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i in range(4):
        axes[i].imshow(permuted_images[i].numpy())
        label_idx = labels_batch[i].item()
        axes[i].set_title(class_names[label_idx])
        axes[i].axis("off")
    plt.show()


def main() -> None:
    load_dotenv()

    train_set, valid_set = split_and_transform_trainset(cats_dogs_training_set_dir)

    train_set_dataloader = DataLoader(train_set, 32, shuffle=True)
    valid_set_dataloader = DataLoader(valid_set, 32, shuffle=True)

    visualize_batch(train_set_dataloader)
    hermes_img = os.getenv("HERMES_JPG_DIR")

    # Model hyperparameters used for training are in README
    loaded_weights = torch.load(
        f=f"{CATS_DOGS_CHECKPOINTS_DIR}/cats_dogs_alexNet_weights.pt", weights_only=True
    )
    loaded_conv_nn = ConvNNCatsDogsClassifier().to(device=get_device())
    loaded_conv_nn.load_state_dict(loaded_weights)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(get_device())

    nn_evaluator = NNEvaluator(
        model=loaded_conv_nn,
        data_loader=valid_set_dataloader,
        metric_fn=accuracy,
        device=get_device(),
    )
    evaluation = nn_evaluator.evaluate()
    logger.info(f"Accuracy on validation set: {evaluation}")

    class_map = {0: "Cat", 1: "Dog"}

    hermes_img_tensor = process_image(hermes_img, size=SIZE)
    pred, proba = predict_image(loaded_conv_nn, hermes_img_tensor)
    confidence = proba[0][pred].item()

    logger.info(f"Przewidywana klasa: {class_map[pred.item()]}, pewność: {confidence}")


if __name__ == "__main__":
    main()
