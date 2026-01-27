import os

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2
from dotenv import load_dotenv

from configs.transformations_config import TransformationsConfig
from image_transformator.transform_subset import (
    SubsetTransformator,
)
from configs.training_params import TrainingParams

load_dotenv()
# get both train set and test set dirs
cats_dogs_training_set_dir = os.getenv("CATS_DOGS_TRAINSET_DIR")
cats_dogs_test_set_dir = os.getenv("CATS_DOGS_TESTSET_DIR")

transform_config = TransformationsConfig(
    img_size=(222, 222),
    # each value representing one color channel (RGB) - mean values comes from PyTorch docs computed on ImageNet
    mean_normalization_values=[0.485, 0.456, 0.406],
    std_normalization_values=[0.229, 0.224, 0.225],
)

torch.manual_seed(42)


class DataLoaderCreator:
    def __init__(self, transform_config: TransformationsConfig):
        self.img_size = transform_config.img_size
        self.mean_normalization_values = transform_config.mean_normalization_values
        self.std_normalization_values = transform_config.std_normalization_values

    def split_and_transform_trainset(self, cats_dogs_training_set: str):
        generator = torch.Generator().manual_seed(42)

        common_transforms = [
            transforms_v2.ToImage(),
            transforms_v2.Resize(size=self.img_size, antialias=True),
        ]

        # create image processing pipeline for train set with augmentation and valid set with just resizing photos
        train_set_transform = transforms_v2.Compose(
            [
                *common_transforms,
                transforms_v2.RandomHorizontalFlip(0.5),
                transforms_v2.ToDtype(torch.float32, scale=True),
                # values from PyTorch docs
                transforms_v2.Normalize(
                    mean=self.mean_normalization_values,
                    std=self.mean_normalization_values,
                ),
            ]
        )
        valid_set_transform = transforms_v2.Compose(
            [
                *common_transforms,
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(
                    mean=self.mean_normalization_values,
                    std=self.std_normalization_values,
                ),
            ]
        )

        # split training set with 8/2 ratio
        raw_train_dataset = datasets.ImageFolder(cats_dogs_training_set)
        train_set_size = int(0.8 * len(raw_train_dataset))
        valid_set_size = len(raw_train_dataset) - train_set_size

        train_set, valid_set = random_split(
            dataset=raw_train_dataset,
            lengths=[train_set_size, valid_set_size],
            generator=generator,
        )

        train_set_transformed = SubsetTransformator(
            data_subset=train_set, transform=train_set_transform
        )
        valid_set_transformed = SubsetTransformator(
            data_subset=valid_set, transform=valid_set_transform
        )
        return train_set_transformed, valid_set_transformed


dataloader_creator = DataLoaderCreator(transform_config)

train_set, valid_set = dataloader_creator.split_and_transform_trainset(
    cats_dogs_training_set_dir
)
train_set_dataloader = DataLoader(
    train_set, TrainingParams.DEFAULT_BATCH_SIZE, shuffle=True
)
valid_set_dataloader = DataLoader(
    valid_set, TrainingParams.DEFAULT_BATCH_SIZE, shuffle=True
)
