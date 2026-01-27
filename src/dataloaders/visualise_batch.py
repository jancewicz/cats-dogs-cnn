import torch
import matplotlib.pyplot as plt

from dataloaders.dataloaders import transform_config


def visualize_batch(data_loader, n_images):
    # batch has shape (images, labes)
    batch = next(iter(data_loader))
    # images_batch has shape [n_images, 3, 222, 222]
    images_batch, labels_batch = batch[:n_images]

    # convert std and mean lists to tensor + apply broadcasting for new tensor to match shape
    std = torch.tensor(transform_config.std_normalization_values).view(1, 3, 1, 1)
    mean = torch.tensor(transform_config.mean_normalization_values).view(1, 3, 1, 1)

    denormalized_imgs = (images_batch * std) + mean
    denormalized_imgs = torch.clamp(denormalized_imgs, 0, 1)

    # move height and width to the middle, channel goes last, batch position stays the same
    permuted_images = denormalized_imgs.permute(0, 2, 3, 1)
    class_names = data_loader.dataset.data_subset.dataset.classes

    fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
    for i in range(n_images):
        axes[i].imshow(permuted_images[i].numpy())
        label_idx = labels_batch[i].item()
        axes[i].set_title(class_names[label_idx])
        axes[i].axis("off")
    plt.show()
