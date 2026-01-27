import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as transforms_v2

from dataloaders.dataloaders import transform_config


def process_image(new_img_path: str, size: tuple[int, int]):
    image = Image.open(fp=new_img_path).convert("RGB")

    transform = transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.Resize(size=size, antialias=True),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(
                # Same values as in transformations made on training and validation set
                mean=transform_config.mean_normalization_values,
                std=transform_config.std_normalization_values,
            ),
        ]
    )
    image_transformed = transform(image).unsqueeze(0)

    return image_transformed


def predict_image(model: nn.Module, X_new: torch.Tensor):
    device = next(model.parameters()).device
    X_new = X_new.to(device)

    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_new)

    y_pred = y_pred_logits.argmax(dim=1)
    y_proba = F.softmax(y_pred_logits, dim=1)

    return y_pred, y_proba
