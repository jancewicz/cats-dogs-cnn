import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

from utils.device import get_device


class ResNetCatsDogsClassifier(nn.Module):
    """
    Cats and dogs classifier class based on resnet18 neural network architecture
    """

    def __init__(self, pretrained: bool):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.resnet = resnet18(weights)
        else:
            self.resnet = resnet18(weights=None)

        input_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(input_features, 2)

    def forward(self, X: torch.Tensor):
        return self.resnet(X)


device = get_device()

resnet_model = ResNetCatsDogsClassifier(pretrained=True)
resnet_model = resnet_model.to(device)
