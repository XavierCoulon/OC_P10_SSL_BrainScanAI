import torch
import torch.nn as nn
from torchvision import models
from typing import Union


def get_brainscan_model(num_classes=2, device: Union[str, torch.device] = "cpu"):
    """Prépare un ResNet50 pour la classification binaire."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Remplacement de la couche FC pour nos 2 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(device)
