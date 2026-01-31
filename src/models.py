import torch
import torch.nn as nn
from torchvision import models
from typing import Union


def get_brainscan_model(num_classes=2, device: Union[str, torch.device] = "cpu"):
    """Prépare un modèle ResNet50 pour la classification d'images de scanners cérébraux.

    Cette fonction charge une architecture ResNet50 avec des poids pré-entraînés
    sur ImageNet. La couche de classification finale (fully connected) est
    remplacée par une nouvelle couche linéaire pour s'adapter au nombre de
    classes spécifié (par défaut, 2 pour une classification binaire).

    Le modèle est ensuite déplacé vers le périphérique de calcul spécifié (CPU ou
    GPU).

    Args:
        num_classes (int, optional): Le nombre de classes pour la couche de
            classification. Défaut à 2.
        device (str or torch.device, optional): Le périphérique sur lequel
            placer le modèle ('cpu' ou 'cuda'). Défaut à 'cpu'.

    Returns:
        torch.nn.Module: Le modèle ResNet50 modifié, prêt pour l'entraînement
            ou l'inférence.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Remplacement de la couche FC pour nos 2 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(device)
