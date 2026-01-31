import torch.nn as nn
from torchvision import models


def get_brainscan_model(num_classes=2):
    """Crée un modèle ResNet50 pré-entraîné pour la classification de scanners cérébraux.

    Cette fonction charge une architecture ResNet50 avec des poids pré-entraînés
    sur ImageNet. Les paramètres du modèle sont débloqués pour permettre le
    fine-tuning. La couche de classification finale est remplacée par une nouvelle
    couche linéaire adaptée au nombre de classes spécifié.

    Args:
        num_classes (int, optional): Le nombre de classes de sortie pour le
            classifieur. Défaut à 2 (par exemple, 'normal' et 'cancer').

    Returns:
        torch.nn.Module: Le modèle ResNet50 modifié et prêt pour le fine-tuning.
    """
    # Charger ResNet50 avec les poids ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # On débloque les dernières couches pour le fine-tuning
    for param in model.parameters():
        param.requires_grad = True  # Ou False pour les premières couches seulement

    # On remplace la couche Identity de l'étape 2 par un vrai classifieur
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model