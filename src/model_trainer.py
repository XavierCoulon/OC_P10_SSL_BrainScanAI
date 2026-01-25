import torch.nn as nn
from torchvision import models

def get_brainscan_model(num_classes=2):
    # Charger ResNet50 avec les poids ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # On débloque les dernières couches pour le fine-tuning
    for param in model.parameters():
        param.requires_grad = True # Ou False pour les premières couches seulement
        
    # On remplace la couche Identity de l'étape 2 par un vrai classifieur
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
	