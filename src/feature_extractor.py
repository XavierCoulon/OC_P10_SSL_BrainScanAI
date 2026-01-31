import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from data_utils import BrainScanDataset


def extract_features(csv_path, output_path):
    """Extrait les caractéristiques d'images à l'aide d'un modèle ResNet50 pré-entraîné.

    Cette fonction charge un ensemble de données d'images spécifié par un fichier CSV,
    puis utilise un modèle ResNet50 (pré-entraîné sur ImageNet) pour extraire
    un vecteur de caractéristiques pour chaque image. La dernière couche du modèle
    est retirée pour obtenir le vecteur avant la classification.

    Les caractéristiques extraites sont ensuite sauvegardées dans un fichier .npy.

    Args:
        csv_path (str): Le chemin vers le fichier CSV contenant les métadonnées
            des images (notamment le chemin de chaque image).
        output_path (str): Le chemin du fichier .npy de sortie où seront
            sauvegardées les caractéristiques extraites.
    
    Returns:
        None
    """
    # 1. Pipeline de transformation (Standard ImageNet)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 2. Chargement du modèle ResNet50
    # On utilise les poids pré-entraînés
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # On gèle les paramètres (pas d'entraînement ici)
    for param in model.parameters():
        param.requires_grad = False

    # On remplace la dernière couche (classification) par une couche "Identité"
    # pour récupérer le vecteur de 2048 dimensions
    model.fc = nn.Identity()  # type: ignore

    # Passage en mode évaluation et sur GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. DataLoader
    dataset = BrainScanDataset(csv_path, transform=preprocess)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = []

    # 4. Boucle d'extraction
    print(f"🚀 Extraction des features sur {device}...")
    with torch.no_grad():
        for imgs, _ in tqdm(loader):
            imgs = imgs.to(device)
            embeddings = model(imgs)
            features.append(embeddings.cpu().numpy())

    # 5. Sauvegarde
    features_array = np.vstack(features)
    np.save(output_path, features_array)
    print(
        f"✅ Terminé ! Features sauvegardées dans {output_path} (Shape: {features_array.shape})"
    )


if __name__ == "__main__":
    extract_features("data/metadata.csv", "data/processed/features_resnet.npy")
