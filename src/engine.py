import torch
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Entraîne le modèle pour une seule époque.

    Args:
        model (torch.nn.Module): Le modèle à entraîner.
        loader (torch.utils.data.DataLoader): Le DataLoader pour les données d'entraînement.
        optimizer (torch.optim.Optimizer): L'optimiseur à utiliser.
        criterion (torch.nn.Module): La fonction de perte.
        device (torch.device): Le périphérique sur lequel effectuer les calculs.

    Returns:
        float: La perte moyenne pour l'époque.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate_model(model, loader, device, target_names=["Normal", "Cancer"]):
    """Évalue les performances du modèle sur un ensemble de données.

    Args:
        model (torch.nn.Module): Le modèle à évaluer.
        loader (torch.utils.data.DataLoader): Le DataLoader pour les données d'évaluation.
        device (torch.device): Le périphérique sur lequel effectuer les calculs.
        target_names (list, optional): Les noms des classes cibles. 
            Défaut à ["Normal", "Cancer"].

    Returns:
        tuple: Un tuple contenant le score F1 et le rapport de classification.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        outputs = model(inputs.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=target_names)
    score = f1_score(all_labels, all_preds)
    return score, report
