import torch
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
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
