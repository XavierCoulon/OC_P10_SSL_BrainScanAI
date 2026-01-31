def compute_dataset_stats(loader):
    """Calcule la moyenne et l'écart-type des images d'un DataLoader.

    Cette fonction itère sur toutes les images d'un DataLoader pour calculer
    la moyenne et l'écart-type des valeurs de pixels sur l'ensemble du dataset.
    Ces statistiques sont utiles pour la normalisation des données.

    Args:
        loader (torch.utils.data.DataLoader): Le DataLoader contenant les
            images à analyser. Les images doivent être des tenseurs PyTorch.

    Returns:
        tuple: Un tuple contenant deux tenseurs :
            - Le premier tenseur est la moyenne des canaux de couleur (R, G, B).
            - Le second tenseur est l'écart-type des canaux de couleur (R, G, B).
    """
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std
