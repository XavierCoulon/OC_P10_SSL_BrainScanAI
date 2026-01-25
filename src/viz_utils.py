import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision.utils import make_grid


def plot_grid(df, label, title, n=4):
    """Affiche une grille d'images filtrées par label depuis un DataFrame."""
    subset = df[df["label"] == label]
    if len(subset) == 0:
        print(f"Aucune donnée pour le label {label}")
        return

    sample = subset.sample(min(n, len(subset)))
    plt.figure(figsize=(15, 5))

    for i, (idx, row) in enumerate(sample.iterrows()):
        plt.subplot(1, n, i + 1)
        img = Image.open(row["path"])
        plt.imshow(img, cmap="gray")
        plt.title(f"ID: {row['filename'][:8]}")
        plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.show()


def show_batch_grid(loader, n_images=16):
    """Affiche une grille de batch pour validation visuelle rapide."""
    images, labels = next(iter(loader))
    grid = make_grid(images[:n_images], nrow=4, padding=2, normalize=True)

    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0))  # Passage de (C,H,W) à (H,W,C)
    plt.axis("off")
    plt.show()


def plot_clustering_comparison(projections, df, labels_dict, scores):
    """Affiche la vérité terrain face aux 3 méthodes de clustering."""
    methods = ["Vérité Terrain"] + list(labels_dict.keys())
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for i, method in enumerate(methods):
        ax = axes[i]
        if method == "Vérité Terrain":
            hue = df["label"].replace({-1: "Inconnu", 0: "Normal", 1: "Cancer"})
            palette = {"Inconnu": "lightgrey", "Normal": "blue", "Cancer": "red"}
            title = "Vérité Terrain (Labels Forts)"
        else:
            hue = labels_dict[method]
            palette = "viridis"
            title = f"{method}\n(ARI: {scores[method]:.3f})"

        sns.scatterplot(
            x=projections[:, 0],
            y=projections[:, 1],
            hue=hue,
            palette=palette,
            ax=ax,
            alpha=0.5,
        )
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
