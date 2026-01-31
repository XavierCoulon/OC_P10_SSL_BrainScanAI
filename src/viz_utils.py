import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from torchvision.utils import make_grid
from sklearn.metrics import classification_report


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


def plot_projections(projections, labels, title="Projection 2D", palette=None):
    """
    Visualise une projection 2D des données avec coloration par label.

    Args:
        projections: Array (N, 2) avec les coordonnées 2D
        labels: Array (N,) ou Series avec les labels/classes
        title: Titre du graphique
        palette: Palette de couleurs (défaut: viridis)
    """
    plt.figure(figsize=(10, 8))

    if palette is None:
        palette = "viridis"

    sns.scatterplot(
        x=projections[:, 0],
        y=projections[:, 1],
        hue=labels,
        palette=palette,
        alpha=0.6,
        s=50,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
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


def plot_classification_report(
    y_true, y_pred, title, f1_score=None, target_names=["Normal", "Cancer"]
):
    """
    Visualise le rapport de classification sous forme de heatmap.

    Args:
        y_true: Labels vrais
        y_pred: Prédictions du modèle
        title: Titre du graphique
        f1_score: F1-Score global (optionnel, affiché dans le titre)
        target_names: Noms des classes
    """
    # Génération du rapport de classification
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    # Conversion en DataFrame (précision, recall, f1-score)
    df_report = pd.DataFrame(report_dict).iloc[:-1, :3].T

    # Création de la heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        df_report, annot=True, cmap="RdYlGn", fmt=".3f", cbar=False, vmin=0, vmax=1
    )

    # Titre avec F1-Score si fourni
    if f1_score is not None:
        title = f"{title}\n(F1-Score Global : {f1_score:.4f})"

    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
