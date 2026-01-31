import pandas as pd
from pathlib import Path


def generate_metadata(root_dir, output_file):
    """Génère un fichier de métadonnées CSV à partir d'un répertoire d'images.

    Cette fonction parcourt un répertoire racine contenant des images de deux types :
    étiquetées et non étiquetées. Les images étiquetées sont classées dans des
    sous-dossiers correspondant à leur label ('normal' ou 'cancer'), tandis que
    les images non étiquetées sont regroupées dans un dossier à part.

    La fonction crée un DataFrame pandas avec les informations de chaque image,
    puis le sauvegarde dans un fichier CSV.

    Args:
        root_dir (str or Path): Le chemin vers le répertoire racine contenant les
            dossiers d'images ('avec_labels', 'sans_label').
        output_file (str or Path): Le chemin du fichier CSV de sortie où seront
            sauvegardées les métadonnées.

    Returns:
        None
    """
    root = Path(root_dir)
    data = []

    # 1. Extraction des données étiquetées
    # On définit le mapping pour transformer le texte en numérique
    label_mapping = {"normal": 0, "cancer": 1}

    for label_name, label_value in label_mapping.items():
        folder = root / "avec_labels" / label_name
        if folder.exists():
            for img_path in folder.glob("*.jpg"):
                data.append(
                    {
                        "path": str(img_path.resolve()),
                        "filename": img_path.name,
                        "label": label_value,
                        "is_labeled": True,
                    }
                )

    # 2. Extraction des données non étiquetées
    unlabeled_folder = root / "sans_label"
    if unlabeled_folder.exists():
        for img_path in unlabeled_folder.glob("*.jpg"):
            data.append(
                {
                    "path": str(img_path.resolve()),
                    "filename": img_path.name,
                    "label": -1,  # Convention pour "absence de label"
                    "is_labeled": False,
                }
            )

    # 3. Création du DataFrame et sauvegarde
    df = pd.DataFrame(data)

    # Création du dossier parent si besoin
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"✅ Metadata généré : {len(df)} images indexées dans {output_file}")
    print(
        f"📊 Stats : {df['is_labeled'].sum()} étiquetées, {(~df['is_labeled']).sum()} non étiquetées."
    )


if __name__ == "__main__":
    # À adapter selon ton arborescence réelle
    DATA_ROOT = "data/raw/mri_dataset_brain_cancer_oc"
    OUTPUT_CSV = "data/metadata.csv"
    generate_metadata(DATA_ROOT, OUTPUT_CSV)
