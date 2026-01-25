import pandas as pd
from pathlib import Path


def generate_metadata(root_dir, output_file):
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
