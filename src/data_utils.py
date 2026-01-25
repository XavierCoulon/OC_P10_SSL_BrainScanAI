import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def get_img_info(path):
    """Extrait la résolution et le mode colorimétrique d'une image."""
    try:
        with Image.open(path) as img:
            return img.size, img.mode
    except Exception as e:
        return None, str(e)


class BrainScanDataset(Dataset):
    def __init__(self, data, transform=None, label_col="label"):
        """
        Args:
            data (str or pd.DataFrame): Chemin vers le CSV OU directement un DataFrame.
            transform (callable, optional): Transformations à appliquer.
            label_col (str): Nom de la colonne pour les labels.
        """
        # --- LA CORRECTION EST ICI ---
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        # -----------------------------

        self.transform = transform
        self.label_col = label_col

        # Sécurité pour les labels
        self.df[self.label_col] = self.df[self.label_col].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        label = self.df.iloc[idx][self.label_col]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
