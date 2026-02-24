# Brain Cancer Detection - Semi-Supervised Learning

Un projet d'apprentissage semi-supervisé pour la détection de tumeurs au cerveau à partir d'images IRM.

## 📋 Aperçu

Ce projet utilise des techniques d'apprentissage semi-supervisé pour classifier des images de scanners cérébraux en deux catégories :

- **Normal** (0) : Cerveaux sains
- **Cancer** (1) : Cerveaux atteints de tumeurs

Le projet exploite à la fois des données étiquetées et non-étiquetées pour améliorer les performances du modèle.

## 🏗️ Architecture du Projet

```
src/
├── make_data.py           # Extraction et préparation des métadonnées
├── feature_extractor.py   # Extraction des features avec ResNet50
├── models.py              # Définition du modèle de classification
├── model_trainer.py       # Entraînement et évaluation
├── engine.py              # Moteur d'exécution principal
├── data_utils.py          # Utilitaires de gestion des données
├── clustering_utils.py    # Utilitaires de clustering
├── stats_utils.py         # Statistiques et analyses
└── viz_utils.py           # Visualisations

notebooks/
├── 01_eda_exploration.ipynb           # Exploration des données
├── 02_clustering_analysis.ipynb       # Analyse de clustering
├── 03_non_supervised_approach.ipynb   # Approche non-supervisée
└── 04_semi_supervised_approach.ipynb  # Approche semi-supervisée

data/
├── raw/                   # Données brutes (images)
├── processed/             # Features extraites
└── *.csv                  # Métadonnées
```

## � Description des scripts (`src/`)

| Fichier                | Rôle                                                                             |
| ---------------------- | -------------------------------------------------------------------------------- |
| `make_data.py`         | Extraction des métadonnées CSV depuis le répertoire d'images (labels et chemins) |
| `feature_extractor.py` | Extraction des vecteurs de features via ResNet50 pré-entraîné                    |
| `models.py`            | Définition du modèle ResNet50 adapté (2 classes pour classification binaire)     |
| `model_trainer.py`     | Fine-tuning et entraînement du modèle sur les données labellisées                |
| `engine.py`            | Boucles d'entraînement/validation (`train_one_epoch`, `evaluate`)                |
| `data_utils.py`        | Dataset PyTorch (`BrainScanDataset`) et transformations d'images                 |
| `clustering_utils.py`  | Algorithmes de clustering (K-Means, Agglomérative, GMM, DBSCAN) et comparaison   |
| `stats_utils.py`       | Calcul de moyennes et écart-types pour normalisation                             |
| `viz_utils.py`         | Visualisations : grilles d'images, matrices de confusion, réductions PCA         |

## �🚀 Installation

### Prérequis

- Python >= 3.12
- uv

### Installation des dépendances

```bash
uv sync
```

Pour exécuter un script ou un notebook :

```bash
uv run python script.py
```

## 🔧 Avant de lancer les notebooks

Pour que les notebooks fonctionnent correctement, vous devez d'abord exécuter les scripts de préparation dans cet ordre :

```bash
# 1. Générer les métadonnées (CSV) à partir des images
make data

# 2. Extraire les features ResNet50 (nécessaire pour clustering et modélisation)
make features
```

**Cela générera :**

- `data/metadata.csv` : Index des images avec leurs labels
- `data/processed/features_resnet.npy` : Vecteurs de features (2048 dimensions)

Ensuite, les notebooks pourront être exécutés dans l'ordre souhaité.

## 📊 Dépendances principales

- **PyTorch** : Framework pour le deep learning
- **torchvision** : Modèles pré-entraînés (ResNet50)
- **scikit-learn** : Machine learning classique
- **pandas** : Manipulation de données
- **matplotlib** : Visualisations
- **numpy** : Calculs numériques

## 🔄 Flux de travail

1. **Préparation des données** : Extraction des métadonnées et organisation des images
2. **Extraction de features** : Utilisation de ResNet50 pré-entraîné pour extraire des vecteurs de features
3. **Clustering** : Analyse non-supervisée des données
4. **Modélisation** : Entraînement avec apprentissage semi-supervisé
5. **Évaluation** : Analyse des performances

## 📝 Fichiers de données

- `metadata.csv` : Métadonnées des images étiquetées et non-étiquetées
- `features_resnet.npy` : Features extraites (ResNet50)
- `avec_labels/` : Images avec labels (normal/cancer)
- `sans_label/` : Images sans labels (pour l'apprentissage semi-supervisé)

## 📚 Notebooks

Les notebooks explorent différentes approches :

1. **EDA** : Exploration initiale du dataset
2. **Clustering** : Analyse des patterns non-supervisée
3. **Non-supervisé** : Apprentissage sans labels
4. **Semi-supervisé** : Combinaison de données labellisées et non-labellisées
