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

## 🚀 Installation

### Prérequis

- Python >= 3.12
- pip ou conda

### Installation des dépendances

```bash
make data
```

Ou manuellement :

```bash
pip install -e .
```

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
