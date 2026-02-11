from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import numpy as np


def infer_cluster_labels(predictions_all, labels_strong, mask_strong):
    """
    Auto-mapping des clusters basé sur les images labellisées.
    Retourne le mapping cluster_id → label réel.
    """
    predictions_strong = predictions_all[mask_strong]
    # confusion_matrix(y_true, y_pred) où y_true=labels_strong, y_pred=predictions_strong
    # Forme : (n_labels_classes, n_clusters)
    confusion = confusion_matrix(labels_strong, predictions_strong)

    label_mapping = {}
    # Itère sur les clusters (colonnes de la matrice)
    for cluster_id in range(confusion.shape[1]):
        votes = confusion[:, cluster_id]  # Votes des labels pour ce cluster
        majority_label = np.argmax(votes)
        label_mapping[cluster_id] = majority_label

    return label_mapping


def compare_clustering_methods(features_pca, df):
    """
    Cluster sur TOUTES les images (1506), puis auto-map les clusters
    selon les 100 images labellisées.
    """
    mask = df["label"] != -1
    true_labels = df[mask]["label"].values
    results_labels = {}
    results_scores = {}

    # ============================================
    # K-Means
    # ============================================
    model = KMeans(n_clusters=2, n_init=10, random_state=42)
    predictions_all = model.fit_predict(features_pca)
    label_mapping = infer_cluster_labels(predictions_all, true_labels, mask)
    predictions_mapped = np.array([label_mapping[p] for p in predictions_all])
    predictions_mapped_strong = np.array(
        [label_mapping[p] for p in predictions_all[mask]]
    )
    if len(set(label_mapping.values())) < 2:
        print(
            "⚠️  K-Means : Les deux clusters pointent vers la même étiquette. Le clustering n'est pas discriminant."
        )
    results_labels["K-Means"] = predictions_mapped
    results_scores["K-Means"] = adjusted_rand_score(
        true_labels, predictions_mapped_strong
    )

    # ============================================
    # GMM (Gaussian Mixture Model)
    # ============================================
    model = GaussianMixture(n_components=2, random_state=42)
    predictions_all = model.fit_predict(features_pca)
    label_mapping = infer_cluster_labels(predictions_all, true_labels, mask)
    predictions_mapped = np.array([label_mapping[p] for p in predictions_all])
    predictions_mapped_strong = np.array(
        [label_mapping[p] for p in predictions_all[mask]]
    )
    if len(set(label_mapping.values())) < 2:
        print(
            "⚠️  GMM : Les deux clusters pointent vers la même étiquette. Le clustering n'est pas discriminant."
        )
    results_labels["GMM"] = predictions_mapped
    results_scores["GMM"] = adjusted_rand_score(true_labels, predictions_mapped_strong)

    # ============================================
    # Agglomératif (Hierarchical Clustering)
    # ============================================
    model = AgglomerativeClustering(n_clusters=2)
    predictions_all = model.fit_predict(features_pca)
    label_mapping = infer_cluster_labels(predictions_all, true_labels, mask)
    predictions_mapped = np.array([label_mapping[p] for p in predictions_all])
    predictions_mapped_strong = np.array(
        [label_mapping[p] for p in predictions_all[mask]]
    )
    if len(set(label_mapping.values())) < 2:
        print(
            "⚠️  Agglomératif : Les deux clusters pointent vers la même étiquette. Le clustering n'est pas discriminant."
        )
    results_labels["Agglomératif"] = predictions_mapped
    results_scores["Agglomératif"] = adjusted_rand_score(
        true_labels, predictions_mapped_strong
    )

    return results_labels, results_scores
