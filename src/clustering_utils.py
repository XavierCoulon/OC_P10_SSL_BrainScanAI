from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score


def compare_clustering_methods(features_pca, df):
    """Calcule les labels et les scores ARI pour 3 méthodes."""
    mask = df["label"] != -1
    true_labels = df[mask]["label"]
    results_labels = {}
    results_scores = {}

    # 1. K-Means
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    results_labels["K-Means"] = km.fit_predict(features_pca)
    results_scores["K-Means"] = adjusted_rand_score(
        true_labels, results_labels["K-Means"][mask]
    )

    # 2. GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    results_labels["GMM"] = gmm.fit_predict(features_pca)
    results_scores["GMM"] = adjusted_rand_score(
        true_labels, results_labels["GMM"][mask]
    )

    # 3. Agglomératif
    agglo = AgglomerativeClustering(n_clusters=2)
    results_labels["Agglomératif"] = agglo.fit_predict(features_pca)
    results_scores["Agglomératif"] = adjusted_rand_score(
        true_labels, results_labels["Agglomératif"][mask]
    )

    return results_labels, results_scores
