from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler


from rich.console import Console

console = Console()

from .io import ImclusterIO


def cluster(
    imcluster_io: ImclusterIO,
    feature_vectors,
    algorithm="SPECTRAL",
    n_clusters: int = 20,
    force: bool = False,
):

    algorithm = algorithm.upper()
    if algorithm == "SPECTRAL":
        if not imcluster_io.has_column("spectral_cluster") or force:
            console.print("spectral clustering")
            clustering = SpectralClustering(n_clusters=n_clusters)
            # scaled_features = StandardScaler().fit_transform(feature_vectors)
            clustering.fit(feature_vectors)
            imcluster_io.save_column("spectral_cluster", clustering.labels_)
        else:
            console.print("Using precomputed spectral clusters")

    elif algorithm == "DBSCAN":
        if not imcluster_io.has_column("dbscan_cluster") or force:
            console.print("dbscan clustering")
            clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
            scaled_features = StandardScaler().fit_transform(feature_vectors)
            clustering.fit(scaled_features)
            imcluster_io.save_column("dbscan_cluster", clustering.labels_)
        else:
            console.print("Using precomputed dbscan clusters")
    else:
        raise Exception(f"Cannot understand algorithm: {algorithm}")
