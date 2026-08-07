"""Clustering operations for image feature vectors."""

from numpy.typing import ArrayLike
from rich.console import Console
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler

from .io import ImclusterIO

console = Console()


def cluster(
    imcluster_io: ImclusterIO,
    feature_vectors: ArrayLike,
    algorithm: str = "SPECTRAL",
    n_clusters: int = 20,
    force: bool = False,
) -> None:
    """Assign images to clusters and cache their labels.

    Args:
        imcluster_io: Image collection and its persisted result table.
        feature_vectors: Feature matrix with one row per image.
        algorithm: Clustering algorithm, either ``SPECTRAL`` or ``DBSCAN``.
        n_clusters: Number of clusters requested for spectral clustering.
        force: Recompute labels even when the appropriate column already exists.

    Raises:
        Exception: If ``algorithm`` is not supported.
    """

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
