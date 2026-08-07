"""Clustering operations for image feature vectors."""

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike
from rich.console import Console
from sklearn.cluster import DBSCAN, SpectralClustering

from .io import ImclusterIO

console = Console()


class ClusteringAlgorithm(str, Enum):
    """Supported clustering algorithms."""

    SPECTRAL = "spectral"
    DBSCAN = "dbscan"


def cluster(
    imcluster_io: ImclusterIO,
    feature_vectors: ArrayLike,
    algorithm: ClusteringAlgorithm | str = ClusteringAlgorithm.SPECTRAL,
    n_clusters: int = 20,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 2,
    force: bool = False,
) -> None:
    """Assign images to clusters and cache their labels.

    Args:
        imcluster_io: Image collection and its persisted result table.
        feature_vectors: Feature matrix with one row per image.
        algorithm: Clustering algorithm, either ``SPECTRAL`` or ``DBSCAN``.
        n_clusters: Number of clusters requested for spectral clustering.
        dbscan_eps: Maximum cosine distance between DBSCAN neighbours.
        dbscan_min_samples: Minimum DBSCAN neighbourhood size.
        force: Recompute labels even when the appropriate column already exists.

    Raises:
        ValueError: If the algorithm or its parameters are invalid.
    """
    try:
        algorithm = (
            algorithm
            if isinstance(algorithm, ClusteringAlgorithm)
            else ClusteringAlgorithm(algorithm.lower())
        )
    except (AttributeError, ValueError) as error:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}") from error

    vectors = np.asarray(feature_vectors)
    if vectors.ndim != 2 or len(vectors) != len(imcluster_io.images):
        raise ValueError("feature_vectors must contain one row per image")

    if algorithm is ClusteringAlgorithm.SPECTRAL:
        if not imcluster_io.has_column("spectral_cluster") or force:
            if n_clusters < 2:
                raise ValueError("n_clusters must be at least 2")
            if n_clusters > len(vectors):
                raise ValueError("n_clusters cannot exceed the number of images")
            console.print("spectral clustering")
            clustering = SpectralClustering(n_clusters=n_clusters, random_state=0)
            clustering.fit(vectors)
            imcluster_io.save_column("spectral_cluster", clustering.labels_)
        else:
            console.print("Using precomputed spectral clusters")

    else:
        if not imcluster_io.has_column("dbscan_cluster") or force:
            if dbscan_eps <= 0:
                raise ValueError("dbscan_eps must be greater than 0")
            if dbscan_min_samples < 1:
                raise ValueError("dbscan_min_samples must be at least 1")
            console.print("dbscan clustering")
            clustering = DBSCAN(
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                metric="cosine",
            )
            clustering.fit(vectors)
            imcluster_io.save_column("dbscan_cluster", clustering.labels_)
        else:
            console.print("Using precomputed dbscan clusters")
