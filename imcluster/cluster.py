"""Clustering operations for image feature vectors."""

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike
from rich.console import Console
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)

from .io import ImclusterIO

console = Console()


class ClusteringAlgorithm(str, Enum):
    """Supported clustering algorithms."""

    SPECTRAL = "spectral"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"
    HIERARCHICAL = "hierarchical"


def cluster(
    imcluster_io: ImclusterIO,
    feature_vectors: ArrayLike,
    algorithm: ClusteringAlgorithm | str = ClusteringAlgorithm.SPECTRAL,
    n_clusters: int = 20,
    dbscan_eps: float = 0.5,
    min_samples: int = 2,
    force: bool = False,
) -> None:
    """Assign images to clusters and cache their labels.

    Args:
        imcluster_io: Image collection and its persisted result table.
        feature_vectors: Feature matrix with one row per image.
        algorithm: Clustering algorithm to use.
        n_clusters: Number of clusters requested by fixed-count algorithms.
        dbscan_eps: Maximum cosine distance between DBSCAN neighbours.
        min_samples: Minimum neighbourhood or cluster size for density methods.
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

    cluster_column = f"{algorithm.value}_cluster"
    if imcluster_io.has_column(cluster_column) and not force:
        console.print(
            f"[green]Using cached {algorithm.value} clusters:[/green] "
            f"loaded {len(imcluster_io.images)} assignments from "
            f"'{imcluster_io.output}'."
        )
        return

    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    fixed_count = {
        ClusteringAlgorithm.SPECTRAL,
        ClusteringAlgorithm.KMEANS,
        ClusteringAlgorithm.AGGLOMERATIVE,
        ClusteringAlgorithm.HIERARCHICAL,
    }
    if algorithm in fixed_count:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if n_clusters > len(vectors):
            raise ValueError("n_clusters cannot exceed the number of images")

    if algorithm is ClusteringAlgorithm.SPECTRAL:
        clustering = SpectralClustering(n_clusters=n_clusters, random_state=0)
    elif algorithm is ClusteringAlgorithm.KMEANS:
        clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    elif algorithm is ClusteringAlgorithm.AGGLOMERATIVE:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm is ClusteringAlgorithm.HIERARCHICAL:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
    elif algorithm is ClusteringAlgorithm.HDBSCAN:
        if min_samples < 2:
            raise ValueError("min_samples must be at least 2 for HDBSCAN")
        clustering = HDBSCAN(
            min_cluster_size=min_samples,
            min_samples=min_samples,
            metric="cosine",
            algorithm="brute",
            copy=True,
        )
    else:
        if dbscan_eps <= 0:
            raise ValueError("dbscan_eps must be greater than 0")
        clustering = DBSCAN(
            eps=dbscan_eps,
            min_samples=min_samples,
            metric="cosine",
        )

    console.print(
        f"[cyan]Running {algorithm.value} clustering:[/cyan] "
        f"assigning {len(imcluster_io.images)} images."
    )
    with console.status(
        f"[cyan]Computing {algorithm.value} cluster assignments...[/cyan]"
    ):
        labels = clustering.fit_predict(vectors)
        imcluster_io.save_column(cluster_column, labels)
    console.print(
        f"[green]Cached {algorithm.value} clusters:[/green] "
        f"wrote {len(labels)} assignments to '{imcluster_io.output}'."
    )
