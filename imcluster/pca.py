"""Principal-component analysis for image feature vectors."""

from numpy.typing import ArrayLike
from rich.console import Console
from sklearn.decomposition import PCA

from .io import ImclusterIO

console = Console()


def fit_pca(
    imcluster_io: ImclusterIO,
    feature_vectors: ArrayLike,
    force: bool = False,
) -> None:
    """Calculate and cache the first two principal components.

    Args:
        imcluster_io: Image collection and its persisted result table.
        feature_vectors: Feature matrix with one row per image.
        force: Recompute PCA even when both component columns already exist.
    """
    missing_components = not imcluster_io.has_column(
        "pca0"
    ) or not imcluster_io.has_column("pca1")
    if missing_components or force:
        console.print("Performing PCA")
        pca = PCA(n_components=2)
        feature_vectors_2D = pca.fit(feature_vectors).transform(feature_vectors)

        imcluster_io.save_column("pca0", feature_vectors_2D[:, 0], autosave=False)
        imcluster_io.save_column("pca1", feature_vectors_2D[:, 1], autosave=True)
    else:
        console.print("Using precomputed PCA")
