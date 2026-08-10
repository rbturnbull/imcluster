"""Optional dimensionality reduction for image feature vectors."""

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from .io import ImclusterIO

console = Console()


class ReductionMethod(str, Enum):
    """Supported dimensionality-reduction methods."""

    NONE = "none"
    UMAP = "umap"
    TSNE = "tsne"
    PCA = "pca"


def reduce_dimensions(
    imcluster_io: ImclusterIO,
    feature_vectors: ArrayLike,
    method: ReductionMethod | str = ReductionMethod.NONE,
    dimensions: int = 50,
    force: bool = False,
) -> NDArray[Any]:
    """Reduce feature dimensions and cache the resulting vectors.

    Args:
        imcluster_io: Image collection used to persist reduced vectors.
        feature_vectors: Feature matrix with one row per image.
        method: Reduction algorithm, or ``none`` to retain original vectors.
        dimensions: Requested number of output dimensions.
        force: Recompute an existing reduced-vector cache.

    Returns:
        Original or dimension-reduced feature vectors.

    Raises:
        ValueError: If the method is unsupported or vectors are malformed.
    """
    try:
        method = (
            method
            if isinstance(method, ReductionMethod)
            else ReductionMethod(method.lower())
        )
    except (AttributeError, ValueError) as error:
        raise ValueError(f"Unsupported reduction method: {method}") from error

    vectors = np.asarray(feature_vectors, dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(imcluster_io.images):
        raise ValueError("feature_vectors must contain one row per image")
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")
    if method is ReductionMethod.NONE:
        return vectors

    cache_column = f"reduction_{method.value}_{dimensions}"
    if imcluster_io.has_column(cache_column) and not force:
        console.print(
            f"[green]Using cached {method.value} reduction:[/green] loaded "
            f"{len(vectors)} vectors from '{imcluster_io.output}'."
        )
        return np.asarray(imcluster_io.get_column(cache_column).to_list(), dtype=float)

    n_samples, n_features = vectors.shape
    if method is ReductionMethod.PCA:
        n_components = min(dimensions, n_samples, n_features)
        reducer: Any = PCA(n_components=n_components)
    elif method is ReductionMethod.TSNE:
        n_components = min(dimensions, 3, n_features)
        reducer = TSNE(
            n_components=n_components,
            perplexity=min(30.0, float(n_samples - 1)),
            metric="cosine",
            init="random",
            learning_rate="auto",
            random_state=0,
        )
    else:
        if n_samples < 3:
            raise ValueError("UMAP reduction requires at least three images")
        n_components = min(dimensions, n_features, max(1, n_samples - 2))
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=min(30, n_samples - 1),
            min_dist=0.0,
            metric="cosine",
            init="random",
            random_state=0,
        )

    console.print(
        f"[cyan]Reducing dimensions with {method.value}:[/cyan] "
        f"{n_features} dimensions to {n_components}."
    )
    with console.status(f"[cyan]Computing {method.value} embedding...[/cyan]"):
        reduced = np.asarray(reducer.fit_transform(vectors), dtype=float)
        imcluster_io.save_column(cache_column, reduced.tolist())
    console.print(
        f"[green]Cached {method.value} reduction:[/green] wrote "
        f"{len(reduced)} vectors to '{imcluster_io.output}'."
    )
    return reduced
