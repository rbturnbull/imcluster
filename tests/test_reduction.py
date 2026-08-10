import numpy as np
import pytest

from imcluster.io import ImclusterIO
from imcluster.reduction import ReductionMethod, reduce_dimensions


def make_store(tmp_path, image_factory, count=4):
    """Return an image store and a small feature matrix."""
    images = [image_factory(f"{index}.jpg") for index in range(count)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    vectors = np.arange(count * 4, dtype=float).reshape(count, 4)
    return store, vectors


def test_no_reduction_returns_original_vectors(tmp_path, image_factory):
    store, vectors = make_store(tmp_path, image_factory)

    reduced = reduce_dimensions(store, vectors, ReductionMethod.NONE)

    np.testing.assert_array_equal(reduced, vectors)


def test_pca_reduction_is_cached(tmp_path, image_factory):
    store, vectors = make_store(tmp_path, image_factory)

    reduced = reduce_dimensions(store, vectors, ReductionMethod.PCA)

    assert reduced.shape == (4, 4)
    assert store.has_column("reduction_pca_50")


def test_pca_honours_requested_dimensions(tmp_path, image_factory):
    store, vectors = make_store(tmp_path, image_factory)

    reduced = reduce_dimensions(store, vectors, ReductionMethod.PCA, dimensions=2)

    assert reduced.shape == (4, 2)
    assert store.has_column("reduction_pca_2")


@pytest.mark.parametrize(
    ("method", "class_name", "expected_components"),
    [
        (ReductionMethod.TSNE, "TSNE", 3),
        (ReductionMethod.UMAP, "UMAP", 2),
    ],
)
def test_nonlinear_reduction_configuration(
    method, class_name, expected_components, tmp_path, image_factory, monkeypatch
):
    store, vectors = make_store(tmp_path, image_factory)
    observed = {}

    class FakeReducer:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def fit_transform(self, values):
            return np.asarray(values)[:, :expected_components]

    monkeypatch.setattr(f"imcluster.reduction.{class_name}", FakeReducer)

    reduced = reduce_dimensions(store, vectors, method)

    assert reduced.shape == (4, expected_components)
    assert observed["n_components"] == expected_components
    assert observed["metric"] == "cosine"
    assert observed["random_state"] == 0
    if method is ReductionMethod.TSNE:
        assert observed["perplexity"] == 3.0
    else:
        assert observed["n_neighbors"] == 3
        assert observed["min_dist"] == 0.0


def test_reduction_uses_cached_vectors(tmp_path, image_factory, monkeypatch):
    store, vectors = make_store(tmp_path, image_factory)
    cached = [[1.0, 2.0]] * len(vectors)
    store.save_column("reduction_umap_50", cached)
    monkeypatch.setattr(
        "imcluster.reduction.UMAP",
        lambda **kwargs: pytest.fail("cached UMAP was recomputed"),
    )

    reduced = reduce_dimensions(store, vectors, "umap")

    np.testing.assert_array_equal(reduced, cached)


@pytest.mark.parametrize("vectors", [[[1.0]], [[1.0], [2.0], [3.0]], [1.0, 2.0]])
def test_reduction_requires_one_vector_per_image(vectors, tmp_path, image_factory):
    store, _ = make_store(tmp_path, image_factory, count=2)

    with pytest.raises(ValueError, match="one row per image"):
        reduce_dimensions(store, vectors, "pca")


def test_reduction_rejects_unknown_method(tmp_path, image_factory):
    store, vectors = make_store(tmp_path, image_factory)

    with pytest.raises(ValueError, match="Unsupported reduction method: unknown"):
        reduce_dimensions(store, vectors, "unknown")


def test_reduction_requires_positive_dimensions(tmp_path, image_factory):
    store, vectors = make_store(tmp_path, image_factory)

    with pytest.raises(ValueError, match="dimensions must be at least 1"):
        reduce_dimensions(store, vectors, "pca", dimensions=0)


def test_umap_requires_three_images(tmp_path, image_factory):
    store, vectors = make_store(tmp_path, image_factory, count=2)

    with pytest.raises(ValueError, match="UMAP reduction requires at least three"):
        reduce_dimensions(store, vectors, ReductionMethod.UMAP)
