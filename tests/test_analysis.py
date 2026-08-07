import numpy as np
import pytest

from imcluster.cluster import cluster, console
from imcluster.io import ImclusterIO


def test_spectral_clustering_saves_labels(tmp_path, image_factory):
    images = [image_factory(f"{index}.jpg") for index in range(4)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    features = np.array([[0.0], [0.1], [9.9], [10.0]])

    cluster(store, features, algorithm="spectral", n_clusters=2)

    assert store.has_column("spectral_cluster")
    assert set(store.get_column("spectral_cluster")) == {0, 1}


def test_dbscan_clustering_saves_labels(tmp_path, image_factory):
    images = [image_factory(f"{index}.jpg") for index in range(4)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    features = np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]])

    cluster(store, features, algorithm="dbscan")

    assert store.has_column("dbscan_cluster")
    assert len(store.get_column("dbscan_cluster")) == 4


@pytest.mark.parametrize("algorithm", ["kmeans", "agglomerative", "hierarchical"])
def test_fixed_count_clustering_methods_save_labels(
    algorithm, tmp_path, image_factory
):
    images = [image_factory(f"{index}.jpg") for index in range(4)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    features = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    )

    cluster(store, features, algorithm=algorithm, n_clusters=2)

    column = f"{algorithm}_cluster"
    assert store.has_column(column)
    assert set(store.get_column(column)) == {0, 1}


def test_hdbscan_clustering_saves_labels(tmp_path, image_factory):
    images = [image_factory(f"{index}.jpg") for index in range(6)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    features = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.01, 0.99],
            [0.02, 0.98],
        ]
    )

    cluster(store, features, algorithm="hdbscan", min_samples=2)

    assert store.has_column("hdbscan_cluster")
    assert len(store.get_column("hdbscan_cluster")) == 6


def test_spectral_clustering_uses_precomputed_labels(
    tmp_path, image_factory, monkeypatch
):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )
    store.save_column("spectral_cluster", [4, 5])
    messages = []
    monkeypatch.setattr(console, "print", messages.append)

    cluster(store, np.empty((2, 0)), algorithm="spectral")

    assert store.get_column("spectral_cluster").tolist() == [4, 5]
    assert messages == ["Using precomputed spectral clusters"]


def test_dbscan_clustering_uses_precomputed_labels(
    tmp_path, image_factory, monkeypatch
):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )
    store.save_column("dbscan_cluster", [-1, 2])
    messages = []
    monkeypatch.setattr(console, "print", messages.append)

    cluster(store, np.empty((2, 0)), algorithm="dbscan")

    assert store.get_column("dbscan_cluster").tolist() == [-1, 2]
    assert messages == ["Using precomputed dbscan clusters"]


@pytest.mark.parametrize(
    "algorithm", ["hdbscan", "kmeans", "agglomerative", "hierarchical"]
)
def test_additional_clustering_methods_use_precomputed_labels(
    algorithm, tmp_path, image_factory, monkeypatch
):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )
    column = f"{algorithm}_cluster"
    store.save_column(column, [0, 1])
    messages = []
    monkeypatch.setattr(console, "print", messages.append)

    cluster(store, np.empty((2, 0)), algorithm=algorithm)

    assert store.get_column(column).tolist() == [0, 1]
    assert messages == [f"Using precomputed {algorithm} clusters"]


def test_unknown_clustering_algorithm_is_rejected(tmp_path, image_factory):
    store = ImclusterIO([image_factory("image.jpg")], tmp_path / "results.parquet")

    with pytest.raises(ValueError, match="Unsupported clustering algorithm: unknown"):
        cluster(store, np.array([[1.0]]), algorithm="unknown")


@pytest.mark.parametrize(
    "features",
    [np.array([1.0, 2.0]), np.array([[1.0]])],
)
def test_feature_vectors_must_have_one_row_per_image(
    features, tmp_path, image_factory
):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )

    with pytest.raises(ValueError, match="one row per image"):
        cluster(store, features)


def test_min_samples_must_be_positive(tmp_path, image_factory):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )

    with pytest.raises(ValueError, match="min_samples must be at least 1"):
        cluster(store, np.array([[1.0], [2.0]]), algorithm="dbscan", min_samples=0)


def test_fixed_cluster_count_must_be_at_least_two(tmp_path, image_factory):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )

    with pytest.raises(ValueError, match="n_clusters must be at least 2"):
        cluster(store, np.array([[1.0], [2.0]]), n_clusters=1)


def test_hdbscan_requires_two_minimum_samples(tmp_path, image_factory):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )

    with pytest.raises(ValueError, match="at least 2 for HDBSCAN"):
        cluster(
            store,
            np.array([[1.0], [2.0]]),
            algorithm="hdbscan",
            min_samples=1,
        )


def test_spectral_cluster_count_cannot_exceed_images(tmp_path, image_factory):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        cluster(store, np.array([[1.0], [2.0]]), n_clusters=3)


def test_dbscan_parameters_are_validated(tmp_path, image_factory):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )

    with pytest.raises(ValueError, match="dbscan_eps"):
        cluster(store, np.array([[1.0], [2.0]]), algorithm="dbscan", dbscan_eps=0)
