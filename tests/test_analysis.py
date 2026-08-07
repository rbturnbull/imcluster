import numpy as np
import pytest

from imcluster.cluster import cluster, console
from imcluster.io import ImclusterIO
from imcluster.pca import console as pca_console
from imcluster.pca import fit_pca


def test_pca_saves_two_components(tmp_path, image_factory):
    images = [image_factory(f"{index}.jpg") for index in range(3)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    fit_pca(store, features)

    assert store.has_column("pca0")
    assert store.has_column("pca1")
    assert len(store.get_column("pca0")) == 3


def test_pca_uses_precomputed_components(tmp_path, image_factory, monkeypatch):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )
    store.save_column("pca0", [1.0, 2.0], autosave=False)
    store.save_column("pca1", [3.0, 4.0])
    messages = []
    monkeypatch.setattr(pca_console, "print", messages.append)

    fit_pca(store, np.empty((2, 0)))

    assert store.get_column("pca0").tolist() == [1.0, 2.0]
    assert store.get_column("pca1").tolist() == [3.0, 4.0]
    assert messages == ["Using precomputed PCA"]


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


def test_unknown_clustering_algorithm_is_rejected(tmp_path, image_factory):
    store = ImclusterIO([image_factory("image.jpg")], tmp_path / "results.parquet")

    try:
        cluster(store, np.array([[1.0]]), algorithm="unknown")
    except Exception as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown clustering algorithm was accepted")


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
