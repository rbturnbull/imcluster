from io import StringIO

import pandas as pd
import pytest
from rich.console import Console

from imcluster.evaluate import (
    clustering_accuracy,
    evaluate_clustering,
    print_evaluation,
    write_evaluation,
)
from imcluster.io import ImclusterIO


def make_store(tmp_path, image_factory):
    """Return a two-class store with permuted predicted labels."""
    images = [image_factory(f"image-{index}.jpg") for index in range(4)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    store.df["spectral_cluster"] = [8, 8, 3, 3]
    return store


def test_evaluate_clustering_is_invariant_to_label_names(tmp_path, image_factory):
    store = make_store(tmp_path, image_factory)
    expected = tmp_path / "expected.csv"
    pd.DataFrame(
        {
            "filename": store.filenames,
            "class": ["airplane", "airplane", "forest", "forest"],
        }
    ).to_csv(expected, index=False)

    metrics = evaluate_clustering(store, expected, "spectral_cluster")

    assert metrics == {"NMI": 1.0, "ARI": 1.0, "ACC": 1.0}


def test_clustering_accuracy_uses_optimal_assignment():
    assert clustering_accuracy(["a", "a", "b", "b"], [0, 1, 1, 1]) == 0.75


@pytest.mark.parametrize(
    ("expected", "predicted"),
    [([], []), (["a"], []), (["a"], [0, 1])],
)
def test_clustering_accuracy_rejects_invalid_labels(expected, predicted):
    with pytest.raises(ValueError, match="equal nonzero length"):
        clustering_accuracy(expected, predicted)


def test_evaluate_clustering_requires_csv_columns(tmp_path, image_factory):
    store = make_store(tmp_path, image_factory)
    expected = tmp_path / "expected.csv"
    pd.DataFrame({"filename": store.filenames}).to_csv(expected, index=False)

    with pytest.raises(ValueError, match="missing columns: class"):
        evaluate_clustering(store, expected, "spectral_cluster")


def test_evaluate_clustering_rejects_duplicate_filenames(tmp_path, image_factory):
    store = make_store(tmp_path, image_factory)
    expected = tmp_path / "expected.csv"
    pd.DataFrame({"filename": [store.filenames[0]] * 2, "class": ["a", "a"]}).to_csv(
        expected, index=False
    )

    with pytest.raises(ValueError, match="duplicate filenames"):
        evaluate_clustering(store, expected, "spectral_cluster")


@pytest.mark.parametrize("class_value", [None, ""])
def test_evaluate_clustering_rejects_empty_classes(
    class_value, tmp_path, image_factory
):
    store = make_store(tmp_path, image_factory)
    expected = tmp_path / "expected.csv"
    pd.DataFrame(
        {"filename": store.filenames, "class": [class_value, "a", "b", "b"]}
    ).to_csv(expected, index=False)

    with pytest.raises(ValueError, match="empty class"):
        evaluate_clustering(store, expected, "spectral_cluster")


def test_evaluate_clustering_requires_cluster_results(tmp_path, image_factory):
    store = make_store(tmp_path, image_factory)
    expected = tmp_path / "expected.csv"
    pd.DataFrame({"filename": store.filenames, "class": ["a", "a", "b", "b"]}).to_csv(
        expected, index=False
    )

    with pytest.raises(ValueError, match="Missing clustering results column"):
        evaluate_clustering(store, expected, "dbscan_cluster")


def test_evaluate_clustering_requires_every_image_filename(tmp_path, image_factory):
    store = make_store(tmp_path, image_factory)
    expected = tmp_path / "expected.csv"
    pd.DataFrame({"filename": store.filenames[:3], "class": ["a", "a", "b"]}).to_csv(
        expected, index=False
    )

    with pytest.raises(ValueError, match=store.filenames[3]):
        evaluate_clustering(store, expected, "spectral_cluster")


def test_print_evaluation_outputs_rich_metrics(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(
        "imcluster.evaluate.console",
        Console(file=output, force_terminal=False, width=100),
    )

    print_evaluation({"NMI": 1.0, "ARI": 0.5, "ACC": 0.75})

    rendered = output.getvalue()
    assert "Clustering evaluation" in rendered
    assert "Normalized Mutual Information" in rendered
    assert "Adjusted Rand Index" in rendered
    assert "Clustering Accuracy" in rendered
    assert "0.7500" in rendered


def test_write_evaluation_creates_metrics_csv_and_parent_directories(tmp_path):
    output = tmp_path / "nested" / "reports" / "metrics.csv"

    write_evaluation({"NMI": 1.0, "ARI": 0.5, "ACC": 0.75}, output)

    assert pd.read_csv(output).to_dict(orient="records") == [
        {"NMI": 1.0, "ARI": 0.5, "ACC": 0.75}
    ]
