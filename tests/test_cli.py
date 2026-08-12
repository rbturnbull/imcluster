import numpy as np
import pandas as pd
import pytest
from rich.text import Text
from typer.testing import CliRunner

from imcluster.io import ImclusterIO
from imcluster.main import app, open_gallery


def plain_output(result) -> str:
    """Return CLI output without terminal styling or layout wrapping."""
    text = Text.from_ansi(result.output).plain
    text = "".join(" " if "\u2500" <= char <= "\u257f" else char for char in text)
    return " ".join(text.split())


def invoke_cli(app, args, **kwargs):
    """Invoke the CLI with deterministic terminal rendering."""
    kwargs.setdefault("color", False)
    kwargs.setdefault("terminal_width", 240)
    return CliRunner().invoke(app, args, **kwargs)


@pytest.fixture(autouse=True)
def unavailable_dinov3(monkeypatch):
    """Avoid network access and exercise the default automatic fallback."""
    monkeypatch.setattr("imcluster.features.dinov3_available", lambda model: False)
    monkeypatch.setattr(
        "imcluster.main.reduce_dimensions",
        lambda store, vectors, **kwargs: vectors,
    )


def test_cli_help_is_available():
    result = invoke_cli(
        app,
        ["--help"],
        color=False,
        terminal_width=240,
    )
    help_text = plain_output(result)

    assert result.exit_code == 0
    assert "inputs" in help_text.lower()
    assert "--cache" in help_text
    assert "--gallery" in help_text
    assert "--no-open" in help_text
    assert "--arch" in help_text
    assert "--dino-version" in help_text
    assert "--size" in help_text
    assert "--model" in help_text
    assert "--reduction-dims" in help_text
    assert "--name" in help_text
    assert "--llm" in help_text
    assert "--llm-temperature" in help_text
    assert "--llm-api-key" in help_text
    assert "--in-group-size" in help_text
    assert "--out-group-size" in help_text
    assert "--reduction-dims" in help_text


def test_open_gallery_uses_file_uri(tmp_path, monkeypatch):
    gallery = tmp_path / "gallery.html"
    gallery.write_text("gallery")
    opened = []
    monkeypatch.setattr("imcluster.main.webbrowser.open", opened.append)

    open_gallery(gallery)

    assert opened == [gallery.resolve().as_uri()]


def test_cli_auto_falls_back_to_dinov2_base(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    observed = {}

    def fake_build_features(*args, **kwargs):
        observed["model_name"] = kwargs["model_name"]
        return [[1.0], [2.0]]

    def fake_reduce(store, vectors, **kwargs):
        observed["reduction"] = kwargs["method"].value
        observed["reduction_dims"] = kwargs["dimensions"]
        return vectors

    def fake_cluster(*args, **kwargs):
        observed["clustering"] = kwargs["algorithm"].value

    monkeypatch.setattr("imcluster.main.build_features", fake_build_features)
    monkeypatch.setattr("imcluster.main.reduce_dimensions", fake_reduce)
    monkeypatch.setattr("imcluster.main.cluster", fake_cluster)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)

    result = invoke_cli(
        app,
        [*(str(image) for image in images), "--no-open"],
    )

    assert result.exit_code == 0
    assert observed == {
        "model_name": "facebook/dinov2-base",
        "reduction": "umap",
        "reduction_dims": 50,
        "clustering": "kmeans",
    }


def test_cli_custom_model_overrides_arch_and_size(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    observed = {}

    def fake_build_features(*args, **kwargs):
        observed["model_name"] = kwargs["model_name"]
        return [[1.0], [2.0]]

    monkeypatch.setattr("imcluster.main.build_features", fake_build_features)
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            "--arch",
            "convnext",
            "--size",
            "max",
            "--model",
            "organization/custom-model",
            "--no-open",
        ],
    )

    assert result.exit_code == 0
    assert observed["model_name"] == "organization/custom-model"


def test_cli_rejects_unavailable_convnext_size(tmp_path, image_factory):
    image = image_factory("image.jpg")

    result = invoke_cli(
        app,
        [
            str(image),
            "--dino-version",
            "3",
            "--arch",
            "convnext",
            "--size",
            "huge",
        ],
    )

    output_text = plain_output(result)
    assert result.exit_code == 2
    assert "Size 'huge' is not available for DINOv3 architecture" in output_text
    assert "'convnext'" in output_text


def test_cli_rejects_cache_for_different_images(tmp_path, image_factory):
    cached_images = [image_factory("cached-one.jpg"), image_factory("cached-two.jpg")]
    requested_images = [
        image_factory("requested-one.jpg"),
        image_factory("requested-two.jpg"),
    ]
    cache = tmp_path / "results.parquet"
    ImclusterIO(cached_images, cache).save()

    result = invoke_cli(
        app,
        [
            *(str(image) for image in requested_images),
            "--cache",
            str(cache),
            "--no-open",
        ],
        color=False,
        terminal_width=240,
    )

    output_text = plain_output(result)
    assert result.exit_code == 2
    assert "Invalid value for --cache" in output_text
    assert "--force" in output_text


def test_cli_rejects_input_without_valid_images(tmp_path):
    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()

    result = invoke_cli(
        app,
        [str(empty_directory), "--no-open"],
    )

    output_text = plain_output(result)
    assert result.exit_code == 2
    assert "Invalid value for inputs" in output_text
    assert "No valid input images were found" in output_text


def test_cli_requires_at_least_two_images(tmp_path, image_factory):
    image = image_factory("only.jpg")

    result = invoke_cli(
        app,
        [str(image), "--no-open"],
    )

    output_text = plain_output(result)
    assert result.exit_code == 2
    assert "Invalid value for inputs" in output_text
    assert "At least two images are required" in output_text


def test_cli_wires_requested_output_and_algorithm(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    cache = tmp_path / "results.parquet"
    gallery = tmp_path / "report.html"
    observed = {}

    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )

    def fake_cluster(*args, **kwargs):
        observed["algorithm"] = kwargs["algorithm"]

    monkeypatch.setattr("imcluster.main.cluster", fake_cluster)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )

    def fake_write_html(*args, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr("imcluster.main.write_html", fake_write_html)

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            "--cache",
            str(cache),
            "--gallery",
            str(gallery),
            "--clustering",
            "dbscan",
            "--no-open",
        ],
    )

    assert result.exit_code == 0
    assert observed == {
        "algorithm": "dbscan",
        "output_html": gallery,
        "cluster_column": "dbscan_cluster",
        "metadata": {
            "Model": "facebook/dinov2-base",
            "Clustering": "dbscan",
            "Reduction": "umap",
            "Reduction dimensions": "50",
            "Images": "2",
        },
        "feature_vectors": [[1.0], [2.0]],
    }


def test_cli_reduces_features_and_invalidates_cached_clusters(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    observed = {}
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )

    def fake_reduce(*args, **kwargs):
        observed["dimensions"] = kwargs["dimensions"]
        return [[9.0], [8.0]]

    monkeypatch.setattr("imcluster.main.reduce_dimensions", fake_reduce)

    def fake_cluster(store, vectors, **kwargs):
        observed["vectors"] = vectors
        observed["force"] = kwargs["force"]
        store.df["spectral_cluster"] = [0, 1]

    monkeypatch.setattr("imcluster.main.cluster", fake_cluster)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            "--reduce",
            "pca",
            "--reduction-dims",
            "12",
            "--no-open",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert observed == {
        "dimensions": 12,
        "vectors": [[9.0], [8.0]],
        "force": True,
    }


def test_cli_names_clusters_with_configured_llm(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    features = [[1.0, 0.0], [0.0, 1.0]]
    observed = {}
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: features
    )

    def fake_cluster(store, *args, **kwargs):
        store.df["kmeans_cluster"] = [0, 1]
        store.df["kmeans_cluster_name"] = ["Stale", "Stale"]

    def fake_thumbnails(store, **kwargs):
        store.df["thumbnail"] = ["one-thumbnail", "two-thumbnail"]

    def fake_name_clusters(store, vectors, **kwargs):
        assert not store.has_column("kmeans_cluster_name")
        observed["vectors"] = vectors
        observed.update(kwargs)
        store.df["kmeans_cluster_name"] = ["Birds", "Trees"]

    monkeypatch.setattr("imcluster.main.cluster", fake_cluster)
    monkeypatch.setattr("imcluster.main.generate_thumbnails", fake_thumbnails)
    monkeypatch.setattr("imcluster.main.name_clusters", fake_name_clusters)
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            "--name",
            "--llm",
            "provider/model",
            "--llm-temperature",
            "0.4",
            "--llm-api-key",
            "secret",
            "--in-group-size",
            "7",
            "--out-group-size",
            "3",
            "--n-clusters",
            "2",
            "--no-open",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert observed == {
        "vectors": features,
        "cluster_column": "kmeans_cluster",
        "llm": "provider/model",
        "temperature": 0.4,
        "api_key": "secret",
        "in_group_size": 7,
        "out_group_size": 3,
        "force": True,
    }


def test_cli_reports_cluster_naming_errors(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "imcluster.main.name_clusters",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad LLM")),
    )

    result = invoke_cli(
        app,
        [*(str(image) for image in images), "--name", "--no-open"],
    )

    assert result.exit_code == 2
    assert "Invalid value for --name" in plain_output(result)
    assert "bad LLM" in plain_output(result)


def test_cli_runs_local_pipeline_and_writes_requested_files(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory(f"{index}.jpg") for index in range(4)]
    cache = tmp_path / "results.parquet"
    gallery = tmp_path / "report.html"
    features = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: features
    )

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            "--cache",
            str(cache),
            "--gallery",
            str(gallery),
            "--n-clusters",
            "2",
            "--no-open",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert cache.is_file()
    assert gallery.is_file()
    assert "Cluster" in gallery.read_text()
    output_text = plain_output(result)
    assert "Wrote processing cache:" in output_text
    assert "results.parquet" in output_text
    assert "Wrote HTML gallery:" in output_text
    assert "report.html" in output_text


def test_cli_opens_saved_gallery(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    gallery = tmp_path / "gallery.html"
    opened = []
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "imcluster.main.write_html",
        lambda *args, **kwargs: kwargs["output_html"].write_text("gallery"),
    )
    monkeypatch.setattr("imcluster.main.open_gallery", opened.append)

    result = invoke_cli(
        app,
        [*(str(image) for image in images), "--gallery", str(gallery)],
    )

    assert result.exit_code == 0, result.exception
    assert opened == [gallery]


def test_cli_uses_temporary_outputs_when_paths_are_omitted(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    temporary_directory = tmp_path / "temporary-output"
    temporary_directory.mkdir()
    written_gallery = []
    monkeypatch.setattr(
        "imcluster.main.tempfile.mkdtemp",
        lambda **kwargs: str(temporary_directory),
    )
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "imcluster.main.write_html",
        lambda *args, **kwargs: written_gallery.append(kwargs["output_html"]),
    )

    result = invoke_cli(
        app,
        [*(str(image) for image in images), "--no-open"],
    )

    assert result.exit_code == 0, result.exception
    assert (temporary_directory / "results.parquet").is_file()
    assert written_gallery == [temporary_directory / "gallery.html"]
    output_text = plain_output(result)
    assert "Cache is temporary" in output_text
    assert "--cache PATH" in output_text
    assert "Gallery is temporary" in output_text
    assert "--gallery PATH" in output_text


def test_cli_uses_existing_cache_without_image_inputs(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    cache = tmp_path / "results.parquet"
    gallery = tmp_path / "gallery.html"
    ImclusterIO(images, cache).save()
    observed = []
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda store, **kwargs: [[1.0], [2.0]]
    )
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "imcluster.main.write_html",
        lambda store, **kwargs: observed.extend(store.images),
    )

    result = invoke_cli(
        app,
        ["--cache", str(cache), "--gallery", str(gallery), "--no-open"],
    )

    assert result.exit_code == 0, result.exception
    assert observed == [image.resolve() for image in images]


def test_cache_only_run_restores_cached_model_without_source_images(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    cache = tmp_path / "results.parquet"
    expected = tmp_path / "expected.csv"
    expected.write_text("filename,class\none.jpg,a\ntwo.jpg,b\n")
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    store = ImclusterIO(images, cache)
    store.df[model_name] = [[1.0, 0.0], [0.0, 1.0]]
    store.df["spectral_cluster"] = [0, 1]
    store.df["thumbnail"] = ["one-thumbnail", "two-thumbnail"]
    store.df["model"] = model_name
    store.df["algorithm"] = "spectral"
    store.save()
    for image in images:
        image.unlink()
    observed = []
    monkeypatch.setattr(
        "imcluster.main.resolve_model",
        lambda *args, **kwargs: pytest.fail("cache-only run resolved a new model"),
    )
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.print_evaluation", lambda metrics: observed.append(metrics)
    )

    result = invoke_cli(
        app,
        [
            "--cache",
            str(cache),
            "--expected",
            str(expected),
            "--clustering",
            "spectral",
            "--reduce",
            "none",
            "--no-open",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert observed == [{"NMI": 1.0, "ARI": 1.0, "ACC": 1.0}]
    assert "Using cached model" in plain_output(result)


def test_cli_requires_inputs_when_cache_does_not_exist(tmp_path):
    result = invoke_cli(
        app,
        ["--cache", str(tmp_path / "missing.parquet"), "--no-open"],
    )

    assert result.exit_code == 2
    assert "Provide image inputs or an existing --cache file" in plain_output(result)


def test_cli_evaluates_expected_classes(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    expected = tmp_path / "expected.csv"
    expected.write_text("filename,class\none.jpg,a\ntwo.jpg,b\n")
    observed = []
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )

    def fake_cluster(store, *args, **kwargs):
        store.df["kmeans_cluster"] = [0, 1]

    monkeypatch.setattr("imcluster.main.cluster", fake_cluster)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.print_evaluation", lambda metrics: observed.append(metrics)
    )

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            "--evaluate",
            str(expected),
            "--no-open",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert observed == [{"NMI": 1.0, "ARI": 1.0, "ACC": 1.0}]


def test_cli_reports_invalid_expected_classes(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    expected = tmp_path / "expected.csv"
    expected.write_text("filename,class\none.jpg,a\ntwo.jpg,b\n")
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.evaluate_clustering",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad labels")),
    )

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            "--expected",
            str(expected),
            "--no-open",
        ],
    )

    assert result.exit_code == 2
    output_text = plain_output(result)
    assert "Invalid value for --evaluate" in output_text
    assert "bad labels" in output_text


def test_cli_writes_evaluation_metrics_csv(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    expected = tmp_path / "expected.csv"
    expected.write_text("filename,class\none.jpg,a\ntwo.jpg,b\n")
    metrics = tmp_path / "nested" / "metrics.csv"
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )

    def fake_cluster(store, *args, **kwargs):
        store.df["kmeans_cluster"] = [4, 9]

    monkeypatch.setattr("imcluster.main.cluster", fake_cluster)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            "--evaluate",
            str(expected),
            "--metric",
            str(metrics),
            "--no-open",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert pd.read_csv(metrics).to_dict(orient="records") == [
        {"NMI": 1.0, "ARI": 1.0, "ACC": 1.0}
    ]
    assert "Wrote evaluation metrics" in plain_output(result)


def test_cli_rejects_metric_without_evaluate(tmp_path):
    result = invoke_cli(
        app,
        ["--metric", str(tmp_path / "metrics.csv"), "--no-open"],
    )

    assert result.exit_code == 2
    output_text = plain_output(result)
    assert "Invalid value for --metric" in output_text
    assert "--metric requires --evaluate" in output_text
