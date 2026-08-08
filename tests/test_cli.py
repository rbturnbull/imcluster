import numpy as np
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

    monkeypatch.setattr("imcluster.main.build_features", fake_build_features)
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "imcluster.main.generate_thumbnails", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)

    result = invoke_cli(
        app,
        [*(str(image) for image in images), "--no-open"],
    )

    assert result.exit_code == 0
    assert observed["model_name"] == "facebook/dinov2-base"


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
            "Images": "2",
        },
        "feature_vectors": [[1.0], [2.0]],
    }


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
