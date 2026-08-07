import numpy as np
from rich.text import Text
from typer.testing import CliRunner

from imcluster.io import ImclusterIO
from imcluster.main import app


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
    assert "output_df" in help_text.lower()
    assert "--arch" in help_text
    assert "--size" in help_text
    assert "--model" in help_text


def test_cli_uses_default_vit_base_model(tmp_path, image_factory, monkeypatch):
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
        [*(str(image) for image in images), str(tmp_path / "results.parquet")],
    )

    assert result.exit_code == 0
    assert observed["model_name"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"


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
            str(tmp_path / "results.parquet"),
            "--arch",
            "convnext",
            "--size",
            "max",
            "--model",
            "organization/custom-model",
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
            str(tmp_path / "results.parquet"),
            "--arch",
            "convnext",
            "--size",
            "huge",
        ],
    )

    output_text = plain_output(result)
    assert result.exit_code == 2
    assert "Size 'huge' is not available for architecture" in output_text
    assert "'convnext'" in output_text


def test_cli_rejects_cache_for_different_images(tmp_path, image_factory):
    cached_images = [image_factory("cached-one.jpg"), image_factory("cached-two.jpg")]
    requested_images = [
        image_factory("requested-one.jpg"),
        image_factory("requested-two.jpg"),
    ]
    output = tmp_path / "results.parquet"
    ImclusterIO(cached_images, output).save()

    result = invoke_cli(
        app,
        [*(str(image) for image in requested_images), str(output)],
        color=False,
        terminal_width=240,
    )

    output_text = plain_output(result)
    assert result.exit_code == 2
    assert "Invalid value for output_df" in output_text
    assert "--force" in output_text


def test_cli_rejects_input_without_valid_images(tmp_path):
    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()

    result = invoke_cli(
        app,
        [str(empty_directory), str(tmp_path / "results.parquet")],
    )

    output_text = plain_output(result)
    assert result.exit_code == 2
    assert "Invalid value for inputs" in output_text
    assert "No valid input images were found" in output_text


def test_cli_requires_at_least_two_images(tmp_path, image_factory):
    image = image_factory("only.jpg")

    result = invoke_cli(
        app,
        [str(image), str(tmp_path / "results.parquet")],
    )

    output_text = plain_output(result)
    assert result.exit_code == 2
    assert "Invalid value for inputs" in output_text
    assert "At least two images are required" in output_text


def test_cli_wires_requested_output_and_algorithm(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    output_df = tmp_path / "results.parquet"
    output_html = tmp_path / "report.html"
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
            str(output_df),
            "--output-html",
            str(output_html),
            "--clustering",
            "dbscan",
        ],
    )

    assert result.exit_code == 0
    assert observed == {
        "algorithm": "dbscan",
        "output_html": output_html,
        "cluster_column": "dbscan_cluster",
        "metadata": {
            "Model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "Algorithm": "dbscan",
            "Images": "2",
        },
    }


def test_cli_runs_local_pipeline_and_writes_requested_files(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory(f"{index}.jpg") for index in range(4)]
    output_df = tmp_path / "results.parquet"
    output_html = tmp_path / "report.html"
    features = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: features
    )

    result = invoke_cli(
        app,
        [
            *(str(image) for image in images),
            str(output_df),
            "--output-html",
            str(output_html),
            "--n-clusters",
            "2",
        ],
    )

    assert result.exit_code == 0, result.exception
    assert output_df.is_file()
    assert output_html.is_file()
    assert "Cluster" in output_html.read_text()
