import numpy as np
from click import unstyle
from typer.testing import CliRunner

from imcluster.main import app


def test_cli_help_is_available():
    result = CliRunner().invoke(app, ["--help"], color=False)
    help_text = unstyle(result.stdout)

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
    monkeypatch.setattr("imcluster.main.fit_pca", lambda *args, **kwargs: None)
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr("imcluster.main.plot", lambda *args, **kwargs: None)
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
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
    monkeypatch.setattr("imcluster.main.fit_pca", lambda *args, **kwargs: None)
    monkeypatch.setattr("imcluster.main.cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr("imcluster.main.plot", lambda *args, **kwargs: None)
    monkeypatch.setattr("imcluster.main.write_html", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
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

    result = CliRunner().invoke(
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

    assert result.exit_code == 2
    assert "Size 'huge' is not available for architecture" in result.output
    assert "'convnext'" in result.output


def test_cli_wires_requested_output_and_algorithm(tmp_path, image_factory, monkeypatch):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    output_df = tmp_path / "results.parquet"
    output_html = tmp_path / "report.html"
    observed = {}

    monkeypatch.setattr(
        "imcluster.main.build_features", lambda *args, **kwargs: [[1.0], [2.0]]
    )
    monkeypatch.setattr("imcluster.main.fit_pca", lambda *args, **kwargs: None)

    def fake_cluster(*args, **kwargs):
        observed["algorithm"] = kwargs["algorithm"]

    monkeypatch.setattr("imcluster.main.cluster", fake_cluster)
    monkeypatch.setattr("imcluster.main.plot", lambda *args, **kwargs: None)

    def fake_write_html(*args, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr("imcluster.main.write_html", fake_write_html)

    result = CliRunner().invoke(
        app,
        [
            *(str(image) for image in images),
            str(output_df),
            "--output-html",
            str(output_html),
            "--algorithm",
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

    result = CliRunner().invoke(
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
