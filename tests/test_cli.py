from typer.testing import CliRunner
import numpy as np

from imcluster.main import app


def test_cli_help_is_available():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "inputs" in result.stdout.lower()
    assert "output_df" in result.stdout.lower()
    assert "facebook/dinov3" in result.stdout


def test_cli_wires_requested_output_and_algorithm(tmp_path, image_factory, monkeypatch):
    image = image_factory("image.jpg")
    output_df = tmp_path / "results.parquet"
    output_html = tmp_path / "report.html"
    observed = {}

    monkeypatch.setattr("imcluster.main.build_features", lambda *args, **kwargs: [[1.0]])
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
            str(image),
            str(output_df),
            "--output-html",
            str(output_html),
            "--algorithm",
            "DBSCAN",
        ],
    )

    assert result.exit_code == 0
    assert observed == {
        "algorithm": "DBSCAN",
        "output_html": output_html,
        "cluster_column": "dbscan_cluster",
    }


def test_cli_runs_local_pipeline_and_writes_requested_files(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory(f"{index}.jpg") for index in range(4)]
    output_df = tmp_path / "results.parquet"
    output_html = tmp_path / "report.html"
    features = np.array(
        [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]]
    )
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
