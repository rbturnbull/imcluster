from typer.testing import CliRunner

from imcluster.main import app


def test_cli_help_is_available():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "inputs" in result.stdout.lower()
    assert "output_df" in result.stdout.lower()
