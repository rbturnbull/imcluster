"""Evaluation metrics for image clustering results."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .io import ImclusterIO

console = Console()


def clustering_accuracy(expected: ArrayLike, predicted: ArrayLike) -> float:
    """Return clustering accuracy under the optimal class-label assignment.

    Args:
        expected: Ground-truth class labels.
        predicted: Predicted cluster labels.

    Returns:
        The fraction of samples assigned correctly after optimally matching
        cluster IDs to expected classes.

    Raises:
        ValueError: If the label arrays are empty or have different lengths.
    """
    expected_labels = np.asarray(expected)
    predicted_labels = np.asarray(predicted)
    if not len(expected_labels) or len(expected_labels) != len(predicted_labels):
        raise ValueError("expected and predicted labels must have equal nonzero length")

    expected_values, expected_codes = np.unique(expected_labels, return_inverse=True)
    predicted_values, predicted_codes = np.unique(predicted_labels, return_inverse=True)
    contingency = np.zeros((len(expected_values), len(predicted_values)), dtype=int)
    np.add.at(contingency, (expected_codes, predicted_codes), 1)
    rows, columns = linear_sum_assignment(contingency, maximize=True)
    return float(contingency[rows, columns].sum() / len(expected_labels))


def evaluate_clustering(
    imcluster_io: ImclusterIO,
    expected_csv: str | Path,
    cluster_column: str,
) -> dict[str, float]:
    """Evaluate cached cluster assignments against filename-based classes.

    Args:
        imcluster_io: Image collection containing filenames and cluster labels.
        expected_csv: CSV with ``filename`` and ``class`` columns.
        cluster_column: DataFrame column containing predicted cluster labels.

    Returns:
        NMI, ARI, and optimally matched clustering accuracy.

    Raises:
        ValueError: If the CSV or clustering results cannot be aligned.
    """
    expected = pd.read_csv(expected_csv, dtype=str)
    required_columns = {"filename", "class"}
    missing_columns = required_columns.difference(expected.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Expected classes CSV is missing columns: {missing}")
    seen_filenames: set[str] = set()
    duplicates: list[str] = []
    for filename in expected["filename"].tolist():
        if filename in seen_filenames and filename not in duplicates:
            duplicates.append(filename)
        seen_filenames.add(filename)
    if duplicates:
        raise ValueError(
            "Expected classes CSV contains duplicate filenames: "
            + ", ".join(duplicates)
        )
    if expected["class"].isna().any() or (expected["class"].str.strip() == "").any():
        raise ValueError("Expected classes CSV contains an empty class")
    if cluster_column not in imcluster_io.df:
        raise ValueError(f"Missing clustering results column: {cluster_column}")

    classes_by_filename = expected.set_index("filename")["class"]
    filenames = imcluster_io.df["filenames"]
    missing_filenames = sorted(set(filenames).difference(classes_by_filename.index))
    if missing_filenames:
        raise ValueError(
            "Expected classes CSV has no class for: " + ", ".join(missing_filenames)
        )

    expected_labels = filenames.map(classes_by_filename).to_numpy()
    predicted_labels = imcluster_io.df[cluster_column].to_numpy()
    return {
        "NMI": float(normalized_mutual_info_score(expected_labels, predicted_labels)),
        "ARI": float(adjusted_rand_score(expected_labels, predicted_labels)),
        "ACC": clustering_accuracy(expected_labels, predicted_labels),
    }


def print_evaluation(metrics: dict[str, float]) -> None:
    """Print clustering metrics as a Rich table."""
    descriptions = {
        "NMI": "Normalized Mutual Information",
        "ARI": "Adjusted Rand Index",
        "ACC": "Clustering Accuracy",
    }
    table = Table(title="Clustering evaluation", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Description")
    table.add_column("Score", justify="right", style="green")
    for metric, score in metrics.items():
        table.add_row(metric, descriptions[metric], f"{score:.4f}")
    console.print(table)


def write_evaluation(metrics: dict[str, float], output_csv: str | Path) -> None:
    """Write evaluation metrics as a one-row CSV file.

    Args:
        metrics: Metric names mapped to numeric scores.
        output_csv: Destination CSV path.
    """
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output, index=False)
