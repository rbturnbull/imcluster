"""HTML cluster-gallery generation."""

import base64
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape
from numpy.typing import ArrayLike

from .io import ImclusterIO


def representative_indices(
    cluster_labels: ArrayLike,
    feature_vectors: ArrayLike,
) -> dict[object, int]:
    """Return the cosine medoid position for each cluster."""
    labels = np.asarray(cluster_labels)
    vectors = np.asarray(feature_vectors, dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(labels):
        raise ValueError("feature_vectors must contain one row per cluster label")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    representatives: dict[object, int] = {}
    for label in dict.fromkeys(labels.tolist()):
        positions = np.flatnonzero(labels == label)
        cluster_vectors = normalized[positions]
        similarity_totals = cluster_vectors @ cluster_vectors.sum(axis=0)
        representatives[label] = int(positions[np.argmax(similarity_totals)])
    return representatives


def write_html(
    imcluster_io: ImclusterIO,
    output_html: str | Path | None = None,
    cluster_column: str = "spectral_cluster",
    metadata: Mapping[str, str] | None = None,
    feature_vectors: ArrayLike | None = None,
) -> None:
    """Write an HTML gallery grouped by cached cluster labels.

    Args:
        imcluster_io: Image collection containing filenames and thumbnails.
        output_html: Destination path. Defaults to the Parquet path with an
            ``.html`` suffix.
        cluster_column: DataFrame column containing cluster labels.
        metadata: Optional report metadata displayed in the header.
        feature_vectors: Embeddings used to select a representative image for
            each cluster. Defaults to the first image when omitted.

    Raises:
        ValueError: If ``cluster_column`` is absent from the result table.
    """

    env = Environment(loader=PackageLoader(__package__), autoescape=select_autoescape())

    template = env.get_template("clusters.html")
    # template = env.get_template("vtab.html")

    if not output_html:
        output_html = imcluster_io.output.with_suffix(".html")

    if cluster_column not in imcluster_io.df:
        raise ValueError(f"Missing clustering results column: {cluster_column}")

    data: defaultdict[object, list[dict[str, Any]]] = defaultdict(list)
    df = imcluster_io.df.assign(_position=range(len(imcluster_io.df))).sort_values(
        cluster_column
    )
    clusters = df[cluster_column]
    thumbnails = df["thumbnail"]
    filenames = df["filenames"]
    paths = df["path"]
    positions = df["_position"]
    for filename, path, cluster, thumbnail, position in zip(
        filenames,
        paths,
        clusters,
        thumbnails,
        positions,
        strict=True,
    ):
        data[cluster].append(
            {
                "filename": filename,
                "path": path,
                "file_uri": Path(path).resolve().as_uri(),
                "thumbnail": thumbnail,
                "position": position,
            }
        )

    if feature_vectors is None:
        representatives = {key: items[0]["thumbnail"] for key, items in data.items()}
    else:
        medoids = representative_indices(
            imcluster_io.df[cluster_column].to_numpy(), feature_vectors
        )
        representatives = {
            key: imcluster_io.df.iloc[position]["thumbnail"]
            for key, position in medoids.items()
        }

    report_metadata = dict(metadata or {})
    report_metadata["Generated"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    header = base64.b64encode(
        files("imcluster").joinpath("assets/imcluster-header.png").read_bytes()
    ).decode("ascii")
    result = template.render(
        data=data,
        metadata=report_metadata,
        header=header,
        representatives=representatives,
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(result)
