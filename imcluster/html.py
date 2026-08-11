"""HTML cluster-gallery generation."""

import base64
import json
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


def similar_indices(
    feature_vectors: ArrayLike,
    limit: int = 30,
) -> dict[int, list[int]]:
    """Return nearest image positions ranked by cosine similarity.

    Args:
        feature_vectors: Feature matrix containing one row per image.
        limit: Maximum number of neighbours returned for each image.

    Returns:
        Mapping from each image position to its most similar image positions.

    Raises:
        ValueError: If vectors are not a matrix or ``limit`` is negative.
    """
    vectors = np.asarray(feature_vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("feature_vectors must be a two-dimensional matrix")
    if limit < 0:
        raise ValueError("limit must not be negative")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    result: dict[int, list[int]] = {}
    for position, vector in enumerate(normalized):
        scores = normalized @ vector
        ranked = np.argsort(-scores, kind="stable")
        result[position] = [
            int(neighbour) for neighbour in ranked if neighbour != position
        ][:limit]
    return result


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
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

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

    cluster_name_column = f"{cluster_column}_name"
    cluster_titles: dict[object, str] = {}
    cluster_ids: dict[object, str] = {}
    for index, (cluster, items) in enumerate(data.items(), start=1):
        if cluster == -1:
            default_title = "Noise"
            cluster_ids[cluster] = "noise"
        elif isinstance(cluster, (int, np.integer)):
            default_title = f"Cluster {int(cluster) + 1}"
            cluster_ids[cluster] = str(int(cluster) + 1)
        else:
            default_title = str(cluster)
            cluster_ids[cluster] = str(index)
        cluster_title = default_title
        if imcluster_io.has_column(cluster_name_column):
            position = int(items[0]["position"])
            cached_title = imcluster_io.df.iloc[position][cluster_name_column]
            if isinstance(cached_title, str) and cached_title.strip():
                cluster_title = cached_title.strip()
        cluster_titles[cluster] = cluster_title

    similar_images: dict[int, list[int]]
    if feature_vectors is None:
        representatives = {key: items[0]["position"] for key, items in data.items()}
        similar_images = {position: [] for position in range(len(imcluster_io.df))}
    else:
        medoids = representative_indices(
            imcluster_io.df[cluster_column].to_numpy(), feature_vectors
        )
        representatives = medoids
        similar_images = similar_indices(feature_vectors)

    report_metadata = dict(metadata or {})
    report_metadata["Generated"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    header = base64.b64encode(
        files("imcluster").joinpath("assets/imcluster-header.png").read_bytes()
    ).decode("ascii")
    favicon = base64.b64encode(
        files("imcluster").joinpath("assets/imcluster-logo.png").read_bytes()
    ).decode("ascii")
    bootstrap_css = files("imcluster").joinpath("assets/bootstrap.min.css").read_text()
    bootstrap_js = (
        files("imcluster").joinpath("assets/bootstrap.bundle.min.js").read_text()
    )
    copy_icon = files("imcluster").joinpath("assets/copy.svg").read_text()
    search_icon = files("imcluster").joinpath("assets/search.svg").read_text()
    previous_icon = files("imcluster").joinpath("assets/chevron-left.svg").read_text()
    next_icon = files("imcluster").joinpath("assets/chevron-right.svg").read_text()
    result = template.render(
        data=data,
        metadata=report_metadata,
        header=header,
        favicon=favicon,
        representatives=representatives,
        cluster_titles=cluster_titles,
        cluster_ids=cluster_ids,
        bootstrap_css=bootstrap_css,
        bootstrap_js=bootstrap_js,
        copy_icon=copy_icon,
        search_icon=search_icon,
        previous_icon=previous_icon,
        next_icon=next_icon,
        similar_images_json=json.dumps(similar_images),
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(result)
