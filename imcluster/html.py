"""HTML cluster-gallery generation."""

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from .io import ImclusterIO


def write_html(
    imcluster_io: ImclusterIO,
    output_html: str | Path | None = None,
    cluster_column: str = "spectral_cluster",
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write an HTML gallery grouped by cached cluster labels.

    Args:
        imcluster_io: Image collection containing filenames and thumbnails.
        output_html: Destination path. Defaults to the Parquet path with an
            ``.html`` suffix.
        cluster_column: DataFrame column containing cluster labels.
        metadata: Optional report metadata displayed in the header.

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

    data: defaultdict[object, list[dict[str, str]]] = defaultdict(list)
    df = imcluster_io.df.sort_values(cluster_column)
    clusters = df[cluster_column]
    thumbnails = df["thumbnail"]
    filenames = df["filenames"]
    paths = df["path"]
    for filename, path, cluster, thumbnail in zip(
        filenames,
        paths,
        clusters,
        thumbnails,
        strict=True,
    ):
        data[cluster].append(
            {"filename": filename, "path": path, "thumbnail": thumbnail}
        )

    report_metadata = dict(metadata or {})
    report_metadata["Generated"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    result = template.render(data=data, metadata=report_metadata)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(result)
