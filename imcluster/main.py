"""Command-line entry point for the imcluster pipeline."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .cluster import ClusteringAlgorithm, cluster
from .features import (
    Device,
    ModelArchitecture,
    ModelSize,
    build_features,
    resolve_model,
)
from .html import write_html
from .io import ImclusterIO
from .pca import fit_pca
from .plotting import plot

console = Console()

app = typer.Typer()


@app.command()
def main(
    inputs: Annotated[
        list[Path],
        typer.Argument(help="Image files, directories, or text manifests to process."),
    ],
    output_df: Annotated[
        Path,
        typer.Argument(help="Destination Parquet file for cached results."),
    ],
    output_html: Annotated[
        Path | None,
        typer.Option(help="Destination path for the HTML cluster gallery."),
    ] = None,
    arch: Annotated[
        ModelArchitecture,
        typer.Option(
            help="DINOv3 architecture family used when --model is not supplied."
        ),
    ] = ModelArchitecture.VIT,
    size: Annotated[
        ModelSize,
        typer.Option(help="DINOv3 model size used when --model is not supplied."),
    ] = ModelSize.BASE,
    model: Annotated[
        str | None,
        typer.Option(help="Hugging Face model ID; overrides --arch and --size."),
    ] = None,
    device: Annotated[
        Device,
        typer.Option(help="Device for model inference."),
    ] = Device.AUTO,
    batch_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of images processed per inference batch.",
        ),
    ] = 8,
    max_images: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum number of input images to process."),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(help="Search directory inputs recursively."),
    ] = False,
    algorithm: Annotated[
        ClusteringAlgorithm,
        typer.Option(help="Clustering algorithm."),
    ] = ClusteringAlgorithm.SPECTRAL,
    n_clusters: Annotated[
        int,
        typer.Option(min=2, help="Number of clusters for spectral clustering."),
    ] = 20,
    dbscan_eps: Annotated[
        float,
        typer.Option(help="Maximum cosine distance between DBSCAN neighbours."),
    ] = 0.5,
    dbscan_min_samples: Annotated[
        int,
        typer.Option(min=1, help="Minimum DBSCAN neighbourhood size."),
    ] = 2,
    thumbnail_width: Annotated[
        int,
        typer.Option(min=1, help="Maximum thumbnail width in pixels."),
    ] = 256,
    thumbnail_height: Annotated[
        int,
        typer.Option(min=1, help="Maximum thumbnail height in pixels."),
    ] = 256,
    force: Annotated[
        bool,
        typer.Option(help="Recompute all cached processing stages."),
    ] = False,
    force_features: Annotated[
        bool,
        typer.Option(help="Recompute feature vectors and downstream stages."),
    ] = False,
    force_pca: Annotated[
        bool,
        typer.Option(help="Recompute PCA coordinates."),
    ] = False,
    force_cluster: Annotated[
        bool,
        typer.Option(help="Recompute cluster labels."),
    ] = False,
    force_thumbnails: Annotated[
        bool,
        typer.Option(help="Regenerate cached thumbnails."),
    ] = False,
) -> None:
    """Cluster images and write cached Parquet data and an HTML gallery."""
    try:
        model_name = resolve_model(model, arch, size)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--size") from error

    try:
        imcluster_io = ImclusterIO(
            inputs,
            output_df,
            max_images=max_images,
            recursive=recursive,
            reset_cache=force,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="output_df") from error
    if not imcluster_io.images:
        raise typer.BadParameter(
            "No valid input images were found", param_hint="inputs"
        )
    if len(imcluster_io.images) < 2:
        raise typer.BadParameter(
            "At least two images are required", param_hint="inputs"
        )

    feature_vectors = build_features(
        imcluster_io,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        force=force or force_features,
    )
    fit_pca(imcluster_io, feature_vectors, force=force or force_features or force_pca)

    cluster(
        imcluster_io,
        feature_vectors,
        algorithm=algorithm,
        n_clusters=n_clusters,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        force=force or force_features or force_cluster,
    )
    imcluster_io.df["model"] = model_name
    imcluster_io.df["algorithm"] = algorithm.value
    imcluster_io.save()
    plot(
        imcluster_io,
        thumbnail_height=thumbnail_height,
        thumbnail_width=thumbnail_width,
        force=force,
        force_thumbnails=force_thumbnails,
    )
    write_html(
        imcluster_io,
        output_html=output_html,
        cluster_column=f"{algorithm.value}_cluster",
        metadata={
            "Model": model_name,
            "Algorithm": algorithm.value,
            "Images": str(len(imcluster_io.images)),
        },
    )
