"""Command-line entry point for the imcluster pipeline."""

import tempfile
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .cluster import ClusteringAlgorithm, cluster
from .features import (
    Device,
    DinoVersion,
    ModelArchitecture,
    ModelSize,
    build_features,
    resolve_model,
)
from .html import write_html
from .io import ImclusterIO
from .thumbnails import generate_thumbnails

console = Console()

app = typer.Typer()


def open_gallery(path: Path) -> None:
    """Open a generated HTML gallery in the default web browser."""
    webbrowser.open(path.resolve().as_uri())


@app.command()
def main(
    inputs: Annotated[
        list[Path] | None,
        typer.Argument(
            help=(
                "Image files, directories, or text manifests. May be omitted when "
                "loading an existing --cache file."
            )
        ),
    ] = None,
    cache: Annotated[
        Path | None,
        typer.Option(help="Preserve processing results in this Parquet file."),
    ] = None,
    gallery: Annotated[
        Path | None,
        typer.Option(help="Preserve the HTML gallery at this path."),
    ] = None,
    dino_version: Annotated[
        DinoVersion,
        typer.Option(help="DINO model generation used for preset selection."),
    ] = DinoVersion.AUTO,
    arch: Annotated[
        ModelArchitecture,
        typer.Option(help="DINOv3 architecture family; ignored for DINOv2."),
    ] = ModelArchitecture.VIT,
    size: Annotated[
        ModelSize,
        typer.Option(help="DINO model size used when --model is not supplied."),
    ] = ModelSize.BASE,
    model: Annotated[
        str | None,
        typer.Option(
            help="Hugging Face model ID; overrides --dino-version, --arch, and --size."
        ),
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
    clustering: Annotated[
        ClusteringAlgorithm,
        typer.Option(help="Clustering algorithm to use."),
    ] = ClusteringAlgorithm.SPECTRAL,
    n_clusters: Annotated[
        int,
        typer.Option(min=2, help="Number of clusters for fixed-count methods."),
    ] = 20,
    dbscan_eps: Annotated[
        float,
        typer.Option(help="Maximum cosine distance between DBSCAN neighbours."),
    ] = 0.5,
    min_samples: Annotated[
        int,
        typer.Option(min=1, help="Minimum sample count for density clustering."),
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
    force_cluster: Annotated[
        bool,
        typer.Option(help="Recompute cluster labels."),
    ] = False,
    force_thumbnails: Annotated[
        bool,
        typer.Option(help="Regenerate cached thumbnails."),
    ] = False,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Do not open the generated gallery."),
    ] = False,
) -> None:
    """Cluster images and open an HTML gallery."""
    try:
        model_name = resolve_model(model, dino_version, arch, size)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--size") from error

    temporary_directory: Path | None = None
    if cache is None:
        temporary_directory = Path(tempfile.mkdtemp(prefix="imcluster-"))
        output_df = temporary_directory / "results.parquet"
    else:
        output_df = cache
    if gallery is None:
        if temporary_directory is None:
            temporary_directory = Path(tempfile.mkdtemp(prefix="imcluster-"))
        output_html = temporary_directory / "gallery.html"
    else:
        output_html = gallery

    try:
        imcluster_io = ImclusterIO(
            inputs or [],
            output_df,
            max_images=max_images,
            recursive=recursive,
            reset_cache=force,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--cache") from error
    if not imcluster_io.images:
        raise typer.BadParameter(
            "No valid input images were found. Provide image inputs or an existing "
            "--cache file.",
            param_hint="inputs",
        )
    if len(imcluster_io.images) < 2:
        raise typer.BadParameter(
            "At least two images are required", param_hint="inputs"
        )
    console.print(
        f"[bold]Processing {len(imcluster_io.images)} images[/bold] with model "
        f"'{model_name}' and {clustering.value} clustering."
    )

    feature_vectors = build_features(
        imcluster_io,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        force=force or force_features,
    )
    cluster(
        imcluster_io,
        feature_vectors,
        algorithm=clustering,
        n_clusters=n_clusters,
        dbscan_eps=dbscan_eps,
        min_samples=min_samples,
        force=force or force_features or force_cluster,
    )
    imcluster_io.df["model"] = model_name
    imcluster_io.df["algorithm"] = clustering.value
    imcluster_io.save()
    generate_thumbnails(
        imcluster_io,
        thumbnail_height=thumbnail_height,
        thumbnail_width=thumbnail_width,
        force=force,
        force_thumbnails=force_thumbnails,
    )
    with console.status("[cyan]Rendering HTML gallery...[/cyan]"):
        write_html(
            imcluster_io,
            output_html=output_html,
            cluster_column=f"{clustering.value}_cluster",
            metadata={
                "Model": model_name,
                "Clustering": clustering.value,
                "Images": str(len(imcluster_io.images)),
            },
            feature_vectors=feature_vectors,
        )
    console.print(f"[green]Wrote processing cache:[/green] {output_df.resolve()}")
    console.print(f"[green]Wrote HTML gallery:[/green] {output_html.resolve()}")
    if cache is None:
        console.print(
            "[yellow]Cache is temporary:[/yellow] no persistent cache file was "
            "requested. Use [bold]--cache PATH[/bold] to preserve it."
        )
    if gallery is None:
        console.print(
            "[yellow]Gallery is temporary:[/yellow] no persistent gallery file "
            "was requested. Use [bold]--gallery PATH[/bold] to preserve it."
        )
    if not no_open:
        console.print(f"[cyan]Opening gallery:[/cyan] {output_html.resolve()}")
        open_gallery(output_html)
    else:
        console.print(
            "[dim]Gallery was not opened because --no-open was specified.[/dim]"
        )
