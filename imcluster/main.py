"""Command-line entry point for the imcluster pipeline."""

import tempfile
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .cluster import ClusteringAlgorithm, cluster
from .evaluate import evaluate_clustering, print_evaluation, write_evaluation
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
from .llm import DEFAULT_LLM, name_clusters
from .reduction import ReductionMethod, reduce_dimensions
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
    ] = ClusteringAlgorithm.KMEANS,
    reduce: Annotated[
        ReductionMethod,
        typer.Option(help="Dimensionality reduction applied before clustering."),
    ] = ReductionMethod.UMAP,
    reduction_dims: Annotated[
        int,
        typer.Option(min=1, help="Target number of dimensions after reduction."),
    ] = 50,
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
    name: Annotated[
        bool,
        typer.Option(help="Generate descriptive cluster names with a multimodal LLM."),
    ] = False,
    llm: Annotated[
        str,
        typer.Option(help="llmloader model identifier used to name clusters."),
    ] = DEFAULT_LLM,
    llm_temperature: Annotated[
        float,
        typer.Option(min=0.0, help="Sampling temperature used for cluster names."),
    ] = 0.2,
    llm_api_key: Annotated[
        str | None,
        typer.Option(help="Optional provider API key used by the naming LLM."),
    ] = None,
    in_group_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum in-cluster thumbnail examples sent to the naming LLM.",
        ),
    ] = 10,
    out_group_size: Annotated[
        int,
        typer.Option(
            min=0,
            help="Maximum contrasting outside examples sent to the naming LLM.",
        ),
    ] = 0,
    evaluate: Annotated[
        Path | None,
        typer.Option(
            "--evaluate",
            "--expected",
            exists=True,
            dir_okay=False,
            readable=True,
            help="CSV containing filename,class labels for clustering evaluation.",
        ),
    ] = None,
    metric: Annotated[
        Path | None,
        typer.Option(help="Write NMI, ARI, and ACC evaluation scores to this CSV."),
    ] = None,
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
    if metric is not None and evaluate is None:
        raise typer.BadParameter(
            "--metric requires --evaluate expected_classes.csv",
            param_hint="--metric",
        )

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
    cached_model_names = (
        {
            value
            for value in imcluster_io.df["model"].tolist()
            if isinstance(value, str) and value
        }
        if not inputs and model is None and imcluster_io.has_column("model")
        else set()
    )
    cached_model_name = next(iter(cached_model_names), None)
    if (
        len(cached_model_names) == 1
        and cached_model_name is not None
        and imcluster_io.has_column(cached_model_name)
        and not force_features
    ):
        model_name = cached_model_name
        console.print(
            f"[green]Using cached model:[/green] restored '{model_name}' from "
            f"'{imcluster_io.output}'."
        )
    else:
        try:
            model_name = resolve_model(model, dino_version, arch, size)
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--size") from error

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
    clustering_vectors = reduce_dimensions(
        imcluster_io,
        feature_vectors,
        method=reduce,
        dimensions=reduction_dims,
        force=force or force_features,
    )
    previous_reductions = (
        {
            value
            for value in imcluster_io.df["reduction"].tolist()
            if isinstance(value, str) and value
        }
        if imcluster_io.has_column("reduction")
        else {ReductionMethod.NONE.value}
    )
    reduction_changed = previous_reductions != {reduce.value}
    previous_reduction_dims = (
        set(imcluster_io.df["reduction_dims"].dropna().astype(int).tolist())
        if imcluster_io.has_column("reduction_dims")
        else set()
    )
    reduction_dims_changed = (
        reduce is not ReductionMethod.NONE
        and previous_reduction_dims != {reduction_dims}
    )
    cluster_column = f"{clustering.value}_cluster"
    cluster_force = (
        force
        or force_features
        or force_cluster
        or reduction_changed
        or reduction_dims_changed
    )
    cluster_was_recomputed = cluster_force or not imcluster_io.has_column(
        cluster_column
    )
    cluster(
        imcluster_io,
        clustering_vectors,
        algorithm=clustering,
        n_clusters=n_clusters,
        dbscan_eps=dbscan_eps,
        min_samples=min_samples,
        force=cluster_force,
    )
    cluster_name_column = f"{cluster_column}_name"
    if cluster_was_recomputed and imcluster_io.has_column(cluster_name_column):
        imcluster_io.df.drop(columns=[cluster_name_column], inplace=True)
    imcluster_io.df["model"] = model_name
    imcluster_io.df["algorithm"] = clustering.value
    imcluster_io.df["reduction"] = reduce.value
    imcluster_io.df["reduction_dims"] = reduction_dims
    imcluster_io.save()
    if evaluate is not None:
        try:
            metrics = evaluate_clustering(
                imcluster_io,
                evaluate,
                cluster_column=f"{clustering.value}_cluster",
            )
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--evaluate") from error
        print_evaluation(metrics)
        if metric is not None:
            write_evaluation(metrics, metric)
            console.print(
                f"[green]Wrote evaluation metrics:[/green] {metric.resolve()}"
            )
    generate_thumbnails(
        imcluster_io,
        thumbnail_height=thumbnail_height,
        thumbnail_width=thumbnail_width,
        force=force,
        force_thumbnails=force_thumbnails,
    )
    if name:
        try:
            name_clusters(
                imcluster_io,
                feature_vectors,
                cluster_column=cluster_column,
                llm=llm,
                temperature=llm_temperature,
                api_key=llm_api_key,
                in_group_size=in_group_size,
                out_group_size=out_group_size,
                force=force or cluster_was_recomputed,
            )
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--name") from error
    with console.status("[cyan]Rendering HTML gallery...[/cyan]"):
        write_html(
            imcluster_io,
            output_html=output_html,
            cluster_column=cluster_column,
            metadata={
                "Model": model_name,
                "Clustering": clustering.value,
                "Reduction": reduce.value,
                "Reduction dimensions": str(reduction_dims),
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
