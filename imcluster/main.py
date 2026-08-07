import typer
from typing import List, Optional
from pathlib import Path

from .io import ImclusterIO
from .features import DEFAULT_MODEL, build_features
from .pca import fit_pca
from .cluster import cluster
from .plotting import plot
from .html import write_html

from rich.console import Console

console = Console()

app = typer.Typer()


@app.command()
def main(
    inputs:List[Path],
    output_df:Path,
    output_html:Optional[Path] = None,
    model:str = typer.Option(
        DEFAULT_MODEL,
        help="Hugging Face image-feature-extraction model name.",
    ),
    max_images:int = None,
    algorithm:str = "SPECTRAL",
    n_clusters:int = 20,
    thumbnail_width:int = 256,
    thumbnail_height:int = 256,
    force:bool = False,
    force_features:bool = False,
    force_pca:bool = False,
    force_cluster:bool = False,
    force_thumbnails:bool = False,
): 
    imcluster_io = ImclusterIO(inputs, output_df, max_images=max_images)
    feature_vectors = build_features(
        imcluster_io, model_name=model, force=force or force_features
    )
    fit_pca(imcluster_io, feature_vectors, force=force or force_features or force_pca)

    cluster(
        imcluster_io,
        feature_vectors,
        algorithm=algorithm,
        n_clusters=n_clusters,
        force=force or force_features or force_cluster,
    )
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
        cluster_column=f"{algorithm.lower()}_cluster",
    )
