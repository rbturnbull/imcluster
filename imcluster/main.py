import imp
import pdb
import typer
from typing import List
from pathlib import Path

from .io import ImclusterIO
from .features import build_features, TorchvisionModelName
from .pca import fit_pca
from .cluster import cluster
from .plotting import plot
from .html import write_html
from .cluster_io import save_clusters

from rich.console import Console

console = Console()

app = typer.Typer()


@app.command()
def main(
    inputs:List[Path],
    output_df:Path,
    output_html:Path = None,
    model:TorchvisionModelName = typer.Option("vgg19", help="The name of the torchvision model to use (see https://pytorch.org/vision/stable/models.html#)."),
    max_images:int = None,
    algorithm:str = "SPECTRAL",
    n_clusters:int = 20,
    batch_size:int = 1,
    force:bool = False,
    force_features:bool = False,
    force_pca:bool = False,
    force_cluster:bool = False,
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
    # save_clusters(
    # imcluster_io=imcluster_io, output_dir=output_path, algorithm=algorithm
    # )
    plot(imcluster_io, output_html)
    write_html(imcluster_io)
