import pdb
import typer
from typing import List
from pathlib import Path

from .io import ImclusterIO
from .features import build_features
from .pca import fit_pca
from .cluster import cluster
from .plotting import plot

from rich.console import Console
console = Console()

app = typer.Typer()

@app.command()
def main(
    inputs:List[Path],
    output_df:Path,
    output_html:Path = None,
    model:str = "vgg19",
    max_images:int = None,
    algorithm:str = "SPECTRAL",
    force:bool = False,
    force_features:bool = False,
    force_pca:bool = False,
    force_cluster:bool = False,
): 
    # find images
    images = []
    for i in inputs:
        if i.is_dir():
            # TODO search directory
            continue
        else:
            # If it is a single file then just add to the list
            images.append(i)

    # truncate list of images if the user sets the maximum allowed
    if max_images and len(images) > max_images:
        images = images[:max_images]

    imcluster_io = ImclusterIO(images, output_df)

    feature_vectors = build_features(imcluster_io, model_name=model, force=force or force_features)
    fit_pca(imcluster_io, feature_vectors, force=force or force_features or force_pca)
    cluster(imcluster_io, feature_vectors, algorithm=algorithm, force=force or force_features or force_cluster)
    plot(imcluster_io, output_html)



    

