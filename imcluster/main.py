from xml.sax.handler import feature_validation
import typer
from typing import List
from pathlib import Path
import numpy as np

import torch
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from .datasets import ImageDataset
from rich.console import Console
console = Console()

app = typer.Typer()

@app.command()
def cluster(
    inputs:List[Path],
    max_images:int = None,
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

    filenames = [image.name for image in images]

    dataset = ImageDataset(images)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = models.vgg19(pretrained=True)

    console.print("Generating feature vectors")
    with torch.no_grad():
        results = [model(batch) for batch in dataloader]
    feature_vectors = torch.cat(results, dim=0).cpu().detach().numpy()

    # PCA
    console.print("Fitting PCA")
    pca = PCA(n_components=2)
    feature_vectors_2D = pca.fit(feature_vectors).transform(feature_vectors)

    import plotly.express as px
    fig = px.scatter(x=feature_vectors_2D[:,0], y=feature_vectors_2D[:,1], hover_name=filenames)
    fig.show()
