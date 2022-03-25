import typer
from typing import List
from pathlib import Path
import numpy as np

import torch
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, SpectralClustering

import pandas as pd

from bokeh.plotting import figure, output_file, show

from .datasets import ImageDataset
from rich.console import Console
console = Console()

app = typer.Typer()

@app.command()
def cluster(
    inputs:List[Path],
    output:Path,
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

    if output.exists():
        df = pd.read_parquet(output, engine="pyarrow")
    else:
        df = pd.Series(filenames, name="filenames").to_frame()


    # Build features if not in dataframe
    if 'features' not in df:
        console.print("Generating feature vectors")
        dataset = ImageDataset(images)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        model = models.vgg19(pretrained=True)

        with torch.no_grad():
            results = [model(batch) for batch in dataloader]
        feature_vectors = torch.cat(results, dim=0).cpu().detach().numpy()

        df['features'] = [feature_vectors[x] for x in range(feature_vectors.shape[0])]
        df.to_parquet(output, engine="pyarrow")
    else:
        feature_vectors = np.array(df['features'].to_list())
        print(feature_vectors.shape)

    # PCA
    if 'pca0' not in df:
        console.print("Fitting PCA")
        pca = PCA(n_components=2)
        feature_vectors_2D = pca.fit(feature_vectors).transform(feature_vectors)
    
        df['pca0'] = feature_vectors_2D[:,0]
        df['pca1'] = feature_vectors_2D[:,1]

    if 'cluster' not in df:
        console.print("Clustering")
        # clustering = DBSCAN(eps=30.0, min_samples=4)
        clustering = SpectralClustering(n_clusters=2)
        clustering.fit(feature_vectors)
        df['cluster'] = clustering.labels_
        df.to_parquet(output, engine="pyarrow")

        print(df['cluster'])

    # output to static HTML file
    output_file("line.html")

    p = figure(width=1200, height=700)

    # add a circle renderer with a size, color, and alpha
    p.circle(df["pca0"], df["pca1"], size=5, color="navy", alpha=0.5)

    # show the results
    show(p)

    # import plotly.express as px
    # fig = px.scatter(df, x='pca0', y='pca1', color='cluster', hover_name='filenames')
    # fig.update_traces(
    #     hovertemplate="<br>".join([
    #         "ColX: %{x}",
    #         "ColY: %{y}",
    #         '<img src="data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==" alt="Red dot" />',
    #     ])
    # )

    # fig.show()


    

