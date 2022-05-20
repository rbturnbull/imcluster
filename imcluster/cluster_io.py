import base64
from io import BytesIO
from PIL import Image
from bokeh.palettes import Spectral6
from bokeh.plotting import figure, output_file, show

from .io import ImclusterIO
from pathlib import Path


def save_clusters(
    imcluster_io: ImclusterIO,
    output_dir=None,
    algorithm="DBSCAN",
    force: bool = False,
):
    """
    save clusters
    """
    all_clusters = imcluster_io.df["dbscan_cluster"].unique()
    for cluster in all_clusters:
        print(cluster)
        # Path(output_dir + "/" + str(cluster)).mkdir(parents=True, exist_ok=True)
        # cluster_df = imcluster_io.df[imcluster_io.df["dbscan_cluster"] == cluster]

        # print(Path(imcluster_io.images[i])) for i in cluster_df.index
