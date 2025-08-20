import base64
from io import BytesIO
from PIL import Image
from bokeh.palettes import Spectral6
from bokeh.plotting import figure, output_file, show
from rich.console import Console

console = Console()

from .io import ImclusterIO


def generate_thumbnail(path, width, height):
    im = Image.open(path)
    size = width, height
    im.thumbnail(size, Image.Resampling.LANCZOS)
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def plot(
    imcluster_io: ImclusterIO,
    output_html=None,
    width=1200,
    height=700,
    size=12,
    alpha=0.5,
    thumbnail_width:int=256,
    thumbnail_height:int=256,
    force: bool = False,
    force_thumbnails:bool = False,
):
    """
    Plot the principle components with tooltips showing the images.
    """

    if not output_html:
        output_html = imcluster_io.output.with_suffix(".html")

    output_file(output_html)
    TOOLTIPS = """
    <div>
        <p>@filenames</p>
        <p>Cluster: @cluster</p>
        <img src="data:image/png;base64, @thumbnail{safe}" alt="Thumbnail" />
    </div>
    """

    imcluster_io.df["path"] = [str(x) for x in imcluster_io.images]
    if not imcluster_io.has_column("thumbnail") or force or force_thumbnails:
        print(f"Generating thumbnails within box ({thumbnail_width}x{thumbnail_height})")
        imcluster_io.save_column(
            "thumbnail",
            imcluster_io.df.apply(lambda row: generate_thumbnail(row["path"], thumbnail_width, thumbnail_height), axis=1),
        )

    return
    cmap = Spectral6
    imcluster_io.df["color"] = imcluster_io.df.apply(
        lambda row: cmap[row["dbscan_cluster"] % len(cmap)], axis=1
    )

    p = figure(width=width, height=height, tooltips=TOOLTIPS)
    p.circle(
        "pca0", "pca1", source=imcluster_io.df, size=size, color="color", alpha=alpha
    )
    show(p)
