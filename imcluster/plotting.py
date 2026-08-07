import base64
from io import BytesIO
from PIL import Image
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
    thumbnail_width:int=256,
    thumbnail_height:int=256,
    force: bool = False,
    force_thumbnails:bool = False,
):
    """
    Generate and cache thumbnails used by the HTML cluster report.
    """

    imcluster_io.df["path"] = [str(x) for x in imcluster_io.images]
    if not imcluster_io.has_column("thumbnail") or force or force_thumbnails:
        print(f"Generating thumbnails within box ({thumbnail_width}x{thumbnail_height})")
        imcluster_io.save_column(
            "thumbnail",
            imcluster_io.df.apply(lambda row: generate_thumbnail(row["path"], thumbnail_width, thumbnail_height), axis=1),
        )
