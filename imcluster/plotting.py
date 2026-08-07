"""Thumbnail generation for cluster reports."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .io import ImclusterIO


def generate_thumbnail(path: str | Path, width: int, height: int) -> str:
    """Create a base64-encoded JPEG thumbnail bounded by given dimensions.

    Args:
        path: Source image path.
        width: Maximum thumbnail width in pixels.
        height: Maximum thumbnail height in pixels.

    Returns:
        ASCII base64 data for the generated JPEG.
    """
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot create thumbnail for '{path}': {error}") from error
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def plot(
    imcluster_io: ImclusterIO,
    thumbnail_width: int = 256,
    thumbnail_height: int = 256,
    force: bool = False,
    force_thumbnails: bool = False,
) -> None:
    """Generate and cache thumbnails used by the HTML cluster report.

    Args:
        imcluster_io: Image collection and its persisted result table.
        thumbnail_width: Maximum thumbnail width in pixels.
        thumbnail_height: Maximum thumbnail height in pixels.
        force: Regenerate thumbnails regardless of cached data.
        force_thumbnails: Regenerate only the thumbnail cache.
    """

    if not imcluster_io.has_column("thumbnail") or force or force_thumbnails:
        print(
            f"Generating thumbnails within box ({thumbnail_width}x{thumbnail_height})"
        )
        imcluster_io.save_column(
            "thumbnail",
            imcluster_io.df.apply(
                lambda row: generate_thumbnail(
                    row["path"],
                    thumbnail_width,
                    thumbnail_height,
                ),
                axis=1,
            ),
        )
