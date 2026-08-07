"""Image input discovery and Parquet-backed result persistence."""

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
}


def valid_image(path: str | Path) -> bool:
    """Return whether a path is an existing image with a supported suffix."""
    path = Path(path)
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


class ImclusterIO:
    """Manage image inputs and cached tabular results.

    Args:
        inputs: Image files, directories, or text manifests containing one path
            per line.
        output: Parquet file used to persist intermediate and final results.
        max_images: Optional maximum number of images to retain. Zero or
            ``None`` means unlimited.
    """

    def __init__(
        self,
        inputs: Iterable[str | Path],
        output: str | Path,
        max_images: int | None = None,
    ) -> None:
        """Initialize an image collection and load any cached results."""
        self.output: Path = Path(output)

        # Copy image paths into a list
        self.images: list[Path] = []
        for path in inputs:
            path = Path(path)

            # If it is a text file, then read each line as an image
            if path.is_dir():
                self.images += [x for x in path.iterdir() if valid_image(x)]
            elif path.suffix.lower() == ".txt":
                with open(path, encoding="utf-8") as f:
                    paths_in_file = [Path(line.strip()) for line in f.readlines()]
                    self.images += [x for x in paths_in_file if valid_image(x)]
            elif valid_image(path):
                self.images.append(path)
            else:
                print(f"File '{path}' does not have a valid extension.")

        # truncate list of images if the user sets the maximum allowed
        if max_images and len(self.images) > max_images:
            self.images = self.images[:max_images]

        self.filenames: list[str] = [image.name for image in self.images]

        if self.output.exists():
            df = pd.read_parquet(self.output, engine="pyarrow")

            # TODO check that the filenames are the same as the list
        else:
            df = pd.Series(self.filenames, name="filenames").to_frame()

        self.df: pd.DataFrame = df

    def has_column(self, column_name: str) -> bool:
        """Return whether the cached table contains a named column."""
        return column_name in self.df.columns

    def get_all_columns(self) -> list[str]:
        """Return all cached table column names."""
        return self.df.columns.tolist()

    def save(self) -> None:
        """Persist the current result table as Parquet."""
        self.df.to_parquet(self.output, engine="pyarrow")

    def save_column(
        self,
        column_name: str,
        data: Any,
        autosave: bool = True,
    ) -> None:
        """Add or replace a result column and optionally persist the table."""
        self.df[column_name] = data
        if autosave:
            self.save()

    def get_column(self, column_name: str) -> pd.Series:
        """Return a cached result column."""
        return self.df[column_name]
