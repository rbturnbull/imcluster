"""Image input discovery and Parquet-backed result persistence."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
        recursive: Search within nested subdirectories of directory inputs.
        reset_cache: Discard an existing cache instead of loading it.
    """

    def __init__(
        self,
        inputs: Iterable[str | Path],
        output: str | Path,
        max_images: int | None = None,
        recursive: bool = False,
        reset_cache: bool = False,
    ) -> None:
        """Initialize an image collection and load any cached results."""
        self.output: Path = Path(output)

        discovered: list[Path] = []
        for path in inputs:
            path = Path(path).expanduser()

            if path.is_dir():
                candidates = path.rglob("*") if recursive else path.iterdir()
                discovered.extend(sorted(x for x in candidates if valid_image(x)))
            elif path.suffix.lower() == ".txt":
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        value = line.strip()
                        if not value:
                            continue
                        candidate = Path(value).expanduser()
                        if not candidate.is_absolute():
                            candidate = path.parent / candidate
                        if valid_image(candidate):
                            discovered.append(candidate)
            elif valid_image(path):
                discovered.append(path)
            else:
                print(f"File '{path}' does not have a valid extension.")

        # Resolve paths and remove duplicates without changing input order.
        self.images = list(dict.fromkeys(image.resolve() for image in discovered))

        if max_images and len(self.images) > max_images:
            self.images = self.images[:max_images]

        self.filenames: list[str] = [image.name for image in self.images]
        self.paths: list[str] = [str(image) for image in self.images]

        if self.output.exists() and not reset_cache:
            df = pd.read_parquet(self.output)
            if "path" not in df or df["path"].tolist() != self.paths:
                raise ValueError(
                    f"Cached results in '{self.output}' do not match the current "
                    "image inputs. Use --force to replace the cache."
                )
        else:
            df = pd.DataFrame({"path": self.paths, "filenames": self.filenames})

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
