"""Feature extraction using pretrained Hugging Face vision models."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from rich.console import Console
from rich.progress import track
from transformers import pipeline

from .io import ImclusterIO

console = Console()

DEFAULT_MODEL = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"


def build_features(
    imcluster_io: ImclusterIO,
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
) -> NDArray[Any]:
    """Build or load normalized image feature vectors.

    Args:
        imcluster_io: Image collection and its persisted result table.
        model_name: Hugging Face image-feature-extraction model identifier.
        force: Rebuild vectors even when a cached model column exists.

    Returns:
        A two-dimensional array with one normalized feature vector per image.

    Notes:
        Results are cached in a column named after ``model_name``.
    """
    model_name = str(model_name)

    if not imcluster_io.has_column(model_name) or force:
        console.print("Setting up dataset")
        feature_extractor = pipeline(
            model=model_name,
            task="image-feature-extraction",
        )

        results: list[NDArray[Any]] = []
        for path in track(
            imcluster_io.images,
            description="Generating feature vectors:",
        ):
            with Image.open(path) as im:
                features = feature_extractor(im, pool=True)
            result = np.asarray(features[0])

            results.append(result)
        feature_vectors = np.vstack(results)
        feature_vectors /= np.linalg.norm(
            feature_vectors,
            axis=1,
            keepdims=True,
        )

        imcluster_io.save_column(
            model_name, [feature_vectors[x] for x in range(feature_vectors.shape[0])]
        )
    else:
        console.print("Using precomputed feature vectors")

        feature_vectors = np.array(imcluster_io.get_column(model_name).to_list())

    return feature_vectors
