import numpy as np
from PIL import Image
from transformers import pipeline
from rich.progress import track
from rich.console import Console

console = Console()

from .io import ImclusterIO

DEFAULT_MODEL = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"


def build_features(
    imcluster_io: ImclusterIO,
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
):
    """
    Build feature vectors using a Hugging Face image-feature-extraction model.

    Results are cached in a column named after the model.
    """
    model_name = str(model_name)

    if not imcluster_io.has_column(model_name) or force:
        console.print("Setting up dataset")
        feature_extractor = pipeline(
            model=model_name,
            task="image-feature-extraction", 
        )

        results = []
        for path in track(imcluster_io.images, description="Generating feature vectors:"):
            with Image.open(path) as im:
                features = feature_extractor(im)
            tokens = np.array(features[0])
            result = tokens.mean(axis=0)

            results.append(result)
        feature_vectors = np.vstack(results)  # stack into shape (n_images, dim)
        feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)  # row-wise normalization

        imcluster_io.save_column(
            model_name, [feature_vectors[x] for x in range(feature_vectors.shape[0])]
        )
    else:
        console.print("Using precomputed feature vectors")

        feature_vectors = np.array(imcluster_io.get_column(model_name).to_list())

    return feature_vectors
