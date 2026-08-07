"""Feature extraction using pretrained Hugging Face vision models."""

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError
from rich.console import Console
from rich.progress import track
from transformers import pipeline

from .io import ImclusterIO

console = Console()


class ModelArchitecture(str, Enum):
    """Supported DINOv3 architecture families."""

    VIT = "vit"
    CONVNEXT = "convnext"


class DinoVersion(str, Enum):
    """Supported DINO model generations."""

    AUTO = "auto"
    TWO = "2"
    THREE = "3"


class ModelSize(str, Enum):
    """Supported model-size tiers."""

    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    HUGE = "huge"
    MAX = "max"


class Device(str, Enum):
    """Supported inference devices."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


VIT_MODELS = {
    ModelSize.TINY: "facebook/dinov3-vits16-pretrain-lvd1689m",
    ModelSize.SMALL: "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    ModelSize.BASE: "facebook/dinov3-vitb16-pretrain-lvd1689m",
    ModelSize.LARGE: "facebook/dinov3-vitl16-pretrain-lvd1689m",
    ModelSize.HUGE: "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    ModelSize.MAX: "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

CONVNEXT_MODELS = {
    ModelSize.TINY: "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    ModelSize.SMALL: "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    ModelSize.BASE: "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    ModelSize.LARGE: "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}

DINOV2_MODELS = {
    ModelSize.SMALL: "facebook/dinov2-small",
    ModelSize.BASE: "facebook/dinov2-base",
    ModelSize.LARGE: "facebook/dinov2-large",
    ModelSize.MAX: "facebook/dinov2-giant",
}

DINOV2_FALLBACK_MODELS = {
    ModelSize.TINY: DINOV2_MODELS[ModelSize.SMALL],
    ModelSize.SMALL: DINOV2_MODELS[ModelSize.SMALL],
    ModelSize.BASE: DINOV2_MODELS[ModelSize.BASE],
    ModelSize.LARGE: DINOV2_MODELS[ModelSize.LARGE],
    ModelSize.HUGE: DINOV2_MODELS[ModelSize.MAX],
    ModelSize.MAX: DINOV2_MODELS[ModelSize.MAX],
}

DEFAULT_MODEL = DINOV2_MODELS[ModelSize.BASE]


def dinov3_available(model_name: str) -> bool:
    """Return whether a DINOv3 model is cached or accessible from the Hub."""
    try:
        snapshot = Path(snapshot_download(model_name, local_files_only=True))
        if any(snapshot.glob("*.safetensors")) or any(
            snapshot.glob("pytorch_model*.bin")
        ):
            return True
    except (HfHubHTTPError, OSError):
        pass

    try:
        hf_hub_download(model_name, "config.json")
    except (HfHubHTTPError, OSError):
        return False
    return True


def resolve_device(device: Device) -> str:
    """Resolve automatic inference device selection."""
    if device is not Device.AUTO:
        return device.value
    if torch.cuda.is_available():
        return Device.CUDA.value
    if torch.backends.mps.is_available():
        return Device.MPS.value
    return Device.CPU.value


def resolve_model(
    model: str | None,
    dino_version: DinoVersion,
    architecture: ModelArchitecture,
    size: ModelSize,
) -> str:
    """Resolve a custom model or DINO architecture-size preset.

    Args:
        model: Explicit Hugging Face model ID, which takes precedence when set.
        dino_version: DINO model generation used for preset selection.
        architecture: ViT or ConvNeXt architecture family.
        size: Requested model-size tier.

    Returns:
        A Hugging Face model identifier.

    Raises:
        ValueError: If the requested size is unavailable for the architecture.
    """
    if model:
        return model

    if dino_version is DinoVersion.TWO:
        models = DINOV2_MODELS
    else:
        models = (
            VIT_MODELS if architecture is ModelArchitecture.VIT else CONVNEXT_MODELS
        )
    try:
        selected_model = models[size]
    except KeyError as error:
        if dino_version is DinoVersion.AUTO:
            fallback = DINOV2_FALLBACK_MODELS[size]
            console.print(
                f"DINOv3 has no {architecture.value}/{size.value} preset; "
                f"using {fallback}"
            )
            return fallback
        selection = (
            "DINOv2"
            if dino_version is DinoVersion.TWO
            else f"DINOv3 architecture '{architecture.value}'"
        )
        raise ValueError(
            f"Size '{size.value}' is not available for {selection}"
        ) from error

    if dino_version is DinoVersion.AUTO and not dinov3_available(selected_model):
        fallback = DINOV2_FALLBACK_MODELS[size]
        console.print(f"DINOv3 is unavailable; using {fallback}")
        return fallback
    return selected_model


def build_features(
    imcluster_io: ImclusterIO,
    model_name: str = DEFAULT_MODEL,
    device: Device = Device.AUTO,
    batch_size: int = 8,
    force: bool = False,
) -> NDArray[Any]:
    """Build or load normalized image feature vectors.

    Args:
        imcluster_io: Image collection and its persisted result table.
        model_name: Hugging Face image-feature-extraction model identifier.
        device: Device used for model inference.
        batch_size: Number of images submitted for each inference call.
        force: Rebuild vectors even when a cached model column exists.

    Returns:
        A two-dimensional array with one normalized feature vector per image.

    Notes:
        Results are cached in a column named after ``model_name``.
    """
    model_name = str(model_name)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if not imcluster_io.has_column(model_name) or force:
        console.print("Setting up dataset")
        feature_extractor = pipeline(
            model=model_name,
            task="image-feature-extraction",
            device=resolve_device(device),
        )

        results: list[NDArray[Any]] = []
        image_batches = [
            imcluster_io.images[index : index + batch_size]
            for index in range(0, len(imcluster_io.images), batch_size)
        ]
        for paths in track(image_batches, description="Generating feature vectors:"):
            images = []
            for path in paths:
                try:
                    with Image.open(path) as image:
                        images.append(
                            ImageOps.exif_transpose(image).convert("RGB").copy()
                        )
                except (OSError, UnidentifiedImageError) as error:
                    raise ValueError(f"Cannot read image '{path}': {error}") from error

            outputs = feature_extractor(images, pool=True, batch_size=batch_size)
            for output in outputs:
                result = np.asarray(output)
                while result.ndim > 1 and result.shape[0] == 1:
                    result = result[0]
                if result.ndim != 1:
                    raise ValueError(
                        f"Model '{model_name}' did not return pooled image embeddings"
                    )
                results.append(result)

        feature_vectors = np.vstack(results)
        norms = np.linalg.norm(
            feature_vectors,
            axis=1,
            keepdims=True,
        )
        if np.any(norms == 0):
            raise ValueError(f"Model '{model_name}' returned a zero-length embedding")
        feature_vectors /= norms

        imcluster_io.save_column(
            model_name, [feature_vectors[x] for x in range(feature_vectors.shape[0])]
        )
    else:
        console.print("Using precomputed feature vectors")

        feature_vectors = np.array(imcluster_io.get_column(model_name).to_list())

    return feature_vectors
