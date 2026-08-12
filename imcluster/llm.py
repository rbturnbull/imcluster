"""LLM-assisted names for image clusters."""

from collections.abc import Sequence
from typing import Any

import llmloader
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from numpy.typing import ArrayLike
from rich.console import Console
from rich.markup import escape

from .io import ImclusterIO

console = Console()

DEFAULT_LLM = "gpt-5.6-luna"


def encode_message(text: str) -> dict[str, str]:
    """Return a multimodal text-content block."""
    return {"type": "text", "text": text}


def encode_thumbnail_message(thumbnail: str) -> dict[str, Any]:
    """Return a multimodal image block for a cached JPEG thumbnail."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{thumbnail}"},
    }


def name_cluster(
    llm: Any,
    in_group: Sequence[str],
    out_group: Sequence[str] | None = None,
) -> str:
    """Generate a concise name from cached thumbnails in and outside a cluster.

    Args:
        llm: LangChain-compatible multimodal chat model.
        in_group: JPEG thumbnails belonging to the cluster, encoded as base64.
        out_group: Optional thumbnails from nearby images outside the cluster.

    Returns:
        A stripped, non-empty cluster name.

    Raises:
        ValueError: If no in-cluster thumbnails or no name is returned.
    """
    if not in_group:
        raise ValueError("At least one in-cluster thumbnail is required")
    if out_group:
        opening = (
            "Name the visual category shared by the images in this cluster. "
            "Images outside the cluster are contrasting examples and should help "
            "you distinguish the category."
        )
        group_heading = "Images inside the cluster:"
    else:
        opening = "Name the visual category shared by these images."
        group_heading = "Images:"
    content: list[str | dict[Any, Any]] = [
        encode_message(opening),
        encode_message(group_heading),
    ]
    content.extend(encode_thumbnail_message(thumbnail) for thumbnail in in_group)
    if out_group:
        content.append(encode_message("Nearby images outside the cluster:"))
        content.extend(encode_thumbnail_message(thumbnail) for thumbnail in out_group)
    content.append(
        encode_message(
            "Respond with only a short descriptive cluster name. Do not add quotes, "
            "a prefix, a sentence, or punctuation."
        )
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=("You assign concise, specific names to clusters of images.")
            ),
            HumanMessage(content=content),
        ]
    )
    response_content = getattr(response, "content", response)
    if isinstance(response_content, list):
        response_content = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in response_content
        )
    name = str(response_content).strip().strip("\"'").strip()
    if not name:
        raise ValueError("The LLM returned an empty cluster name")
    return name


def _normalize_vectors(feature_vectors: ArrayLike, image_count: int) -> Any:
    """Validate and cosine-normalize image feature vectors."""
    vectors = np.asarray(feature_vectors, dtype=float)
    if vectors.ndim != 2 or len(vectors) != image_count:
        raise ValueError("feature_vectors must contain one row per image")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


def _representative_positions(
    positions: Any,
    normalized: Any,
    limit: int,
) -> list[int]:
    """Select a medoid followed by diverse images using farthest-first traversal."""
    cluster_vectors = normalized[positions]
    medoid_offset = int(np.argmax(cluster_vectors @ cluster_vectors.sum(axis=0)))
    selected = [int(positions[medoid_offset])]
    while len(selected) < min(limit, len(positions)):
        similarities = normalized[positions] @ normalized[selected].T
        nearest_selected = similarities.max(axis=1)
        for selected_position in selected:
            nearest_selected[positions == selected_position] = np.inf
        selected.append(int(positions[np.argmin(nearest_selected)]))
    return selected


def _outside_positions(
    cluster_positions: Any,
    representatives: Sequence[int],
    normalized: Any,
    limit: int,
) -> list[int]:
    """Select outside images nearest to any representative image."""
    if limit == 0:
        return []
    outside = np.setdiff1d(
        np.arange(len(normalized)), cluster_positions, assume_unique=True
    )
    if not len(outside):
        return []
    similarities = normalized[outside] @ normalized[list(representatives)].T
    scores = similarities.max(axis=1)
    ranked = np.argsort(-scores, kind="stable")[:limit]
    return [int(position) for position in outside[ranked]]


def name_clusters(
    imcluster_io: ImclusterIO,
    feature_vectors: ArrayLike,
    cluster_column: str,
    llm: Any = DEFAULT_LLM,
    temperature: float = 0.2,
    api_key: str | None = None,
    in_group_size: int = 10,
    out_group_size: int = 0,
    force: bool = False,
) -> dict[object, str]:
    """Name clusters from representative cached thumbnails and save the results.

    Args:
        imcluster_io: Image cache containing cluster labels and thumbnails.
        feature_vectors: Original image embeddings used to choose prompt examples.
        cluster_column: DataFrame column containing cluster assignments.
        llm: Loaded multimodal model or an llmloader model identifier.
        temperature: Sampling temperature used when loading a model identifier.
        api_key: Optional provider API key passed to llmloader.
        in_group_size: Maximum representative thumbnails from inside each cluster.
        out_group_size: Maximum nearby contrasting thumbnails outside each cluster.
        force: Regenerate names even when a complete name cache exists.

    Returns:
        Mapping from cluster labels to their generated display names.

    Raises:
        ValueError: If required columns, vectors, thumbnails, or sizes are invalid.
    """
    if cluster_column not in imcluster_io.df:
        raise ValueError(f"Missing clustering results column: {cluster_column}")
    if "thumbnail" not in imcluster_io.df:
        raise ValueError("Cached thumbnails are required to name clusters")
    if in_group_size < 1:
        raise ValueError("in_group_size must be at least 1")
    if out_group_size < 0:
        raise ValueError("out_group_size must not be negative")

    name_column = f"{cluster_column}_name"
    labels = imcluster_io.df[cluster_column].to_numpy()
    unique_labels = list(dict.fromkeys(labels.tolist()))
    if imcluster_io.has_column(name_column) and not force:
        cached_names: dict[object, str] = {}
        complete = True
        for label in unique_labels:
            values = {
                value.strip()
                for value in imcluster_io.df.loc[labels == label, name_column].tolist()
                if isinstance(value, str) and value.strip()
            }
            if len(values) != 1:
                complete = False
                break
            cached_names[label] = next(iter(values))
        if complete:
            console.print(
                f"[green]Using cached cluster names:[/green] loaded "
                f"{len(cached_names)} names from '{imcluster_io.output}'."
            )
            return cached_names

    normalized = _normalize_vectors(feature_vectors, len(imcluster_io.images))
    thumbnails = imcluster_io.df["thumbnail"].tolist()
    if any(not isinstance(thumbnail, str) or not thumbnail for thumbnail in thumbnails):
        raise ValueError("Every image must have a cached thumbnail before naming")
    if isinstance(llm, str):
        load_kwargs: dict[str, Any] = {"temperature": temperature}
        if api_key is not None:
            load_kwargs["api_key"] = api_key
        llm = llmloader.load(llm, **load_kwargs)

    names: dict[object, str] = {}
    output_names = np.empty(len(labels), dtype=object)
    with console.status("[cyan]Generating descriptive cluster names...[/cyan]"):
        for label in unique_labels:
            cluster_positions = np.flatnonzero(labels == label)
            if label == -1:
                cluster_name = "Noise"
            else:
                representatives = _representative_positions(
                    cluster_positions, normalized, in_group_size
                )
                outside = _outside_positions(
                    cluster_positions, representatives, normalized, out_group_size
                )
                cluster_name = name_cluster(
                    llm,
                    [thumbnails[position] for position in representatives],
                    [thumbnails[position] for position in outside],
                )
            if label == -1:
                display_label = "noise"
            elif isinstance(label, (int, np.integer)):
                display_label = f"cluster {int(label) + 1}"
            else:
                display_label = f"cluster {label}"
            console.print(
                f"[green]Named {escape(display_label)}:[/green] {escape(cluster_name)}"
            )
            names[label] = cluster_name
            output_names[cluster_positions] = cluster_name
    imcluster_io.save_column(name_column, output_names.tolist())
    console.print(
        f"[green]Wrote cluster names:[/green] saved {len(names)} names to "
        f"'{imcluster_io.output}'."
    )
    return names
