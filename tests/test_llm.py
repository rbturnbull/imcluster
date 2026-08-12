import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from imcluster.io import ImclusterIO
from imcluster.llm import (
    DEFAULT_LLM,
    encode_message,
    encode_thumbnail_message,
    name_cluster,
    name_clusters,
)
from imcluster.main import main


class FakeLLM:
    """Record multimodal messages without contacting an LLM provider."""

    def __init__(self, response="Coastal Birds"):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.response)


def test_default_llm_is_defined_once_for_api_and_cli():
    assert DEFAULT_LLM
    assert inspect.signature(name_clusters).parameters["llm"].default == DEFAULT_LLM
    assert inspect.signature(main).parameters["llm"].default == DEFAULT_LLM
    assert inspect.signature(name_clusters).parameters["in_group_size"].default == 10
    assert inspect.signature(name_clusters).parameters["out_group_size"].default == 0
    assert inspect.signature(main).parameters["in_group_size"].default == 10
    assert inspect.signature(main).parameters["out_group_size"].default == 0


def make_cluster_store(tmp_path, image_factory):
    """Return a cache with thumbnails and three cluster labels."""
    images = [image_factory(f"{index}.jpg") for index in range(6)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    store.df["kmeans_cluster"] = [0, 0, 0, 1, 1, -1]
    store.df["thumbnail"] = [f"thumbnail-{index}" for index in range(6)]
    features = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.7, 0.3],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
        ]
    )
    return store, features


def test_message_encoders_use_multimodal_content_blocks():
    assert encode_message("hello") == {"type": "text", "text": "hello"}
    assert encode_thumbnail_message("encoded") == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,encoded"},
    }


def test_name_cluster_uses_cached_thumbnails_and_strips_response():
    llm = FakeLLM('  "Coastal Birds"  ')

    result = name_cluster(llm, ["inside-one", "inside-two"], ["outside"])

    assert result == "Coastal Birds"
    content = llm.messages[1].content
    image_urls = [
        block["image_url"]["url"] for block in content if block["type"] == "image_url"
    ]
    assert image_urls == [
        "data:image/jpeg;base64,inside-one",
        "data:image/jpeg;base64,inside-two",
        "data:image/jpeg;base64,outside",
    ]


def test_name_cluster_supports_text_blocks_and_rejects_empty_inputs():
    llm = FakeLLM([{"text": "  Forest Animals  "}])

    assert name_cluster(llm, ["inside"]) == "Forest Animals"
    prompt_text = " ".join(
        block["text"] for block in llm.messages[1].content if block["type"] == "text"
    )
    assert "outside" not in prompt_text
    assert "contrasting" not in prompt_text
    with pytest.raises(ValueError, match="At least one in-cluster"):
        name_cluster(llm, [])
    with pytest.raises(ValueError, match="empty cluster name"):
        name_cluster(FakeLLM("  "), ["inside"])


def test_name_clusters_selects_thumbnails_and_saves_names(
    tmp_path, image_factory, monkeypatch
):
    store, features = make_cluster_store(tmp_path, image_factory)
    calls = []
    messages = []

    def fake_name_cluster(llm, in_group, out_group):
        calls.append((in_group, out_group))
        return f"Named cluster {len(calls)}"

    monkeypatch.setattr("imcluster.llm.name_cluster", fake_name_cluster)
    monkeypatch.setattr("imcluster.llm.console.print", messages.append)

    names = name_clusters(
        store,
        features,
        "kmeans_cluster",
        llm=FakeLLM(),
        in_group_size=2,
        out_group_size=2,
    )

    assert names == {0: "Named cluster 1", 1: "Named cluster 2", -1: "Noise"}
    assert store.df["kmeans_cluster_name"].tolist() == [
        "Named cluster 1",
        "Named cluster 1",
        "Named cluster 1",
        "Named cluster 2",
        "Named cluster 2",
        "Noise",
    ]
    assert len(calls) == 2
    assert all(len(in_group) == 2 for in_group, _ in calls)
    assert all(len(out_group) == 2 for _, out_group in calls)
    assert all(
        value.startswith("thumbnail-")
        for groups in calls
        for group in groups
        for value in group
    )
    assert messages == [
        "[green]Named cluster 1:[/green] Named cluster 1",
        "[green]Named cluster 2:[/green] Named cluster 2",
        "[green]Named noise:[/green] Noise",
        f"[green]Wrote cluster names:[/green] saved 3 names to '{store.output}'.",
    ]


def test_name_clusters_loads_configured_llm(tmp_path, image_factory, monkeypatch):
    store, features = make_cluster_store(tmp_path, image_factory)
    loaded = {}
    fake_llm = FakeLLM()

    def fake_load(model, **kwargs):
        loaded["model"] = model
        loaded.update(kwargs)
        return fake_llm

    monkeypatch.setattr("imcluster.llm.llmloader.load", fake_load)
    monkeypatch.setattr("imcluster.llm.name_cluster", lambda *args: "Wildlife")

    name_clusters(
        store,
        features,
        "kmeans_cluster",
        llm="provider/model",
        temperature=0.4,
        api_key="secret",
        out_group_size=0,
    )

    assert loaded == {
        "model": "provider/model",
        "temperature": 0.4,
        "api_key": "secret",
    }


def test_name_clusters_uses_complete_cached_names(tmp_path, image_factory, monkeypatch):
    store, features = make_cluster_store(tmp_path, image_factory)
    store.df["kmeans_cluster_name"] = ["Birds"] * 3 + ["Trees"] * 2 + ["Noise"]
    messages = []
    monkeypatch.setattr("imcluster.llm.console.print", messages.append)

    names = name_clusters(
        store,
        features,
        "kmeans_cluster",
        llm=pytest.fail,
    )

    assert names == {0: "Birds", 1: "Trees", -1: "Noise"}
    assert "Using cached cluster names" in messages[0]


def test_name_clusters_replaces_incomplete_cache_and_handles_no_outside_images(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    store.df["kmeans_cluster"] = ["group-a", "group-a"]
    store.df["kmeans_cluster_name"] = ["Old name", "Different old name"]
    store.df["thumbnail"] = ["one-thumbnail", "two-thumbnail"]
    observed = []

    def fake_name_cluster(llm, in_group, out_group):
        observed.append(out_group)
        return "New name"

    monkeypatch.setattr("imcluster.llm.name_cluster", fake_name_cluster)

    names = name_clusters(
        store,
        [[1.0, 0.0], [0.9, 0.1]],
        "kmeans_cluster",
        llm=FakeLLM(),
        out_group_size=1,
    )

    assert names == {"group-a": "New name"}
    assert observed == [[]]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_cluster", "Missing clustering results"),
        ("missing_thumbnail", "Cached thumbnails are required"),
        ("in_group_size", "in_group_size must be at least 1"),
        ("out_group_size", "out_group_size must not be negative"),
        ("bad_vectors", "one row per image"),
        ("missing_thumbnail_value", "Every image must have a cached thumbnail"),
    ],
)
def test_name_clusters_validates_inputs(change, message, tmp_path, image_factory):
    store, features = make_cluster_store(tmp_path, image_factory)
    kwargs = {}
    cluster_column = "kmeans_cluster"
    if change == "missing_cluster":
        cluster_column = "missing"
    elif change == "missing_thumbnail":
        store.df.drop(columns=["thumbnail"], inplace=True)
    elif change == "in_group_size":
        kwargs["in_group_size"] = 0
    elif change == "out_group_size":
        kwargs["out_group_size"] = -1
    elif change == "bad_vectors":
        features = features[:-1]
    else:
        store.df.loc[0, "thumbnail"] = ""

    with pytest.raises(ValueError, match=message):
        name_clusters(
            store,
            features,
            cluster_column,
            llm=FakeLLM(),
            **kwargs,
        )
