import numpy as np
import pytest

from imcluster.features import (
    CONVNEXT_MODELS,
    VIT_MODELS,
    Device,
    ModelArchitecture,
    ModelSize,
    build_features,
    resolve_model,
)
from imcluster.io import ImclusterIO


class FakeExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        assert all(item.mode == "RGB" for item in image)
        # One pooled image embedding for every input image.
        return [[[3.0, 4.0]] for _ in image]


@pytest.mark.parametrize(
    ("size", "expected"),
    list(VIT_MODELS.items()),
)
def test_resolve_model_maps_all_vit_sizes(size, expected):
    assert resolve_model(None, ModelArchitecture.VIT, size) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    list(CONVNEXT_MODELS.items()),
)
def test_resolve_model_maps_supported_convnext_sizes(size, expected):
    assert resolve_model(None, ModelArchitecture.CONVNEXT, size) == expected


@pytest.mark.parametrize("size", [ModelSize.HUGE, ModelSize.MAX])
def test_resolve_model_rejects_unavailable_convnext_sizes(size):
    with pytest.raises(
        ValueError,
        match=f"Size '{size.value}' is not available for architecture 'convnext'",
    ):
        resolve_model(None, ModelArchitecture.CONVNEXT, size)


def test_explicit_model_overrides_architecture_and_size():
    assert (
        resolve_model(
            "organization/custom-model",
            ModelArchitecture.CONVNEXT,
            ModelSize.MAX,
        )
        == "organization/custom-model"
    )


def test_build_features_normalizes_and_caches_vectors(
    tmp_path, image_factory, monkeypatch
):
    images = [image_factory("one.jpg"), image_factory("two.jpg")]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    calls = []
    extractor = FakeExtractor()

    def fake_pipeline(**kwargs):
        calls.append(kwargs)
        return extractor

    monkeypatch.setattr("imcluster.features.pipeline", fake_pipeline)

    generated = build_features(store, model_name="test-model")
    cached = build_features(store, model_name="test-model")

    np.testing.assert_allclose(generated, [[0.6, 0.8], [0.6, 0.8]])
    np.testing.assert_allclose(cached, generated)
    assert len(calls) == 1
    assert calls[0]["task"] == "image-feature-extraction"
    assert calls[0]["model"] == "test-model"
    assert extractor.calls == [{"pool": True, "batch_size": 8}]


def test_build_features_rejects_invalid_batch_size(tmp_path, image_factory):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        build_features(store, batch_size=0)


def test_build_features_rejects_zero_embedding(tmp_path, image_factory, monkeypatch):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")

    class ZeroExtractor:
        def __call__(self, images, **kwargs):
            return [[[0.0, 0.0]] for _ in images]

    monkeypatch.setattr(
        "imcluster.features.pipeline",
        lambda **kwargs: ZeroExtractor(),
    )

    with pytest.raises(ValueError, match="zero-length embedding"):
        build_features(store, device=Device.CPU)
