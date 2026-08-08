import numpy as np
import pytest

from imcluster.features import (
    CONVNEXT_MODELS,
    DINOV2_FALLBACK_MODELS,
    DINOV2_MODELS,
    DINOV3_ACCESS_DOCS,
    VIT_MODELS,
    Device,
    DinoVersion,
    ModelArchitecture,
    ModelSize,
    build_features,
    dinov3_available,
    resolve_device,
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


@pytest.mark.parametrize("device", [Device.CPU, Device.CUDA, Device.MPS])
def test_resolve_device_preserves_explicit_selection(device):
    assert resolve_device(device) == device.value


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr("imcluster.features.torch.cuda.is_available", lambda: True)

    assert resolve_device(Device.AUTO) == "cuda"


def test_resolve_device_auto_uses_mps_without_cuda(monkeypatch):
    monkeypatch.setattr("imcluster.features.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "imcluster.features.torch.backends.mps.is_available", lambda: True
    )

    assert resolve_device(Device.AUTO) == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("imcluster.features.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "imcluster.features.torch.backends.mps.is_available", lambda: False
    )

    assert resolve_device(Device.AUTO) == "cpu"


@pytest.mark.parametrize(
    ("size", "expected"),
    list(VIT_MODELS.items()),
)
def test_resolve_model_maps_all_vit_sizes(size, expected):
    assert (
        resolve_model(None, DinoVersion.THREE, ModelArchitecture.VIT, size) == expected
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    list(CONVNEXT_MODELS.items()),
)
def test_resolve_model_maps_supported_convnext_sizes(size, expected):
    assert (
        resolve_model(None, DinoVersion.THREE, ModelArchitecture.CONVNEXT, size)
        == expected
    )


@pytest.mark.parametrize(("size", "expected"), list(DINOV2_MODELS.items()))
@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_resolve_model_maps_dinov2_sizes_and_ignores_architecture(
    size, expected, architecture
):
    assert resolve_model(None, DinoVersion.TWO, architecture, size) == expected


@pytest.mark.parametrize(("size", "expected"), list(VIT_MODELS.items()))
def test_resolve_model_auto_uses_accessible_dinov3(size, expected, monkeypatch):
    monkeypatch.setattr("imcluster.features.dinov3_available", lambda model: True)

    assert (
        resolve_model(None, DinoVersion.AUTO, ModelArchitecture.VIT, size) == expected
    )


@pytest.mark.parametrize(("size", "expected"), list(DINOV2_FALLBACK_MODELS.items()))
def test_resolve_model_auto_falls_back_to_dinov2(size, expected, monkeypatch):
    monkeypatch.setattr("imcluster.features.dinov3_available", lambda model: False)

    assert (
        resolve_model(None, DinoVersion.AUTO, ModelArchitecture.VIT, size) == expected
    )


def test_auto_fallback_warns_about_dinov3_access(monkeypatch):
    messages = []
    monkeypatch.setattr("imcluster.features.dinov3_available", lambda model: False)
    monkeypatch.setattr("imcluster.features.console.print", messages.append)

    result = resolve_model(
        None,
        DinoVersion.AUTO,
        ModelArchitecture.VIT,
        ModelSize.BASE,
    )

    assert result == "facebook/dinov2-base"
    assert len(messages) == 1
    assert "[bold yellow]Warning:[/bold yellow]" in messages[0]
    assert "access approval and authentication" in messages[0]
    assert DINOV3_ACCESS_DOCS in messages[0]


def test_resolve_model_auto_falls_back_when_dinov3_preset_is_missing():
    assert (
        resolve_model(
            None,
            DinoVersion.AUTO,
            ModelArchitecture.CONVNEXT,
            ModelSize.HUGE,
        )
        == "facebook/dinov2-giant"
    )


def test_dinov3_available_detects_cached_weights(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        "imcluster.features.snapshot_download", lambda *args, **kwargs: str(tmp_path)
    )

    assert dinov3_available("organization/model")


def test_dinov3_available_checks_hub_when_not_cached(monkeypatch):
    monkeypatch.setattr(
        "imcluster.features.snapshot_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not cached")),
    )
    monkeypatch.setattr(
        "imcluster.features.hf_hub_download", lambda *args, **kwargs: "config.json"
    )

    assert dinov3_available("organization/model")


def test_dinov3_available_returns_false_without_cache_or_access(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr("imcluster.features.snapshot_download", unavailable)
    monkeypatch.setattr("imcluster.features.hf_hub_download", unavailable)

    assert not dinov3_available("organization/model")


@pytest.mark.parametrize("size", [ModelSize.HUGE, ModelSize.MAX])
def test_resolve_model_rejects_unavailable_convnext_sizes(size):
    with pytest.raises(
        ValueError,
        match=(
            f"Size '{size.value}' is not available for DINOv3 architecture 'convnext'"
        ),
    ):
        resolve_model(None, DinoVersion.THREE, ModelArchitecture.CONVNEXT, size)


@pytest.mark.parametrize("size", [ModelSize.TINY, ModelSize.HUGE])
def test_resolve_model_rejects_unavailable_dinov2_sizes(size):
    with pytest.raises(
        ValueError,
        match=f"Size '{size.value}' is not available for DINOv2",
    ):
        resolve_model(None, DinoVersion.TWO, ModelArchitecture.VIT, size)


def test_explicit_model_overrides_architecture_and_size():
    assert (
        resolve_model(
            "organization/custom-model",
            DinoVersion.TWO,
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


def test_build_features_rejects_unreadable_image(tmp_path, image_factory, monkeypatch):
    image = image_factory("broken.jpg")
    store = ImclusterIO([image], tmp_path / "results.parquet")
    image.write_bytes(b"not an image")
    monkeypatch.setattr(
        "imcluster.features.pipeline",
        lambda **kwargs: FakeExtractor(),
    )

    with pytest.raises(ValueError, match=r"Cannot read image '.+broken\.jpg'"):
        build_features(store, device=Device.CPU)


def test_build_features_rejects_unpooled_embeddings(
    tmp_path, image_factory, monkeypatch
):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")

    class UnpooledExtractor:
        def __call__(self, images, **kwargs):
            return [[[[1.0, 2.0], [3.0, 4.0]]] for _ in images]

    monkeypatch.setattr(
        "imcluster.features.pipeline",
        lambda **kwargs: UnpooledExtractor(),
    )

    with pytest.raises(ValueError, match="did not return pooled image embeddings"):
        build_features(store, model_name="unpooled-model", device=Device.CPU)
