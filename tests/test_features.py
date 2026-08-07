import numpy as np

from imcluster.features import build_features
from imcluster.io import ImclusterIO


class FakeExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        # One batch containing one pooled image embedding.
        return [[[3.0, 4.0]]]


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
    assert extractor.calls == [{"pool": True}, {"pool": True}]
