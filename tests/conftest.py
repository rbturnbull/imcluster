from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def image_factory(tmp_path):
    """Create small RGB image fixtures without storing binary files in git."""

    def create(name: str, color=(255, 0, 0), size=(12, 8)) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        return path

    return create
