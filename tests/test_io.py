from pathlib import Path

import pandas as pd

from imcluster.io import ImclusterIO, valid_image


def test_valid_image_accepts_supported_extensions_case_insensitively(image_factory):
    image = image_factory("photo.PNG")

    assert valid_image(image)
    assert not valid_image(image.parent)
    assert not valid_image(image.parent / "missing.jpg")


def test_collects_images_from_files_directories_and_manifests(tmp_path, image_factory):
    direct = image_factory("direct.jpg")
    directory_image = image_factory("images/nested.png")
    ignored = directory_image.parent / "notes.csv"
    ignored.write_text("not an image")
    listed = image_factory("listed.tiff")
    manifest = tmp_path / "images.txt"
    manifest.write_text(f"{listed}\n{tmp_path / 'missing.png'}\n")

    store = ImclusterIO(
        [direct, directory_image.parent, manifest], tmp_path / "results.parquet"
    )

    assert store.images == [direct, directory_image, listed]
    assert store.filenames == ["direct.jpg", "nested.png", "listed.tiff"]
    assert store.df["filenames"].tolist() == store.filenames


def test_max_images_limits_collected_images(tmp_path, image_factory):
    images = [image_factory(f"{index}.jpg") for index in range(3)]

    store = ImclusterIO(images, tmp_path / "results.parquet", max_images=2)

    assert store.images == images[:2]


def test_invalid_input_prints_message_and_is_not_collected(tmp_path, capsys):
    invalid = tmp_path / "document.csv"
    invalid.write_text("not an image", encoding="utf-8")

    store = ImclusterIO([invalid], tmp_path / "results.parquet")

    assert store.images == []
    assert store.filenames == []
    assert capsys.readouterr().out == (
        f"File '{invalid}' does not have a valid extension.\n"
    )


def test_columns_are_saved_to_and_loaded_from_parquet(tmp_path, image_factory):
    image = image_factory("photo.jpg")
    output = tmp_path / "results.parquet"
    store = ImclusterIO([image], output)

    store.save_column("score", [0.75])
    loaded = ImclusterIO([image], output)

    assert output.is_file()
    pd.testing.assert_series_equal(
        loaded.get_column("score"), pd.Series([0.75], name="score")
    )
    assert loaded.has_column("score")
    assert loaded.get_all_columns() == ["filenames", "score"]
