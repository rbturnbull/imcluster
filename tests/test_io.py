import pandas as pd
import pytest

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


def test_directory_search_is_not_recursive_by_default(tmp_path, image_factory):
    direct = image_factory("images/direct.jpg")
    image_factory("images/nested/child.jpg")

    store = ImclusterIO([direct.parent], tmp_path / "results.parquet")

    assert store.images == [direct]


def test_recursive_directory_search_collects_nested_images(tmp_path, image_factory):
    direct = image_factory("images/direct.jpg")
    nested = image_factory("images/one/two/nested.jpg")

    store = ImclusterIO(
        [direct.parent],
        tmp_path / "results.parquet",
        recursive=True,
    )

    assert set(store.images) == {direct, nested}


def test_directory_images_are_sorted_and_duplicates_removed(tmp_path, image_factory):
    second = image_factory("images/b.jpg")
    first = image_factory("images/a.jpg")

    store = ImclusterIO(
        [first.parent, second],
        tmp_path / "results.parquet",
    )

    assert store.images == [first.resolve(), second.resolve()]


def test_manifest_paths_are_relative_to_manifest(tmp_path, image_factory):
    image = image_factory("collection/image.jpg")
    manifest = image.parent / "images.txt"
    manifest.write_text("\nimage.jpg\n   \n", encoding="utf-8")

    store = ImclusterIO([manifest], tmp_path / "results.parquet")

    assert store.images == [image.resolve()]


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
    assert loaded.get_all_columns() == ["path", "filenames", "score"]


def test_cache_must_match_current_images(tmp_path, image_factory):
    first = image_factory("first.jpg")
    second = image_factory("second.jpg")
    output = tmp_path / "results.parquet"
    ImclusterIO([first], output).save()

    with pytest.raises(ValueError, match="do not match the current image inputs"):
        ImclusterIO([second], output)


def test_reset_cache_accepts_changed_images(tmp_path, image_factory):
    first = image_factory("first.jpg")
    second = image_factory("second.jpg")
    output = tmp_path / "results.parquet"
    ImclusterIO([first], output).save()

    store = ImclusterIO([second], output, reset_cache=True)

    assert store.images == [second.resolve()]
    assert store.get_all_columns() == ["path", "filenames"]
