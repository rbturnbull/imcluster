import base64
from io import BytesIO

import pytest
from PIL import Image

from imcluster.html import write_html
from imcluster.io import ImclusterIO
from imcluster.plotting import generate_thumbnail, plot


def test_generate_thumbnail_returns_bounded_jpeg(image_factory):
    source = image_factory("wide.png", size=(100, 50))

    encoded = generate_thumbnail(source, width=20, height=20)
    thumbnail = Image.open(BytesIO(base64.b64decode(encoded)))

    assert thumbnail.format == "JPEG"
    assert thumbnail.size == (20, 10)


def test_plot_generates_and_persists_thumbnails(tmp_path, image_factory):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )

    plot(store, thumbnail_width=10, thumbnail_height=10)

    assert store.has_column("thumbnail")
    assert len(store.get_column("thumbnail")) == 2


def test_write_html_groups_images_by_cluster_and_escapes_filenames(
    tmp_path, image_factory
):
    image = image_factory("unsafe.jpg")
    store = ImclusterIO([image], tmp_path / "results.parquet")
    store.df["filenames"] = ["<script>alert(1)</script>.jpg"]
    store.df["spectral_cluster"] = [3]
    store.df["thumbnail"] = ["encoded-thumbnail"]
    output = tmp_path / "clusters.html"

    write_html(store, output)
    rendered = output.read_text()

    assert "Cluster 3" in rendered
    assert "encoded-thumbnail" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.jpg" in rendered
    assert "<script>alert(1)</script>.jpg" not in rendered


def test_write_html_supports_dbscan_clusters(tmp_path, image_factory):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")
    store.df["dbscan_cluster"] = [-1]
    store.df["thumbnail"] = ["encoded-thumbnail"]
    output = tmp_path / "dbscan.html"

    write_html(store, output, cluster_column="dbscan_cluster")

    assert "Cluster -1" in output.read_text()


def test_write_html_defaults_beside_parquet_output(tmp_path, image_factory):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")
    store.df["spectral_cluster"] = [0]
    store.df["thumbnail"] = ["encoded-thumbnail"]

    write_html(store)

    assert (tmp_path / "results.html").is_file()


def test_write_html_rejects_missing_cluster_column(tmp_path, image_factory):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")

    with pytest.raises(
        ValueError,
        match="^Missing clustering results column: dbscan_cluster$",
    ):
        write_html(store, cluster_column="dbscan_cluster")
