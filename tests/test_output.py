import base64
from io import BytesIO

import pytest
from PIL import Image

from imcluster.html import write_html
from imcluster.io import ImclusterIO
from imcluster.thumbnails import generate_thumbnail, generate_thumbnails


def test_generate_thumbnail_returns_bounded_jpeg(image_factory):
    source = image_factory("wide.png", size=(100, 50))

    encoded = generate_thumbnail(source, width=20, height=20)
    thumbnail = Image.open(BytesIO(base64.b64decode(encoded)))

    assert thumbnail.format == "JPEG"
    assert thumbnail.size == (20, 10)


def test_generate_thumbnail_converts_rgba_to_jpeg(tmp_path):
    source = tmp_path / "transparent.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 100)).save(source)

    encoded = generate_thumbnail(source, width=10, height=10)
    thumbnail = Image.open(BytesIO(base64.b64decode(encoded)))

    assert thumbnail.mode == "RGB"


def test_generate_thumbnail_rejects_unreadable_image(tmp_path):
    source = tmp_path / "broken.jpg"
    source.write_bytes(b"not an image")

    with pytest.raises(
        ValueError,
        match=r"Cannot create thumbnail for '.+broken\.jpg'",
    ):
        generate_thumbnail(source, width=10, height=10)


def test_generate_thumbnails_generates_and_persists_thumbnails(tmp_path, image_factory):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )

    generate_thumbnails(store, thumbnail_width=10, thumbnail_height=10)

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

    write_html(
        store,
        output,
        metadata={
            "Model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "Algorithm": "spectral",
        },
    )
    rendered = output.read_text()

    assert "Cluster 4" in rendered
    assert 'aria-label="Cluster table of contents"' in rendered
    assert 'href="#cluster-4"' in rendered
    assert 'id="cluster-4"' in rendered
    assert rendered.count('<span class="badge">1 image</span>') == 2
    assert "Cluster 4 (1)" not in rendered
    assert "encoded-thumbnail" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.jpg" in rendered
    assert "<script>alert(1)</script>.jpg" not in rendered
    assert "data:image/jpeg;base64," in rendered
    assert "data:image/png;base64," in rendered
    assert 'class="report-header"' in rendered
    assert 'href="https://github.com/rbturnbull/imcluster"' in rendered
    assert f'href="{image.resolve().as_uri()}"' in rendered
    assert 'target="_blank"' in rendered
    assert (
        'href="https://huggingface.co/'
        'facebook/dinov3-vitb16-pretrain-lvd1689m"' in rendered
    )
    assert 'href="https://scikit-learn.org/stable/modules/clustering.html"' in rendered
    assert "cdn.jsdelivr.net" not in rendered


def test_write_html_supports_dbscan_clusters(tmp_path, image_factory):
    store = ImclusterIO(
        [image_factory("one.jpg"), image_factory("two.jpg")],
        tmp_path / "results.parquet",
    )
    store.df["dbscan_cluster"] = [-1, -1]
    store.df["thumbnail"] = ["encoded-thumbnail", "encoded-thumbnail"]
    output = tmp_path / "dbscan.html"

    write_html(store, output, cluster_column="dbscan_cluster")

    rendered = output.read_text()
    assert "Noise" in rendered
    assert rendered.count('<span class="badge">2 images</span>') == 2


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
