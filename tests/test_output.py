import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from imcluster.html import representative_indices, write_html
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


def test_generate_thumbnails_reports_cached_results(
    tmp_path, image_factory, monkeypatch
):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")
    store.df["thumbnail"] = ["cached-thumbnail"]
    messages = []
    monkeypatch.setattr("imcluster.thumbnails.console.print", messages.append)

    generate_thumbnails(store)

    assert messages == [
        "[green]Using cached thumbnails:[/green] "
        f"loaded 1 thumbnails from '{store.output}'."
    ]


def test_representative_indices_select_cosine_medoids():
    labels = np.array([0, 0, 0, 1])
    features = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]])

    assert representative_indices(labels, features) == {0: 1, 1: 3}


def test_representative_indices_require_one_vector_per_label():
    with pytest.raises(ValueError, match="one row per cluster label"):
        representative_indices([0, 1], [[1.0, 0.0]])


def test_write_html_uses_cluster_medoid_in_contents(tmp_path, image_factory):
    images = [image_factory(f"{index}.jpg") for index in range(3)]
    store = ImclusterIO(images, tmp_path / "results.parquet")
    store.df["spectral_cluster"] = [0, 0, 0]
    store.df["thumbnail"] = ["edge-one", "representative", "edge-two"]
    output = tmp_path / "clusters.html"

    write_html(
        store,
        output,
        feature_vectors=np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )

    rendered = output.read_text()
    assert (
        'class="sidebar-thumbnail flex-shrink-0" '
        'src="data:image/jpeg;base64,representative"' in rendered
    )


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
            "Clustering": "spectral",
            "Images": "1",
        },
    )
    rendered = output.read_text()

    assert "Cluster 4" in rendered
    assert 'aria-label="Cluster table of contents"' in rendered
    assert 'href="#cluster-4"' in rendered
    assert 'id="cluster-4"' in rendered
    assert rendered.count(">1 image</span>") == 1
    assert "Cluster 4 (1)" not in rendered
    assert "encoded-thumbnail" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.jpg" in rendered
    assert "<script>alert(1)</script>.jpg" not in rendered
    assert "data:image/jpeg;base64," in rendered
    assert "data:image/png;base64," in rendered
    assert '<link rel="icon" type="image/png" href="data:image/png;base64,' in rendered
    assert 'class="report-logo"' in rendered
    assert 'class="report-topbar navbar sticky-top bg-light border-bottom' in rendered
    assert (
        'class="logo-column navbar-brand d-flex flex-shrink-0 '
        'justify-content-center m-0"' in rendered
    )
    assert 'id="cluster-contents"' in rendered
    assert ">1 clusters</h2>" in rendered
    assert 'placeholder="Search images..."' in rendered
    assert 'class="gallery-search input-group me-3 me-lg-4"' in rendered
    assert 'class="bi bi-search"' in rendered
    assert 'class="cluster-count badge rounded-pill">1</span>' in rendered
    assert (
        'class="image-item col" data-image-name="'
        '&lt;script&gt;alert(1)&lt;/script&gt;.jpg"' in rendered
    )
    assert 'id="search-previous"' in rendered
    assert 'id="search-next"' in rendered
    assert 'class="bi bi-chevron-left"' in rendered
    assert 'class="bi bi-chevron-right"' in rendered
    assert 'aria-label="Previous match"' in rendered
    assert 'aria-label="Next match"' in rendered
    assert 'id="search-position"' in rendered
    assert 'image.classList.toggle("search-miss"' in rendered
    assert "scrollIntoView" in rendered
    assert "`${matchIndex + 1} of ${matches.length}`" in rendered
    assert "Sidebar colour" not in rendered
    assert "imcluster-sidebar-theme" not in rendered
    assert "Bootstrap v5.3.8" in rendered
    assert 'class="metadata-table table table-sm small mb-0"' in rendered
    assert rendered.count('<th scope="row">') == 4
    assert ">Clustering</th>" in rendered
    assert ">spectral</a>" in rendered
    assert ">Images</th>" in rendered
    assert ">Architecture</th>" not in rendered
    assert ">Generated</th>" in rendered
    assert "UTC" in rendered
    assert 'class="card h-100 shadow-sm"' in rendered
    assert "--imcluster-accent: #5C6BA4" in rendered
    assert 'class="cluster-heading h3 mb-0"' in rendered
    assert 'class="d-flex align-items-baseline gap-5 mb-3"' in rendered
    assert 'class="text-body-secondary fst-italic small">1 image</span>' in rendered
    assert 'class="image-filename card-title h6 mb-0 text-break"' in rendered
    assert 'class="btn btn-sm copy-path flex-shrink-0"' in rendered
    assert 'data-bs-title="Copy full path" aria-label="Copy full path"' in rendered
    assert 'class="bi bi-copy"' in rendered
    assert '<span class="visually-hidden">Copy full path</span>' in rendered
    assert "bootstrap.Tooltip.getOrCreateInstance(element)" in rendered
    assert 'data-path="' in rendered
    assert 'id="path-toast"' in rendered
    assert 'id="path-toast" class="toast fade"' in rendered
    assert "delay: 5000" in rendered
    assert "toastPath.textContent = path" in rendered
    assert "<details" not in rendered
    assert "<summary" not in rendered
    assert 'class="path"' not in rendered
    assert "border-bottom pb-2" not in rendered
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
    assert rendered.count(">2 images</span>") == 1


def test_write_html_defaults_beside_parquet_output(tmp_path, image_factory):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")
    store.df["spectral_cluster"] = [0]
    store.df["thumbnail"] = ["encoded-thumbnail"]

    write_html(store)

    assert (tmp_path / "results.html").is_file()


def test_write_html_creates_parent_directories(tmp_path, image_factory):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")
    store.df["spectral_cluster"] = [0]
    store.df["thumbnail"] = ["encoded-thumbnail"]
    output = tmp_path / "nested" / "gallery" / "clusters.html"

    write_html(store, output)

    assert output.is_file()


def test_write_html_rejects_missing_cluster_column(tmp_path, image_factory):
    store = ImclusterIO([image_factory("one.jpg")], tmp_path / "results.parquet")

    with pytest.raises(
        ValueError,
        match="^Missing clustering results column: dbscan_cluster$",
    ):
        write_html(store, cluster_column="dbscan_cluster")
