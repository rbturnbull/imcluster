=========
imcluster
=========

``imcluster`` groups image collections using normalized features from pretrained
vision models. It produces a reusable Parquet cache and a self-contained HTML
gallery organized by cluster.

The default model is DINOv3 ViT-B/16. Spectral clustering is the default, with
DBSCAN available for collections where the number of groups is not known.

Installation
============

``imcluster`` requires Python 3.10--3.13::

    python -m pip install imcluster

DINOv3 model repositories are gated. Before the first run, accept Meta's model
terms on Hugging Face and authenticate locally::

    hf auth login

Model weights use the DINOv3 license; the ``imcluster`` source code uses the
Apache License 2.0.

Quick start
===========

Cluster the images directly inside a directory into 20 groups::

    imcluster photos/ results.parquet --output-html clusters.html

Include nested directories and request 12 groups::

    imcluster photos/ results.parquet --recursive --n-clusters 12

Inputs may be individual image files, directories, or UTF-8 text manifests with
one image path per line. Relative manifest entries are resolved from the
manifest's directory. Supported formats are PNG, JPEG, TIFF, BMP, and GIF.

Outputs
-------

The Parquet output contains resolved paths, filenames, feature vectors, PCA
coordinates, cluster labels, thumbnails, and run metadata. It is also used as a
cache on subsequent runs. The HTML output is a standalone gallery: it embeds
its styles and JPEG thumbnails and does not require an internet connection.

If the input list no longer matches an existing cache, ``imcluster`` stops with
a clear error. Pass ``--force`` to intentionally replace the cache. More
targeted controls are available as ``--force-features``, ``--force-pca``,
``--force-cluster``, and ``--force-thumbnails``.

Models
======

The default selection is ``--arch vit --size base``. Available presets are:

===============  ======  ====================================================
Architecture     Size    Hugging Face model
===============  ======  ====================================================
``vit``          tiny    ``facebook/dinov3-vits16-pretrain-lvd1689m``
``vit``          small   ``facebook/dinov3-vits16plus-pretrain-lvd1689m``
``vit``          base    ``facebook/dinov3-vitb16-pretrain-lvd1689m``
``vit``          large   ``facebook/dinov3-vitl16-pretrain-lvd1689m``
``vit``          huge    ``facebook/dinov3-vith16plus-pretrain-lvd1689m``
``vit``          max     ``facebook/dinov3-vit7b16-pretrain-lvd1689m``
``convnext``     tiny    ``facebook/dinov3-convnext-tiny-pretrain-lvd1689m``
``convnext``     small   ``facebook/dinov3-convnext-small-pretrain-lvd1689m``
``convnext``     base    ``facebook/dinov3-convnext-base-pretrain-lvd1689m``
``convnext``     large   ``facebook/dinov3-convnext-large-pretrain-lvd1689m``
===============  ======  ====================================================

An arbitrary compatible Hugging Face model overrides the preset::

    imcluster photos/ results.parquet --model organization/model-id

Inference
=========

``--device auto`` selects CUDA, then Apple MPS, then CPU. A device can be
selected explicitly with ``--device cpu|cuda|mps``. ``--batch-size`` defaults
to 8; reduce it if inference runs out of memory.

ViT-B is suitable for a quality-oriented default but can be slow on CPU. Use
``--size tiny`` or ``--arch convnext --size tiny`` for a lighter run. The
``huge`` and ``max`` ViT variants require substantial accelerator memory.

Clustering
==========

Spectral clustering requires a cluster count::

    imcluster photos/ results.parquet --algorithm spectral --n-clusters 10

DBSCAN discovers groups and marks outliers as the noise cluster::

    imcluster photos/ results.parquet --algorithm dbscan \
        --dbscan-eps 0.35 --dbscan-min-samples 3

Run ``imcluster --help`` for the complete command-line reference.

Limitations
===========

Model downloads can be large, and the biggest presets are impractical without
a high-memory GPU. Clustering quality depends on the visual domain and chosen
parameters. DINOv3's training data also carries the biases documented by its
model authors.

Development
===========

Clone the repository, install Poetry, and run::

    poetry install
    poetry run pytest
    poetry run ruff check imcluster tests
    poetry run mypy imcluster
    poetry run sphinx-build -W -b html docs docs/_build/html
    poetry build

See ``CONTRIBUTING.rst`` for the contribution workflow. Issues may be reported
at https://github.com/rbturnbull/imcluster/issues.

Credits
-------

``imcluster`` is maintained by Robert Turnbull at the Melbourne Data Analytics
Platform. Zaher Joukhadar was instrumental in the original idea, and James
Quang helped implement DINO feature extraction.
