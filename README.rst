.. image:: docs/assets/imcluster-banner.png
   :alt: imcluster
   :align: center

|pypi| |docs| |tests|

.. |docs| image:: https://github.com/rbturnbull/imcluster/actions/workflows/docs.yml/badge.svg
   :alt: Documentation build status
   :target: https://github.com/rbturnbull/imcluster/actions/workflows/docs.yml
.. |tests| image:: https://github.com/rbturnbull/imcluster/actions/workflows/tests.yml/badge.svg
   :alt: Test suite status
   :target: https://github.com/rbturnbull/imcluster/actions/workflows/tests.yml
.. |pypi| image:: https://img.shields.io/pypi/v/imcluster.svg?color=blue
   :alt: PyPI package version
   :target: https://pypi.org/project/imcluster/

``imcluster`` clusters images using features from pretrained
vision models. It produces a reusable cache and a self-contained HTML
gallery organized by cluster.

The default model is DINOv3 ViT-B/16. Spectral clustering is the default, with
DBSCAN available for collections where the number of groups is not known.

Installation
============

``imcluster`` requires Python 3.10--3.13::

    python -m pip install imcluster

DINOv3 model repositories are gated. Before the first run:

#. Sign in to Hugging Face and open the `DINOv3 ViT-B/16 model page
   <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>`_.
#. Review and accept Meta's DINOv3 license and agree to share the requested
   contact information. Approval is usually automatic, but access can take
   several minutes (often 5--15 minutes) to propagate.
#. Authenticate the machine that will run ``imcluster``::

       hf auth login

   This opens Hugging Face's browser login flow and stores the resulting token
   locally. Confirm the active account with::

       hf auth whoami

For a server or non-interactive environment, create a read token in `Hugging
Face token settings <https://huggingface.co/settings/tokens>`_ and expose it to
the process instead::

    export HF_TOKEN=hf_your_token_here

Never commit a Hugging Face token to the repository or place it directly in a
script. Accepting access on the website and authenticating locally are both
required; a valid token from an account without model access cannot download
the weights.

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

The Parquet output contains resolved paths, filenames, feature vectors, cluster
labels, thumbnails, and run metadata. It is also used as a
cache on subsequent runs. The HTML output is a standalone gallery: it embeds
its styles and JPEG thumbnails and does not require an internet connection.

If the input list no longer matches an existing cache, ``imcluster`` stops with
a clear error. Pass ``--force`` to intentionally replace the cache. More
targeted controls are available as ``--force-features``, ``--force-cluster``,
and ``--force-thumbnails``.

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

Spectral, K-means, agglomerative, and hierarchical clustering use a cluster
count::

    imcluster photos/ results.parquet --clustering spectral --n-clusters 10

DBSCAN discovers groups and marks outliers as the noise cluster::

    imcluster photos/ results.parquet --clustering dbscan \
        --dbscan-eps 0.35 --min-samples 3

HDBSCAN also discovers groups and noise while adapting to varying densities::

    imcluster photos/ results.parquet --clustering hdbscan --min-samples 5

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
    poetry run coverage run -m pytest
    poetry run coverage report --fail-under=80
    poetry run ruff check imcluster tests
    poetry run mypy imcluster
    poetry run sphinx-build -W -b html docs docs/_build/html
    poetry build

See ``CONTRIBUTING.rst`` for the contribution workflow. Issues may be reported
at https://github.com/rbturnbull/imcluster/issues.

Credits
-------

``imcluster`` is maintained by `Robert Turnbull <https://robturnbull.com>`_ at
the `Melbourne Data Analytics Platform <https://www.unimelb.edu.au/mdap>`_.
`Zaher Joukhadar <https://joukhadar.me/>`_ was instrumental in the original
idea, and James
Quang helped implement DINO feature extraction.
