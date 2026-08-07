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

By default, ``imcluster`` uses DINOv3 when its weights are cached or accessible
and otherwise falls back to DINOv2. Spectral clustering is the default, with
DBSCAN available when the number of groups is not known.

Installation
============

``imcluster`` requires Python 3.10--3.13::

    python -m pip install imcluster

DINOv2 presets are public and require no authentication. DINOv3 model
repositories are gated. Before using ``--dino-version 3``:

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

DINOv2 weights use the Apache License 2.0. DINOv3 weights use the DINOv3
license; the ``imcluster`` source code uses the Apache License 2.0.

Quick start
===========

Cluster the images directly inside a directory and open the gallery::

    imcluster photos/

Include nested directories, request 12 groups, and preserve the outputs::

    imcluster photos/ --recursive --n-clusters 12 \
        --cache results.parquet --gallery clusters.html

Inputs may be individual image files, directories, or UTF-8 text manifests with
one image path per line. Relative manifest entries are resolved from the
manifest's directory. Supported formats are PNG, JPEG, TIFF, BMP, and GIF.

Outputs
-------

Without output options, ``imcluster`` writes temporary processing data and a
temporary HTML gallery, then opens the gallery in the default browser. Pass
``--no-open`` to suppress browser launching.

``--cache PATH`` preserves the Parquet cache, which contains resolved paths,
filenames, feature vectors, cluster labels, thumbnails, and run metadata.
``--gallery PATH`` preserves the standalone HTML gallery. It embeds its styles
and JPEG thumbnails and does not require an internet connection.

If the input list no longer matches an existing cache, ``imcluster`` stops with
a clear error. Pass ``--force`` to intentionally replace the cache. More
targeted controls are available as ``--force-features``, ``--force-cluster``,
and ``--force-thumbnails``.

Models
======

The default selection is ``--dino-version auto --size base``. Automatic mode
uses DINOv3 when the selected model is cached or accessible with the active
Hugging Face account. Otherwise it reports the fallback and uses DINOv2.

Explicit DINOv2 selection uses ``--dino-version 2``. Its presets are ``small``,
``base``, ``large``, and ``max``; ``max`` selects DINOv2 Giant. For DINOv2,
``--arch`` is ignored. In automatic mode, ``tiny`` falls back to DINOv2 Small
and ``huge`` falls back to DINOv2 Giant.

======  ==========================
Size    Hugging Face model
======  ==========================
small   ``facebook/dinov2-small``
base    ``facebook/dinov2-base``
large   ``facebook/dinov2-large``
max     ``facebook/dinov2-giant``
======  ==========================

DINOv3 is selected with ``--dino-version 3``. Its available presets are:

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

    imcluster photos/ --model organization/model-id

Inference
=========

``--device auto`` selects CUDA, then Apple MPS, then CPU. A device can be
selected explicitly with ``--device cpu|cuda|mps``. ``--batch-size`` defaults
to 8; reduce it if inference runs out of memory.

ViT-B is suitable for a quality-oriented default but can be slow on CPU.
``--dino-version 2 --size small`` or ``--dino-version 3 --size tiny`` provides
a lighter run. The largest variants require substantial accelerator memory.

Clustering
==========

Spectral, K-means, agglomerative, and hierarchical clustering use a cluster
count::

    imcluster photos/ --clustering spectral --n-clusters 10

DBSCAN discovers groups and marks outliers as the noise cluster::

    imcluster photos/ --clustering dbscan \
        --dbscan-eps 0.35 --min-samples 3

HDBSCAN also discovers groups and noise while adapting to varying densities::

    imcluster photos/ --clustering hdbscan --min-samples 5

Run ``imcluster --help`` for the complete command-line reference.

Limitations
===========

Model downloads can be large, and the biggest presets are impractical without
a high-memory GPU. Clustering quality depends on the visual domain and chosen
parameters. The models' training data also carries the biases documented by
their authors.

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
