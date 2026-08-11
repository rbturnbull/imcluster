.. image:: https://raw.githubusercontent.com/rbturnbull/imcluster/master/docs/assets/imcluster-banner.png
   :alt: imcluster
   :align: center

|pypi| |docs| |coverage| |tests|

.. |pypi| image:: https://img.shields.io/pypi/v/imcluster.svg?color=blue
   :alt: PyPI package version
   :target: https://pypi.org/project/imcluster/

.. |docs| image:: https://github.com/rbturnbull/imcluster/actions/workflows/docs.yml/badge.svg
   :alt: Documentation build status
   :target: https://github.com/rbturnbull/imcluster/actions/workflows/docs.yml

.. |tests| image:: https://github.com/rbturnbull/imcluster/actions/workflows/tests.yml/badge.svg
   :alt: Test suite status
   :target: https://github.com/rbturnbull/imcluster/actions/workflows/tests.yml

.. |coverage| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/74d271aef18559583efb89100748fe11/raw/coverage-badge.json
    :alt: Coverage badge
    :target: https://rbturnbull.github.io/imcluster/coverage/

.. start-quickstart

``imcluster`` clusters images using features from pretrained
vision models. It produces a reusable cache and a self-contained HTML
gallery organized by cluster.

By default, ``imcluster`` uses `DINOv3
<https://huggingface.co/docs/transformers/en/model_doc/dinov3>`_ when its
weights are cached or accessible and otherwise falls back to `DINOv2
<https://huggingface.co/docs/transformers/en/model_doc/dinov2>`_. UMAP and
K-means are the default reduction and clustering methods, with density-based
methods available when the number of groups is not known.

.. image:: https://raw.githubusercontent.com/rbturnbull/imcluster/master/docs/assets/gallery-screenshot.png
   :alt: Example imcluster gallery showing clustered image cards and navigation
   :align: center
   :width: 100%


Installation
============

``imcluster`` requires Python 3.10–3.13:

.. code-block:: bash

    pip install imcluster

DINOv2 presets are public and require no authentication. DINOv3 model
repositories are gated. Before using ``--dino-version 3``:

#. Sign in to Hugging Face and open the `DINOv3 ViT-B/16 model page
   <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>`_.
#. Review and accept Meta's DINOv3 license and agree to share the requested
   contact information. Approval is usually automatic, but access can take
   several minutes (often 5--15 minutes) to propagate.
#. Authenticate the machine that will run ``imcluster``:

   .. code-block:: bash

       hf auth login

   This opens Hugging Face's browser login flow and stores the resulting token
   locally. Confirm the active account with:

   .. code-block:: bash

       hf auth whoami

For a server or non-interactive environment, create a read token in `Hugging
Face token settings <https://huggingface.co/settings/tokens>`_ and expose it to
the process instead:

.. code-block:: bash

    export HF_TOKEN=hf_your_token_here

Never commit a Hugging Face token to the repository or place it directly in a
script. Accepting access on the website and authenticating locally are both
required; a valid token from an account without model access cannot download
the weights.

DINOv2 weights use the Apache License 2.0. DINOv3 weights use the DINOv3
license; the ``imcluster`` source code uses the Apache License 2.0.

Quick start
===========

Cluster the images directly inside a directory and open the gallery:

.. code-block:: bash

    imcluster photos/

Include nested directories, request 12 groups, and preserve the outputs:

.. code-block:: bash

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

After creating a cache, rerun it without repeating the image inputs:

.. code-block:: bash

    imcluster --cache results.parquet

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

An arbitrary compatible Hugging Face model overrides the preset:

.. code-block:: bash

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

The available methods use `scikit-learn's clustering algorithms
<https://scikit-learn.org/stable/modules/clustering.html>`_.

By default, UMAP reduces the DINO vectors and K-means clusters the result. Use
``--reduce tsne`` or ``--reduce pca`` to select another reduction method, or
``--reduce none`` to cluster the original vectors. ``--reduction-dims`` sets
the PCA or UMAP target and defaults to 50. Reduced vectors are cached.

Spectral, K-means, agglomerative, and hierarchical clustering use a cluster
count:

.. code-block:: bash

    imcluster photos/ --clustering spectral --n-clusters 10

DBSCAN discovers groups and marks outliers as the noise cluster:

.. code-block:: bash

    imcluster photos/ --clustering dbscan \
        --dbscan-eps 0.35 --min-samples 3

HDBSCAN also discovers groups and noise while adapting to varying densities:

.. code-block:: bash

    imcluster photos/ --clustering hdbscan --min-samples 5

Name clusters with a multimodal language model:

.. code-block:: bash

    export OPENAI_API_KEY=your-api-key
    imcluster photos/ --name --llm gpt-5.6-luna

Cluster naming sends representative cached thumbnails—not the source image
files—to the configured model. It does not send out-of-cluster examples by
default. The generated names are stored in the cache and displayed in the
gallery while the underlying numeric cluster IDs remain available for
evaluation and reuse. ``--llm-temperature`` controls sampling;
``--llm-api-key`` can pass a key directly, although a provider environment
variable is safer than exposing a secret in shell history.

To evaluate clusters against known classes, provide a CSV with ``filename`` and
``class`` columns:

.. code-block:: bash

    imcluster photos/ --evaluate expected_classes.csv

The class names only need to be consistent. The CLI reports Normalized Mutual
Information (NMI), Adjusted Rand Index (ARI), and clustering accuracy (ACC)
using an optimal mapping between cluster IDs and expected classes.

Add ``--metric metrics.csv`` to save the three scores as a CSV file.

Run ``imcluster --help`` for the complete command-line reference.

.. end-quickstart

Explore similar images
======================

The gallery can compare every image with its nearest visual neighbours. Click
an image card to open the comparison modal: the selected image stays on the
left while the 30 most similar images, ranked using cosine similarity between
the original model feature vectors, are available on the right. Use the
thumbnail strip or arrow buttons to move through the matches.

Click the image on the right to promote it to the selected image. The modal
then updates with that image's nearest neighbours, making it easy to explore
related groups without closing the comparison view.

.. image:: https://raw.githubusercontent.com/rbturnbull/imcluster/master/docs/assets/gallery-similar-items.png
   :alt: imcluster gallery modal comparing a selected image with similar items
   :align: center
   :width: 100%

The modal loads full-resolution originals from their file paths when they are
available and otherwise falls back to the embedded thumbnails. Each thumbnail
is embedded only once in the standalone report and reused by JavaScript.

Limitations
===========

Model downloads can be large, and the biggest presets are impractical without
a high-memory GPU. Clustering quality depends on the visual domain and chosen
parameters. The models' training data also carries the biases documented by
their authors.

Credits
===========

.. start-credits

``imcluster`` is maintained by `Robert Turnbull <https://robturnbull.com>`_ at
the `Melbourne Data Analytics Platform <https://www.unimelb.edu.au/mdap>`_.
`Zaher Joukhadar <https://joukhadar.me/>`_ was instrumental in the original
idea, and James Quang helped implement DINO feature extraction.

.. end-credits
