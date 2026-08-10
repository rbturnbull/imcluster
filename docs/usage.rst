=====
Usage
=====

Basic run
=========

.. code-block:: bash

   imcluster photos/

Pass ``--recursive`` to search nested directories. Inputs may also be image
paths or text manifests containing one image path per line.

The gallery opens in the default browser. Use ``--no-open`` to suppress this,
``--gallery PATH`` to preserve the HTML report, and ``--cache PATH`` to retain
the processing cache. Without those paths, temporary outputs are used.

Caching
=======

The Parquet file selected by ``--cache`` caches expensive stages. It can be
reused only with the same ordered set of resolved image paths. Use ``--force``
to replace it, or a stage-specific force option to recompute part of the
pipeline.

Once a cache exists, its stored image paths can be used without repeating the
original inputs:

.. code-block:: bash

   imcluster --cache results.parquet

Clustering
==========

K-means is the default clustering method. Spectral, K-means, agglomerative,
and hierarchical clustering use ``--n-clusters``. DBSCAN uses
``--dbscan-eps``; DBSCAN and HDBSCAN share ``--min-samples`` and report
outliers as noise.

Dimensionality reduction
========================

By default, UMAP reduces the DINO feature vectors before clustering. PCA and
t-SNE can be selected instead, or reduction can be disabled:

.. code-block:: bash

   imcluster photos/ --reduce umap
   imcluster photos/ --reduce tsne
   imcluster photos/ --reduce pca
   imcluster photos/ --reduce none

Use ``--reduction-dims`` to change the default target of 50 dimensions:

.. code-block:: bash

   imcluster photos/ --reduce umap --reduction-dims 25

PCA and t-SNE use scikit-learn. UMAP uses umap-learn. The reduced vectors are
cached, and changing the reduction method or requested dimensions recomputes
cluster assignments. UMAP requires at least three images. t-SNE is capped at
three output dimensions.

Evaluation
==========

Evaluate cluster assignments against expected classes with a CSV containing
``filename`` and ``class`` columns:

.. code-block:: text

   filename,class
   airplane-01.jpg,airplane
   airplane-02.jpg,airplane
   forest-01.jpg,forest

Then pass it to the CLI:

.. code-block:: bash

   imcluster photos/ --evaluate expected_classes.csv

The class values may be any consistent names. imcluster reports Normalized
Mutual Information (NMI), Adjusted Rand Index (ARI), and clustering accuracy
(ACC). ACC optimally matches numeric cluster IDs to expected classes, so the
specific cluster numbers do not affect the score.

Save the scores as a one-row CSV with ``NMI``, ``ARI``, and ``ACC`` columns:

.. code-block:: bash

   imcluster photos/ --evaluate expected_classes.csv --metric metrics.csv
