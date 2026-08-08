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

Clustering
==========

Spectral, K-means, agglomerative, and hierarchical clustering use
``--n-clusters``. DBSCAN uses ``--dbscan-eps``; DBSCAN and HDBSCAN share
``--min-samples`` and report outliers as noise.
