=====
Usage
=====

Basic run
=========

::

   imcluster photos/ results.parquet --output-html clusters.html

Pass ``--recursive`` to search nested directories. Inputs may also be image
paths or text manifests containing one image path per line.

Caching
=======

The Parquet output caches expensive stages. It can be reused only with the same
ordered set of resolved image paths. Use ``--force`` to replace it, or a
stage-specific force option to recompute part of the pipeline.

Clustering
==========

Spectral clustering uses ``--n-clusters``. DBSCAN instead uses
``--dbscan-eps`` and ``--dbscan-min-samples`` and reports outliers as noise.
