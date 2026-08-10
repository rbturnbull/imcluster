======================
Command-line reference
======================

Run ``imcluster --help`` to see the reference for the installed version.

Positional arguments
====================

``INPUTS...``
   Optional image files, directories, or text manifests to process. Inputs may
   be omitted when ``--cache`` names an existing imcluster cache.

Main options
============

``--cache PATH``
   Preserve processing results in a reusable Parquet cache.

``--gallery PATH``
   Preserve the standalone HTML cluster gallery at this path.

``--no-open``
   Do not launch the generated gallery in the default browser.

``--dino-version [auto|2|3]``
   Select the DINO generation. Automatic mode prefers an accessible or cached
   DINOv3 model and otherwise falls back to DINOv2.

``--arch [vit|convnext]`` and ``--size [tiny|small|base|large|huge|max]``
   Select a model preset. DINOv2 ignores ``--arch`` and supports ``small``,
   ``base``, ``large``, and ``max`` (Giant). DINOv3 uses both options.

``--model TEXT``
   Use an arbitrary compatible Hugging Face model instead of the preset.

``--device [auto|cpu|cuda|mps]`` and ``--batch-size INTEGER``
   Control model inference hardware and batching.

``--recursive`` and ``--max-images INTEGER``
   Control input discovery.

``--clustering [spectral|dbscan|hdbscan|kmeans|agglomerative|hierarchical]``
   Select clustering behavior. Fixed-count methods use ``--n-clusters``;
   DBSCAN uses ``--dbscan-eps``. DBSCAN and HDBSCAN share ``--min-samples``.

``--thumbnail-width INTEGER`` and ``--thumbnail-height INTEGER``
   Set the maximum embedded thumbnail dimensions.

``--evaluate PATH`` (alias: ``--expected PATH``)
   Evaluate clustering against a CSV with ``filename,class`` columns and print
   Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), and
   optimally matched clustering accuracy (ACC).

``--metric PATH``
   Write NMI, ARI, and ACC scores to a CSV. Requires ``--evaluate``.

Cache options
=============

``--force`` replaces every cached stage. ``--force-features``,
``--force-cluster``, and ``--force-thumbnails`` recompute
individual stages and their required downstream results.
