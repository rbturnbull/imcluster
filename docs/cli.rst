======================
Command-line reference
======================

Run ``imcluster --help`` to see the reference for the installed version.

Positional arguments
====================

``INPUTS...``
   Image files, directories, or text manifests to process.

``OUTPUT_DF``
   Destination Parquet file for cached results.

Main options
============

``--output-html PATH``
   Destination path for the standalone HTML cluster gallery.

``--arch [vit|convnext]`` and ``--size [tiny|small|base|large|huge|max]``
   Select a DINOv3 preset. The default is ``vit`` and ``base``.

``--model TEXT``
   Use an arbitrary compatible Hugging Face model instead of the preset.

``--device [auto|cpu|cuda|mps]`` and ``--batch-size INTEGER``
   Control model inference hardware and batching.

``--recursive`` and ``--max-images INTEGER``
   Control input discovery.

``--algorithm [spectral|dbscan]``
   Select clustering behavior. Spectral clustering uses ``--n-clusters``;
   DBSCAN uses ``--dbscan-eps`` and ``--dbscan-min-samples``.

``--thumbnail-width INTEGER`` and ``--thumbnail-height INTEGER``
   Set the maximum embedded thumbnail dimensions.

Cache options
=============

``--force`` replaces every cached stage. ``--force-features``,
``--force-cluster``, and ``--force-thumbnails`` recompute
individual stages and their required downstream results.
