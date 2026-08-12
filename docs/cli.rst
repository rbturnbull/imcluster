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
   The default is ``kmeans``.

``--reduce [none|umap|tsne|pca]``
   Optionally reduce DINO feature dimensions before clustering. Reduced vectors
   are cached. The default is ``umap``; use ``--reduce none`` to disable it.

``--reduction-dims INTEGER``
   Set the requested output dimensionality for PCA and UMAP. The default is
   ``50``. t-SNE is capped at three dimensions.

``--name``
   Generate and cache descriptive cluster names from representative cached
   thumbnails using a multimodal LLM.

``--llm TEXT`` and ``--llm-temperature FLOAT``
   Select the llmloader-compatible naming model and its sampling temperature.
   Defaults are ``gpt-5.6-luna`` and ``0.2``.

``--llm-api-key TEXT``
   Pass a provider API key for cluster naming. Prefer the provider's environment
   variable to avoid exposing a secret in shell history or process listings.

``--in-group-size INTEGER`` and ``--out-group-size INTEGER``
   Set the maximum in-cluster and contrasting outside thumbnails sent to the
   naming LLM. The defaults are ``10`` and ``0`` respectively.

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
