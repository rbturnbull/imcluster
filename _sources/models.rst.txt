======
Models
======

The default is DINOv3 ViT-B/16, selected by ``--arch vit --size base``.
Smaller models are faster and need less memory. An arbitrary compatible Hugging
Face model can be supplied with ``--model``, which overrides both preset flags.

The DINOv3 repositories are gated. Accept the model terms and authenticate with
``hf auth login`` before the first run. Model weights are governed by the
DINOv3 license rather than imcluster's Apache License 2.0.

Inference uses pooled, normalized embeddings. ``--device auto`` prefers CUDA,
then Apple MPS, then CPU. Adjust ``--batch-size`` to fit available memory.
