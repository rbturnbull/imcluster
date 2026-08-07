======
Models
======

The default is DINOv3 ViT-B/16, selected by ``--arch vit --size base``.
Smaller models are faster and need less memory. An arbitrary compatible Hugging
Face model can be supplied with ``--model``, which overrides both preset flags.

Getting access to DINOv3
------------------------

The DINOv3 repositories are gated. Before the first run:

#. Sign in to Hugging Face and open the `DINOv3 ViT-B/16 model page
   <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>`_.
#. Review and accept Meta's DINOv3 license and agree to share the requested
   contact information. Approval is usually automatic, but access can take
   several minutes (often 5--15 minutes) to propagate.
#. Authenticate locally and verify the active account::

      hf auth login
      hf auth whoami

The login command uses a browser flow and stores a token locally. On a server
or in another non-interactive environment, create a read token in `Hugging
Face token settings <https://huggingface.co/settings/tokens>`_ and provide it
through the environment::

   export HF_TOKEN=hf_your_token_here

Do not commit the token or embed it in scripts. The token must belong to the
same Hugging Face account that accepted the DINOv3 terms. Model weights are
governed by the DINOv3 license rather than imcluster's Apache License 2.0.

Inference uses pooled, normalized embeddings. ``--device auto`` prefers CUDA,
then Apple MPS, then CPU. Adjust ``--batch-size`` to fit available memory.
