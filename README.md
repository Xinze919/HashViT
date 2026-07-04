# Beyond Post-Quantization: Native Hash Learning with a Dedicated HASH Token

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
    <img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://github.com/Xinze919/HashViT">
    <img src="https://img.shields.io/badge/Code-HashViT-blue.svg" alt="Code">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>

Official PyTorch implementation of:

> **Beyond Post-Quantization: Native Hash Learning with a Dedicated HASH Token**  
> Xinze Liu, et al.  
> arXiv preprint, 2026

HashViT is a Native Vision Transformer framework for deep image hashing. Instead of
generating binary codes only through a terminal projection, HashViT introduces
a dedicated **HASH token** that evolves inside the transformer and learns
binary-oriented representations natively.

---

## News
- **2026-xx-xx**: 🎉🎉🎉 Paper available on arXiv.
- **2026-07-04**: 🎉🎉🎉 Code released.
---

## Overview

Most existing deep hashing methods follow a post-quantization paradigm: they
first learn continuous visual features and then convert them into binary codes
through a terminal hash layer or binarization operation. This creates a
discrepancy between the continuous feature space used for optimization and the
discrete Hamming space used for retrieval.

HashViT addresses this issue by introducing a dedicated **HASH token** into the
Vision Transformer. The HASH token is decomposed into:

- **Hash Register**: directly used for binary code generation.
- **Semantic Workspace**: preserves auxiliary continuous semantics.
- **Hash Refinement Adapter**: progressively refines the Hash Register across
  transformer layers.

---

## Installation

```bash
git clone https://github.com/Xinze919/HashViT.git
cd HashViT
```

The basic environment follows
[DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch), whose reference
setup uses Python 3.7, PyTorch 1.4, and torchvision 0.5. These versions are
provided only as a reference and are not strict requirements. HashViT can be
used with other compatible Python and PyTorch environments.

In addition to PyTorch and torchvision, the code uses the following components:

- `timm` for Vision Transformer backbones.
- `transformers` for the CLIP text encoder and tokenizer.
- NumPy and SciPy for numerical computation and hash-center construction.
- Pillow and tqdm for image loading and progress reporting.
- Matplotlib and NLTK for analysis, visualization, and semantic utilities.

## Data preparation

Datasets are not included in this repository. Dataset preparation, evaluation
protocols, and split conventions follow the
[DeepHash-pytorch dataset instructions](https://github.com/swuxyj/DeepHash-pytorch#dataset).
Prepare the required datasets locally, then update `config_dataset` in
`utils/tools.py` with the corresponding dataset roots and split-file paths.

For list-based datasets, each line of a split file should contain an image path
followed by its class label or multi-hot labels.

## Training

The shared settings and dataset-specific hyperparameters are defined in
`HashViT.py`. Select a dataset from the command line:

```bash
python HashViT.py --dataset cifar10 --device cuda:0
python HashViT.py --dataset imagenet --device cuda:0
python HashViT.py --dataset nuswide_81_m --device cuda:0
```

Hash-code lengths and proxy initialization can be overridden when needed:

```bash
python HashViT.py \
  --dataset cifar10 \
  --bits 16 32 48 64 \
  --proxy-init text
```

## Acknowledgements

This implementation was developed with reference to
[DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch) by
[swuxyj](https://github.com/swuxyj). We sincerely thank the original author
and contributors for making their deep hashing implementations publicly
available. Several baseline implementations in this repository are inherited
from or adapted from that project, with their original source notices
retained.

## License

This project is released under the [MIT License](LICENSE). The license file
retains the copyright notice for the upstream DeepHash-pytorch code.
