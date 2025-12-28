# PertVCFM: Perturbation-Centric Virutal Cell Foundation Model

This repository contains multiple model implementations for predicting cellular perturbation effects on gene expression. The project includes proof-of-concept models, baseline comparisons, and advanced architectures for perturbation prediction.

## Overview

PertVCFM is a collection of models and baselines for predicting transcriptional responses to genetic and chemical perturbations in single-cell RNA-seq data. The repository includes:

- **Baseline model implementations** for comparison
- **PoC Experiments on various perturbation embeddings**
- **Advanced model architectures** (Random Walk Neural Networks & Transformers)

## Directory Structure

```
PertVCFM/
├── PoC_model/          # Proof-of-concept perturbation models
├── MainModel/          # Advanced model implementations
│   ├── random-walk/    # Random walk neural networks for graph learning
│   └── TransformerModel/  # Transformer-based models for perturbation embedding
└── baselines/          # Baseline model implementations for comparison
```

## Components

### 1. PoC_model/

Contains the main proof-of-concept implementation of perturbation prediction models: Shallow auto encoder model to reconstruct the perturbation effects.

**Key concepts:**
- Graph-structured knowledge integration (STRINGdb, GO)
- Genetic description LLM token embedding integration (NCBI)
- Gene related protein sequence embedding integratioin (ESM)

See [`PoC_model/README.md`](PoC_model/README.md) for detailed documentation and usage instructions.

### 2. MainModel/

Contains advanced model architectures:

#### random-walk/
Random Walk Neural Networks for large scale graph-structured learning. This implementation is based on the ICLR 2025 spotlight paper "Revisiting Random Walks for Learning on Graphs".

- Global genetic path generation using random walk 2025
- Perturbatioin-constraint path generation using random walk 2025
- DeBERTa/LLaMA3 integration for sequence modeling

See [`MainModel/random-walk/README.md`](MainModel/random-walk/README.md) for details.

#### TransformerModel/
Transformer-based models for perturbation embedding learning.

- Gene tokenization and encoding
- Attention mechanisms for gene-gene interactions
- Stage 1: Maksed language modeling pretraining
- Stage 2: Uncertainty-aware Perturbation sequence Alignment learning 
- Stage 3(Finetuning): LoRA finetuning for perturbation prediction

See individual script READMEs in `MainModel/TransformerModel/scripts/` for more learning strategies.


### 3. baselines/

Implementation of baseline models for comparison and evaluation:

- **LowRankLinear (LRM)**: Low-rank matrix factorization approach
- **CPA**: Compositional Perturbation Autoencoder
- **scGPT**: Single-cell GPT model
- **scVI**: Single-cell Variational Inference
- **GEARS**: Gene Expression Analysis via Random Subsets

These baselines are evaluated on standard perturbation datasets (Replogle, Tahoe, Parse) with cross-validation support.

See [`baselines/README.md`](baselines/README.md) for installation and usage instructions.


## Dependencies

### Common Dependencies
- Python 3.12+
- PyTorch 2.4.0+cu124
- PyTorch Lightning
- PyTorch Geometric (for graph models)
- Scanpy / AnnData (for single-cell data)
- Hydra (for configuration management)

### Component-Specific Dependencies

Each component may have additional dependencies. Please refer to:
- `PoC_model/pyproject.toml` for PoC_model dependencies
- `baselines/requirements.txt` for baseline dependencies
- Component-specific README files for other dependencies

## Citation

If you use any component of this repository, please cite the relevant papers:

- **Transformer Architecture**: Refer to [Tahoe-x1 model](https://github.com/tahoebio/tahoe-x1).
-  **Random Walk Neural Networks**: Refer to [arXiv:2407.01214](https://arxiv.org/abs/2407.01214)
- **Baseline models**: Refer to respective original papers (GEARS, CPA, scGPT, scVI, etc.)

## License

Please refer to individual component directories for license information. The PoC_model uses the Recursion Non-Commercial End User License Agreement.
