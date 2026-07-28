# Equivariant Meta-Neural Networks for Learning in Weight Space: From 3D Vision to Operator Learning for Solving Differential Equations

> **Research repository for learning directly on neural network parameters using Scale-Equivariant Graph Meta-Networks (ScaleGMN).**

This repository extends the original **ScaleGMN (NeurIPS 2024)** implementation with additional research contributions on **3D implicit neural representations (INRs), scientific machine learning, neural operators, and computational fluid dynamics (CFD)**.

Instead of operating on images, meshes, or simulation grids, ScaleGMN learns directly from the **weights of neural networks**, enabling machine learning in weight space.

---

# Overview

This repository contains implementations for:

- ScaleGMN for weight-space learning
- Graph Meta-Networks (GMNs)
- INR classification
- INR editing
- Generalization prediction
- Neural operators over weight space
- CFD surrogate modeling
- Navier–Stokes experiments
- GRAM / IFW competition experiments

The repository accompanies research on extending ScaleGMN beyond computer vision toward scientific machine learning and PDE surrogate models.

---

# Research Directions

Current applications include:

- 3D object classification from INR weights
- Weight-space representation learning
- Scientific machine learning
- Neural operators
- Navier–Stokes surrogate modeling
- Computational fluid dynamics
- Time-dependent flow prediction

---

# Repository Structure

```text
.
├── src/                           Core ScaleGMN implementation
├── configs/                       Experiment configurations
├── helpers/                       Dataset preparation utilities
│
├── inr_classification.py          INR classification
├── inr_editing.py                 INR editing
├── predicting_generalization.py   Generalization prediction
│
├── scalegmn_iclr_2026_gram.py     Weight-space neural operator
├── physics_neural_operator_slicing.py
├── ifw-baseline.py
│
├── gram-inference-competition.py
├── inference_scalegmn_navier.py
├── gram-inf-viz.py
│
├── checkpoints/
├── outputs/
└── environment.yml
```

---

# Supported Experiments

## 1. Weight-space Learning

- INR classification
- Model editing
- Generalization prediction

---

## 2. Scientific Machine Learning

Learning mappings directly between neural fields representing physical systems.

Applications include:

- Pipe geometry → velocity prediction
- Neural operators
- PDE surrogate models

---

## 3. GRAM Competition

Implementation of ScaleGMN for the GRAM benchmark on transient airflow prediction.

Pipeline:

```text
Input INR
      │
      ▼
ScaleGMN
      │
      ▼
Predicted INR
      │
      ▼
Continuous Physical Field
```

---

# Installation

```bash
git clone <repository>

conda env create -f environment.yml
conda activate scalegmn
```

---

# Typical Workflow

```text
Dataset
     │
     ▼
Train INRs
     │
     ▼
Extract Network Weights
     │
     ▼
Construct Graph
     │
     ▼
ScaleGMN
     │
     ▼
Predict Weight Space
     │
     ▼
Reconstruct Continuous Field
```

---

# Main Scripts

### Weight-space Tasks

- `inr_classification.py`
- `inr_editing.py`
- `predicting_generalization.py`

### Neural Operators

- `scalegmn_iclr_2026_gram.py`
- `physics_neural_operator_slicing.py`

### Inference

- `gram-inference-competition.py`
- `inference_scalegmn_navier.py`

### Visualization

- `gram-inf-viz.py`

---

# Datasets

The repository supports experiments on:

| Dataset | Task |
|---------|------|
| MNIST-INRs | Classification |
| FashionMNIST-INRs | Classification |
| CIFAR10-INRs | Classification |
| ShapeNet | 3D INR classification |
| Pipe (Navier–Stokes) | Neural operator |
| GRAM / IFW | Time-dependent CFD |

---

# Research Contributions

Compared to the original ScaleGMN implementation, this repository additionally contains:

- Scientific machine learning experiments
- Neural operator architectures in weight space
- CFD pipelines
- Navier–Stokes benchmarks
- GRAM competition implementation
- Extended inference and visualization utilities

---

# Citation

If you use this repository, please cite the original ScaleGMN paper together with any associated publications built upon this codebase.

```bibtex
@inproceedings{kalogeropoulos2024scalegmn,
  title={Scale Equivariant Graph MetaNetworks},
  author={Kalogeropoulos, Ioannis and Bouritsas, Giorgos and Panagakis, Yannis},
  booktitle={NeurIPS},
  year={2024}
}
```

---

# License

This repository is released under the terms of the LICENSE file.
