
# RASS (Readiness-Aware Sample Selection)

Official PyTorch implementation of **"Readiness-Aware Sample Selection for Noisy Labels with Class Imbalance"** (NeurIPS 2026).

## Requirements

The code has been tested with the following environment:

- Python 3.8.10
- PyTorch 1.13.0
- torchvision 0.15.0
- NumPy 1.22.2
- scikit-learn 0.24.2

## Datasets

Experiments are conducted on datasets with synthetic label noise and real-world label noise.

Download the datasets and specify their paths using `--data_path`.

## Training

We provide implementations for both supervised and semi-supervised training.

- **Synthetic noise:** `benchmark/`
- **Real-world noise:** `real_world/`