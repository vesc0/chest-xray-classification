# NIH Chest X-Ray Classification

This project implements a multi-label classification pipeline for the NIH Chest X-ray dataset.

## Setup

```bash
pip install -r requirements.txt
```

The pipeline expects the NIH dataset directory (containing `Data_Entry_2017.csv`,
`train_val_list.txt`, `test_list.txt`, and the `images_001/` … `images_012/` folders).
Point it at your copy with:

```bash
export XRAY_DATASET_ROOT=/path/to/archive-chest-xrays-nih
```

Without this variable the default in `config.py` is used. If the dataset cannot be
found, or too few of the images listed in the CSV resolve on disk, the run fails
immediately rather than training on a partial dataset.

## Dataflow

1. **Data Indexing & Loading**: Images and metadata are parsed from the NIH dataset (`dataset.py`).
2. **Model Initialization**: Pre-trained backbones (CNN or ViT) are loaded and their classification heads are adapted for 15 classes (`models.py`).
3. **Training**: The model is trained using an Asymmetric Loss function to handle class imbalance, with performance tracked via AUPRC/AUROC metrics (`train.py`).
4. **Calibration**: Probabilistic thresholds are calibrated on a validation set to optimize metrics (`evaluate.py`).
5. **Evaluation**: Calibrated models are evaluated on the held-out test set to produce detailed final metrics (`evaluate.py`).
6. **Explainability (XAI)**: Visualizations (Grad-CAM or Attention Rollout) are generated for the predictions to ensure interpretability (`explainability.py`).

The entire workflow is orchestrated natively via the `main.py` entry point.

## Python Modules

- `config.py`: Centralizes all configurations, hyperparameters, and dataset directory paths.
- `dataset.py`: Handles metadata parsing, image augmentations, multi-hot label encoding, and group-aware data splitting.
- `models.py`: Defines the supported architectures — two CNNs (`DenseNet121Classifier`, `EfficientNetB4Classifier`) and two transformers (`ViTClassifier`, `SwinV2Classifier`).
- `train.py`: Contains the training loop, optimizer, mixed precision setup, and asymmetric loss implementation.
- `evaluate.py`: Responsible for inference, computing metrics (AUROC, AUPRC, F1, Brier, ECE), and determining optimal class-specific decision thresholds.
- `explainability.py`: Implements XAI techniques (Grad-CAM for CNNs, Attention Rollout for ViTs) to visualize model predictions spatially.
- `utils.py`: Provides helper functions for reproducible seeding, plotting training curves, and building model/experiment comparison tables.
- `main.py`: The primary command-line interface and orchestrator for running the training, evaluation, and XAI pipelines.

## Supported Models

| `--model`         | Architecture   | Type        | XAI method        |
| ----------------- | -------------- | ----------- | ----------------- |
| `densenet121`     | DenseNet-121   | CNN         | Grad-CAM          |
| `efficientnet_b4` | EfficientNet-B4| CNN         | Grad-CAM          |
| `vit_b_16`        | ViT-B/16       | Transformer | Attention Rollout |
| `swin_v2_b`       | SwinV2-B       | Transformer | not implemented   |

`--model all` runs every architecture in the table.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks showcasing different phases of the project. **These are primarily included to help understand the dataflow step-by-step and are not strictly necessary to run the core pipeline**, with the exception of `1-EDA.ipynb` which is recommended for initial exploratory data analysis.

- `1-EDA.ipynb`
- `2-Data-Preprocessing.ipynb`
- `3-Model-Training.ipynb`
- `4-Model-Evalutaion.ipynb`
- `5-XAI.ipynb`

## Usage

`--model` is required. By default a run trains on the **full dataset**; use `--subset`
for quick local experiments.

```bash
# Train DenseNet-121
python main.py --model densenet121

# Train ViT
python main.py --model vit_b_16

# Run every supported architecture
python main.py --model all

# Quick experiment on a 5000-image subset, under a named output folder
python main.py --model densenet121 --subset 5000 --experiment quick_test

# Skip training: load the best checkpoint, then calibrate / evaluate / explain
python main.py --model densenet121 --eval-only

# Print a comparison table across all experiments and exit
python main.py --compare-all
```

Common overrides: `--epochs`, `--batch-size`, `--lr`, `--num-workers`,
`--tuning-mode {full,head_only,partial}`, `--loss {asymmetric,weighted_bce,bce}`,
`--checkpoint-metric {val_loss,val_auroc,val_auprc}`,
`--threshold-metric {f1,fbeta,youden}`. Run `python main.py --help` for the full list.

## Outputs

Each run writes to `outputs/<experiment>/`, where `<experiment>` is `--experiment` if
given, otherwise `subset_<N>` or `full_dataset`:

```
outputs/<experiment>/
├── checkpoints/   # best model weights per architecture
├── results/       # metrics JSON, tuned thresholds, training curves
└── xai/           # Grad-CAM / Attention Rollout visualizations
```
