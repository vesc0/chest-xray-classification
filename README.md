# NIH Chest X-Ray Classification

This project implements a multi-label classification pipeline for the NIH Chest X-ray dataset.

## Dataflow

1. **Data Indexing & Loading**: Images and metadata are parsed from the NIH dataset (`dataset.py`).
2. **Model Initialization**: Pre-trained backbones (CNN or ViT) are loaded and their classification heads are adapted for 15 classes (`models.py`).
3. **Training**: The model is trained using an Asymmetric Loss function to handle class imbalance, with performance tracked via AUPRC/AUROC metrics (`train.py`).
4. **Calibration**: Probabilistic thresholds are calibrated on a validation set to optimize metrics (`evaluate.py`).
5. **Evaluation**: Calibrated models are evaluated on the held-out test set to produce detailed final metrics (`evaluate.py`).
7. **Explainability (XAI)**: Visualizations (Grad-CAM or Attention Rollout) are generated for the predictions to ensure interpretability (`explainability.py`).

The entire workflow is orchestrated natively via the `main.py` entry point.

## Python Modules

- `config.py`: Centralizes all configurations, hyperparameters, and dataset directory paths.
- `dataset.py`: Handles metadata parsing, image augmentations, multi-hot label encoding, and group-aware data splitting.
- `models.py`: Defines the core neural network architectures: a CNN (`DenseNet121Classifier`) and a Vision Transformer (`ViTClassifier`).
- `train.py`: Contains the training loop, optimizer, mixed precision setup, and asymmetric loss implementation.
- `evaluate.py`: Responsible for inference, computing metrics (AUROC, AUPRC, F1), and determining optimal class-specific decision thresholds.
- `explainability.py`: Implements XAI techniques (Grad-CAM for CNNs, Attention Rollout for ViTs) to visualize model predictions spatially.
- `utils.py`: Provides helper functions for setting random seeds (reproducibility) and plotting training curves.
- `main.py`: The primary command-line interface and orchestrator for running training, evaluation, ensembling, and XAI pipelines.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks showcasing different phases of the project. **These are primarily included to help understand the dataflow step-by-step and are not strictly necessary to run the core pipeline**, with the exception of `1-EDA.ipynb` which is recommended for initial exploratory data analysis.

- `1-EDA.ipynb`
- `2-Data-Preprocessing.ipynb`
- `3-Model-Training.ipynb`
- `4-Model-Evalutaion.ipynb`
- `5-XAI.ipynb`

## Usage Example

To run the full pipeline (train, calibrate, evaluate, XAI) using defualt values via the `main.py` script:

```bash
# Train DenseNet-121
python main.py --model densenet121

# Train ViT
python main.py --model vit_b_16

# Run both
python main.py --model both
```
