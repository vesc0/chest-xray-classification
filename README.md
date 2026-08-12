# NIH Chest X-Ray Classification

This project implements a multi-label classification pipeline for the NIH Chest X-ray dataset.

## Setup

```bash
pip install -r requirements.txt
```

To reproduce published numbers rather than just run the code, install the
pinned environment instead — same package versions, transitive dependencies
included:

```bash
pip install -r requirements.lock.txt
```

`requirements.txt` declares what the code needs as version ranges, each capped
below the next major release. `requirements.lock.txt` is `pip freeze` output
recording the exact environment the results were produced under. Ranges alone
are not enough for reproducibility: floors like `numpy>=1.24` and `pandas>=2.0`
resolve to numpy 2.x and pandas 3.x today, so a fresh install lands on majors
the results were never produced under. Regenerate the lock file with
`pip freeze > requirements.lock.txt` after deliberately upgrading anything.

The pipeline expects the NIH dataset directory (containing `Data_Entry_2017.csv`,
`train_val_list.txt`, `test_list.txt`, and the `images_001/` … `images_012/` folders).
Point it at your copy with:

```bash
export XRAY_DATASET_ROOT=/path/to/archive-chest-xrays-nih
```

Without this variable the default in `config.py` is used. If the dataset cannot be
found, or too few of the images listed in the CSV resolve on disk, the run fails
immediately rather than training on a partial dataset.

## Experimental protocol

The pipeline predicts the **14 ChestX-ray14 pathologies**. "No Finding" is not a
class — it is exactly the absence of all 14, so normal studies are an all-zero
label vector. Training it would add a 54%-prevalence target to a long-tail loss
without adding information, and averaging it into macro/micro metrics would
inflate them and break comparability with the Wang et al. / CheXNet results.
Normal-vs-abnormal is still reported, derived from the 14 outputs.

Evaluation data is held fixed across every run so that experiments at different
training sizes remain comparable:

| Split | Size | Behaviour |
| ----- | ---- | --------- |
| Test  | 25,596 | Full official split. Never subset. |
| Val   | 8,652 | Carved once from the full train_val pool, before any subsetting. |
| Train | up to 77,872 | The only split `--subset` scales. |

Both splits are patient-disjoint and label-stratified via grouped iterative
stratification. Training subsets are **nested** (5k ⊂ 15k ⊂ 30k), so a scaling
curve reflects data volume rather than which patients each draw happened to
catch. Each run writes `results/dataset_summary.json` recording the class list
and per-split support, and `--compare-all` warns if two runs scored different
class sets.

## Dataflow

1. **Data Indexing & Loading**: Images and metadata are parsed from the NIH dataset (`dataset.py`).
2. **Model Initialization**: Pre-trained backbones (CNN or ViT) are loaded and their classification heads are adapted for the 14 pathology classes (`models.py`).
3. **Training**: The model is trained using an Asymmetric Loss function to handle class imbalance, with performance tracked via AUPRC/AUROC metrics (`train.py`).
4. **Calibration**: Probabilistic thresholds are calibrated on a validation set to optimize metrics (`evaluate.py`).
5. **Evaluation**: Calibrated models are evaluated on the held-out test set to produce detailed final metrics (`evaluate.py`).
6. **Explainability (XAI)**: Heatmaps are generated for the predictions — Grad-CAM for every architecture, plus Attention Rollout for ViT (`explainability.py`). Off by default; enable with `--xai-samples N`.
7. **Weakly-supervised localization**: The heatmaps are scored against the 984 hand-drawn boxes shipped with ChestX-ray14, asking whether a classifier trained only on image-level labels looks in the right place (`localization.py`).

The entire workflow is orchestrated natively via the `main.py` entry point.

## Python Modules

- `config.py`: Centralizes all configurations, hyperparameters, and dataset directory paths.
- `device.py`: Accelerator selection (CUDA > MPS > CPU) and synchronization around timed regions.
- `metrics.py`: Threshold-free ranking metrics (per-class and macro AUROC/AUPRC) shared by training and evaluation, so the per-epoch validation curve and the reported test numbers are computed identically.
- `dataset.py`: Handles metadata parsing, image augmentations, multi-hot label encoding, and group-aware data splitting.
- `models.py`: Defines the supported architectures — two CNNs (`DenseNet121Classifier`, `EfficientNetB4Classifier`) and two transformers (`ViTClassifier`, `SwinV2Classifier`).
- `train.py`: Contains the training loop, optimizer, mixed precision setup, and asymmetric loss implementation.
- `evaluate.py`: Responsible for inference, computing metrics (AUROC, AUPRC, F1, Brier, ECE), and determining optimal class-specific decision thresholds.
- `explainability.py`: Implements the XAI methods — Grad-CAM for all four architectures (each with its own reshape transform), plus Attention Rollout for ViT — and renders heatmap overlays.
- `localization.py`: Scores those heatmaps against the ground-truth boxes (pointing game, energy fraction, IoU/IoBB) and saves the diagnostic figures.
- `utils.py`: Provides helper functions for reproducible seeding, run logging, plotting training curves, and building model/experiment comparison tables.
- `main.py`: The primary command-line interface and orchestrator for running the training, evaluation, and XAI pipelines.

## Supported Models

| `--model`         | Architecture   | Type        | Grad-CAM target layer     | Extra XAI         |
| ----------------- | -------------- | ----------- | ------------------------- | ----------------- |
| `densenet121`     | DenseNet-121   | CNN         | `features.denseblock4`    | —                 |
| `efficientnet_b4` | EfficientNet-B4| CNN         | `features[-1]`            | —                 |
| `vit_b_16`        | ViT-B/16       | Transformer | `encoder.layers[-1].ln_1` | Attention Rollout |
| `swin_v2_b`       | SwinV2-B       | Transformer | `norm`                    | —                 |

`--model all` runs every architecture in the table.

**Grad-CAM runs for all four models** so heatmaps are comparable across
architectures — comparing localization is meaningless if the CNNs and the
transformers are explained by different methods. Transformers do not emit NCHW
feature maps, so each architecture registers a reshape transform in
`explainability.GRADCAM_TARGETS` (Swin is NHWC; ViT is a token sequence that is
folded back into a 14×14 grid). ViT additionally gets Attention Rollout as a
native second view; note it is *class-agnostic*, showing where the model
attends rather than what supported a specific pathology.

## Tests

```bash
pytest
```

132 tests, no dataset required — everything is synthetic, so the suite
runs with the drive unmounted, no GPU, and no checkpoint on disk. It
covers the pure functions behind the reported numbers:

| Area | What is pinned |
| ---- | -------------- |
| `metrics.py` | NaN-vs-zero handling for unscorable classes; that training's per-epoch AUROC and evaluation's macro AUROC agree on identical input |
| `dataset.py` | train/val patient-disjointness; nested subsets (5k ⊂ 15k ⊂ 30k); that a stratified prefix tracks the pool's label distribution better than the best of 20 random draws |
| `localization.py` | box scaling, mask union and clipping, largest-connected-component detection, and the IoBB denominator (intersection over the *predicted* box, per Wang et al.) |
| `evaluate.py` | threshold tuning and its low-support fallback; ECE at the calibrated and confidently-wrong extremes; normal-vs-abnormal derivation |
| `train.py` | asymmetric loss under fp16 and saturating logits; head/backbone split for all four architectures; that freezing also stops BatchNorm statistics drifting |
| `explainability.py` | ViT token→grid and Swin NHWC reshapes; that every architecture's Grad-CAM target layer still resolves; Grad-CAM under a frozen backbone; hook cleanup |

The suite was checked by mutation: deliberately switching the IoBB denominator
to the union, dropping the loss's fp32 cast, moving the ViT Grad-CAM target back
to `encoder.ln`, letting frozen BatchNorm keep updating, removing the
low-support threshold fallback, breaking patient grouping, and replacing
stratified subsetting with a random sample are each caught by a failing test.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks showcasing different phases of the project. **These are primarily included to help understand the dataflow step-by-step and are not strictly necessary to run the core pipeline**, with the exception of `1-EDA.ipynb` which is recommended for initial exploratory data analysis.

- `1-EDA.ipynb`
- `2-Data-Preprocessing.ipynb`
- `3-Model-Training.ipynb`
- `4-Model-Evalutaion.ipynb`
- `5-XAI.ipynb`

## Usage

`--model` is required. By default a run trains on the **full training pool**;
`--subset N` scales the training set only (validation and test stay fixed).

```bash
# Train DenseNet-121
python main.py --model densenet121

# Train ViT
python main.py --model vit_b_16

# Run every supported architecture
python main.py --model all

# Scaling curve: nested training subsets, identical val/test throughout
python main.py --model densenet121 --subset 5000  --experiment scale_05k
python main.py --model densenet121 --subset 15000 --experiment scale_15k
python main.py --model densenet121 --subset 30000 --experiment scale_30k

# Skip training: load the best checkpoint, then calibrate / evaluate / explain
python main.py --model densenet121 --eval-only

# Print a comparison table across all experiments and exit
python main.py --compare-all
```

Common overrides: `--epochs`, `--batch-size`, `--lr`, `--num-workers`,
`--tuning-mode {full,head_only,partial}`, `--loss {asymmetric,weighted_bce,bce}`,
`--checkpoint-metric {val_loss,val_auroc,val_auprc}`,
`--threshold-metric {f1,fbeta,youden}`. Run `python main.py --help` for the full list.

## Timing

Every run records wall-clock cost into the `timing` block of its results JSON,
and `--compare-all` surfaces it alongside the accuracy metrics:

- **Training** — total, epochs actually run, mean seconds/epoch, and time to the
  best epoch. Compare on **mean seconds/epoch**: total is confounded by when
  early stopping happened to fire.
- **Inference** — measured on the test set, reported two ways. *End-to-end*
  includes PNG decode and transforms, so it reflects deployment cost but is
  dominated by the DataLoader for small models. *Model-only* isolates the
  forward pass and is the fair architecture comparison. The first batch is
  excluded from model-only time as warm-up.

Timings are recorded together with the conditions that produced them (device,
batch size, workers, AMP, parameter count) — they mean nothing without those.
The `run.versions` block additionally stamps Python and the packages that can
move a number — `torchvision` decides which pretrained weights `.DEFAULT`
resolves to, `scikit-learn` implements every metric, `Pillow` decodes the
images — so two results files written months apart can be checked for
comparability rather than assumed to be comparable.
`device.synchronize()` is called around every timed region, because CUDA and MPS
queue kernels asynchronously and a timer stopped without it measures dispatch
rather than execution (understating GPU time by ~3× on MPS here).

For timing to be comparable across models, run them on the same machine under
the same load. On a laptop especially, a model trained third in an overnight
sweep can look slower purely from thermal throttling.

## Outputs

Each run writes to `outputs/<experiment>/`, where `<experiment>` is `--experiment` if
given, otherwise `subset_<N>` or `full_dataset`:

```
outputs/<experiment>/
├── checkpoints/   # best model weights per architecture
├── results/       # metrics JSON, tuned thresholds, training curves, split summary,
│                  # and <model>_localization_<method>.json
├── logs/          # full console output per run, timestamped
└── xai/<model>/
    ├── localization/<method>/   # figures with ground-truth boxes drawn
    └── <method>/                # only with --xai-samples N (see below)
```

Localization figures are selected diagnostically — true positives that localized
well, true positives that looked in the wrong place, false negatives, false
positives, and the best/worst cases by heatmap energy inside the box.

`--xai-samples N` additionally renders heatmaps for N unselected test images
under `xai/<model>/<method>/`. It is off by default: those images are simply the
first N of the unshuffled test set, so the localization figures above are a
better choice. Its one real use is inspecting the six classes with no ground-truth
boxes — Consolidation, Edema, Emphysema, Fibrosis, Hernia, Pleural_Thickening.
