# NIH Chest X-Ray Classification

Multi-label classification of the 14 ChestX-ray14 pathologies across five
architecture families, with calibrated thresholds, bootstrap confidence
intervals, subgroup analysis, and explainability that is itself sanity-checked.

The reasoning behind every choice here — experimental protocol, backbone and
checkpoint selection, thresholds, intervals, ensembling, subgroup findings — is
in [`docs/design-notes.md`](docs/design-notes.md).

## Setup

```bash
pip install -r requirements.txt
```

Use `requirements.lock.txt` to reproduce published numbers; `requirements.txt`
only declares ranges.

Point the pipeline at your copy of the NIH dataset — the folder holding
`Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt` and `images_001/`
… `images_012/`:

```bash
export XRAY_DATASET_ROOT=/path/to/archive-chest-xrays-nih
```

A missing or incomplete dataset fails the run immediately. Pretrained weights
download on first use and are cached; the test suite downloads nothing.

## Usage

`--model` is required. `--subset N` scales the training set only, leaving
validation and test fixed.

```bash
# Train one architecture, or every one on the roster
python main.py --model densenet121
python main.py --model all

# Scaling curve: nested training subsets, identical val/test throughout
python main.py --model densenet121 --subset 5000 --experiment scale_05k

# Skip training: load the best checkpoint, then calibrate / evaluate / explain
python main.py --model densenet121 --eval-only

# Compare every finished experiment and exit
python main.py --compare-all
```

Common overrides: `--epochs`, `--batch-size`, `--lr`, `--num-workers`,
`--image-size`, `--weight-tag`, `--tuning-mode {full,head_only,partial}`,
`--loss {asymmetric,weighted_bce,bce}`,
`--checkpoint-metric {val_loss,val_auroc,val_auprc}`,
`--threshold-metric {f1,fbeta,youden,sensitivity}`, `--target-sensitivity`,
`--ece-bin-strategy {quantile,uniform}`, `--xai-samples N`.
`python main.py --help` lists them all.

Three post-hoc tools run over the saved probability arrays — no model, no GPU,
no dataset:

```bash
python threshold_analysis.py --model all --experiment full_dataset --metric f1 youden sensitivity
python bias_analysis.py --all
python ensemble.py --auto
```

## Models

One backbone per architecture family, so a difference between two results is
attributable to the architecture rather than to capacity, resolution, or
pretraining data. All are ImageNet-1k pretrained at 224×224.

| `--model`      | Architecture | Family         | Params | Weights |
| -------------- | ------------ | -------------- | ------ | ------- |
| `densenet121`  | DenseNet-121 | CNN baseline   | 7.0M   | torchvision `IMAGENET1K_V1` |
| `vit_s_16`     | ViT-S/16     | pure ViT       | 21.7M  | timm `deit3_small_patch16_224.fb_in1k` |
| `convnextv2_t` | ConvNeXtV2-T | modern CNN     | 27.9M  | timm `convnextv2_tiny.fcmae_ft_in1k` |
| `swin_t`       | Swin-T       | modern ViT     | 27.5M  | torchvision `IMAGENET1K_V1` |
| `maxvit_t`     | MaxViT-T     | hybrid CNN/ViT | 30.4M  | torchvision `IMAGENET1K_V1` |

Two more are buildable but off the roster: `swin_v2_t`, kept so its documented
224-transfer failure stays reproducible, and `densenet121_xrv`, DenseNet-121
initialized from [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
chest-radiograph weights to isolate pretraining domain against the `densenet121`
control. `--model all` runs everything except `densenet121_xrv`; run that by
name. `--weight-tag` swaps the checkpoint on the timm models and is guarded
against tags with the wrong normalization or with ChestX-ray14 in their training
data. See [design notes](docs/design-notes.md#supported-models).

## Pipeline

Data indexing (`dataset.py`) → backbone with a 14-class head (`models.py`) →
training under asymmetric loss (`train.py`) → per-class threshold calibration on
validation and evaluation on the held-out test set (`evaluate.py`) → Grad-CAM
and Attention Rollout heatmaps (`explainability.py`) → scoring those heatmaps
against ChestX-ray14's 984 ground-truth boxes (`localization.py`) → randomizing
the model's weights to confirm the explanations track them (`sanity_checks.py`).
`main.py` orchestrates it.

Evaluation data is fixed across every run: test is the full official
25,596-image split, and validation is 8,652 images carved once from the
train_val pool before any subsetting. Both are patient-disjoint and
label-stratified, and training subsets are nested (5k ⊂ 15k ⊂ 30k).

| Module | Role |
| ------ | ---- |
| `config.py` | Hyperparameters, paths, feature switches |
| `device.py` | Accelerator selection (CUDA > MPS > CPU) and timing synchronization |
| `dataset.py` | Metadata parsing, augmentation, multi-hot labels, group-aware splits |
| `models.py` | The architecture roster, pinned weight tags, and checkpoint guards |
| `metrics.py` | AUROC/AUPRC shared by training and evaluation, so both agree |
| `train.py` | Training loop, optimizer, mixed precision, asymmetric loss |
| `evaluate.py` | Inference, metrics with bootstrap CIs, threshold tuning, saved probabilities |
| `threshold_analysis.py` | Post-hoc comparison of thresholding schemes |
| `bias_analysis.py` | Sex- and age-stratified AUROC and miss rates |
| `ensemble.py` | Averages saved probabilities of finished runs |
| `explainability.py` | Grad-CAM (all architectures) and Attention Rollout (ViT) |
| `localization.py` | Pointing game, energy fraction, IoU/IoBB against the boxes |
| `sanity_checks.py` | Cascading model-parameter randomization (Adebayo et al., 2018) |
| `utils.py` | Seeding, run logging, curves, comparison tables |
| `main.py` | Command-line interface and orchestration |

## What gets measured

**Ranking.** Architecture comparison rests on AUROC and AUPRC, which are
threshold-free. Every one of them carries a patient-level bootstrap confidence
interval, because one seed per configuration cannot support a ranking claim.
**If two models' intervals overlap, the comparison is not settled.**

**Operating point.** Thresholds are fitted per class, on validation only, and
frozen before the test set is read. A class needs 50 validation positives to be
fitted at all; classes below that stay at 0.5 and are marked `*`.

**Subgroups.** `bias_analysis.py` re-attaches the patient metadata behind every
saved prediction and asks separately whether the model *ranks* worse for a group
and whether the group gets *missed* more at the deployed threshold. Across 22
runs sex is a bounded null, while patients aged 0-19 carry a median +0.072 FNR
gap that no amount of macro AUROC closes.

**Explanations.** Grad-CAM runs for every architecture, since comparing
localization across models explained by different methods is meaningless.
Heatmaps are scored against the ground-truth boxes, and `sanity_checks.py`
re-runs each explainer with the weights progressively randomized to confirm the
maps depend on the model at all.

**Ensembling.** Members are picked by a stated rule rather than a search, and
the run bootstraps its margin over the best single member on shared resamples.
`--rule stack` fits a per-class combiner and is the only rule that improves the
operating point rather than just the ranking — it drops macro ECE from 0.284 to
0.028.

Full numbers, protocols and caveats for all five are in the
[design notes](docs/design-notes.md).

## Outputs

Each run writes to `outputs/<experiment>/`, where `<experiment>` is
`--experiment` if given, otherwise `subset_<N>` or `full_dataset`:

```
outputs/<experiment>/
├── checkpoints/   # best model weights per architecture
├── results/       # metrics JSON, tuned thresholds, raw val/test predictions (.npz),
│                  # training curves, split summary, threshold and subgroup
│                  # analyses, localization and sanity-check reports
├── logs/          # full console output per run, timestamped
└── xai/<model>/
    ├── localization/<method>/   # figures with ground-truth boxes drawn
    └── <method>/                # only with --xai-samples N
```

Localization figures are selected diagnostically — well- and poorly-localized
true positives, false negatives, false positives, and the extremes by heatmap
energy inside the box. `ensemble.py` writes the same layout minus `checkpoints/`
and `xai/`.

## Tests

```bash
pytest
```

373 tests, no dataset, GPU, checkpoint or network required — everything is
synthetic and backbones are built with `pretrained=False`. They pin the pure
functions behind the reported numbers, and the suite was checked by mutation:
switching the IoBB denominator to the union, dropping the loss's fp32 cast,
bootstrapping images instead of patients, inserting a centre crop, and a dozen
other deliberate breakages each fail a test. Coverage table in the
[design notes](docs/design-notes.md#tests).

## Notebooks

`notebooks/` walks through the dataflow step by step (`1-EDA`,
`2-Data-Preprocessing`, `3-Model-Training`, `4-Model-Evalutaion`, `5-XAI`).
They are explanatory, not required by the pipeline — except `1-EDA.ipynb`,
which is worth reading first.
