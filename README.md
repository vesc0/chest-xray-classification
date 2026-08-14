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

The first run of each architecture downloads its ImageNet weights: the three
torchvision backbones from `download.pytorch.org`, and ViT-S/ConvNeXtV2 from
the HuggingFace Hub via `timm`. Both are cached afterwards. The test suite
never downloads anything.

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

### Preprocessing

Three choices in `dataset.py` are deliberate and easy to mistake for oversights:

- **Direct square resize to 224, no centre crop.** The NIH images are already
  square, so nothing is distorted, and a resize-shorter-side-plus-crop would
  remove the costophrenic angles and lung apices — exactly where effusions and
  pneumothoraces present.
- **No horizontal flip.** Chest radiographs have fixed laterality, so mirroring
  one would teach the model that dextrocardia is unremarkable and blunt a
  left-sided finding like cardiomegaly. The remaining augmentation (±7°, ±2%
  translation, 0.95–1.05 scale, occasional light blur) approximates real
  positioning variation rather than the aggressive crops used on natural images.
- **Bicubic resampling**, matching the recipe four of the five backbones were
  pretrained under. torchvision's `Resize` defaults to bilinear, so leaving it
  implicit would fine-tune under a different filter than pretraining.

Train and evaluation share one geometry: thresholds are calibrated on the
validation pipeline and applied to the test pipeline, so a difference between
them would silently decalibrate every reported number.

## Dataflow

1. **Data Indexing & Loading**: Images and metadata are parsed from the NIH dataset (`dataset.py`).
2. **Model Initialization**: Pre-trained backbones (CNN, ViT, or hybrid) are loaded and their classification heads are adapted for the 14 pathology classes (`models.py`).
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
- `models.py`: Defines the five supported architectures — one per family, with the pinned pretrained-weight tags and the reasoning behind each choice.
- `train.py`: Contains the training loop, optimizer, mixed precision setup, and asymmetric loss implementation.
- `evaluate.py`: Responsible for inference, computing metrics (AUROC, AUPRC, F1, Brier, ECE) with bootstrap intervals, determining class-specific decision thresholds on validation, and saving the raw probabilities behind every reported number.
- `threshold_analysis.py`: Post-hoc comparison of thresholding schemes over those saved probabilities — fits F1 / Youden / fixed-sensitivity operating points, puts a confidence interval on each threshold, and scores them on test. Runs no model.
- `explainability.py`: Implements the XAI methods — Grad-CAM for all five architectures (each with its own reshape transform), plus Attention Rollout for ViT — and renders heatmap overlays.
- `localization.py`: Scores those heatmaps against the ground-truth boxes (pointing game, energy fraction, IoU/IoBB) and saves the diagnostic figures.
- `utils.py`: Provides helper functions for reproducible seeding, run logging, plotting training curves, and building model/experiment comparison tables.
- `main.py`: The primary command-line interface and orchestrator for running the training, evaluation, and XAI pipelines.

## Supported Models

One backbone per architecture family, so a difference between two results is
attributable to the architecture rather than to capacity, input resolution, or
pretraining data:

| `--model`      | Architecture | Family        | Params | Source | Weights |
| -------------- | ------------ | ------------- | ------ | ------ | ------- |
| `densenet121`  | DenseNet-121 | CNN baseline  | 7.0M   | torchvision | `IMAGENET1K_V1` |
| `vit_s_16`     | ViT-S/16     | pure ViT      | 21.7M  | timm   | `deit_small_patch16_224.fb_in1k` |
| `convnextv2_t` | ConvNeXtV2-T | modern CNN    | 27.9M  | timm   | `convnextv2_tiny.fcmae_ft_in1k` |
| `swin_v2_t`    | SwinV2-T     | modern ViT    | 27.6M  | torchvision | `IMAGENET1K_V1` |
| `maxvit_t`     | MaxViT-T     | hybrid CNN/ViT| 30.4M  | torchvision | `IMAGENET1K_V1` |

`--model all` runs every architecture in the table. Parameter counts are as
built here, with the 14-class head — a few percent below the 1000-class
ImageNet figures usually quoted.

**What is held constant.** The last four models sit in a 22–31M band, so
"modern CNN vs modern ViT" is a comparison at matched capacity. Every model is
pretrained on **ImageNet-1k only** — no IN21k/IN22k checkpoint is used even
where one exists, because DenseNet, SwinV2 and MaxViT have no IN21k option
here, and using 21k for the two that do would confound architecture with
pretraining data while specifically flattering the pure ViT. Every model takes
**224×224 input with ImageNet normalization**, so a single shared transform
serves the whole roster.

**What is deliberately not constant.** DenseNet-121 is not scale-matched: its
value is being the exact backbone behind the CheXNet results, which a deeper
variant chosen to close the parameter gap would throw away. Its ImageNet
weights also come from torchvision's original recipe (74.4% top-1) rather than
the modern recipes behind SwinV2-T (82.1%) and MaxViT-T (83.7%), so the
comparison is between architectures *as they are normally obtained*, not
between architectures pretrained identically.

Two constraints are worth knowing before changing `IMAGE_SIZE`:

- **MaxViT-T fixes the resolution at 224.** torchvision builds its attention
  partition sizes from a declared input size and reshapes against them, so any
  other resolution raises inside the partitioning rather than adapting.
- **SwinV2 is therefore used slightly off-resolution.** torchvision's SwinV2
  weights — every size — were trained at 256. At 224 the final stage is a 7×7
  map against a window size of 8. SwinV2's log-spaced continuous position bias
  is designed for exactly this transfer (it is the headline change from
  SwinV1), so the effect is a soft degradation, but it is a real limitation of
  this comparison.

### Explainability per architecture

| `--model`      | Grad-CAM target layer  | Layout       | CAM grid | Extra XAI         |
| -------------- | ---------------------- | ------------ | -------- | ----------------- |
| `densenet121`  | `features.denseblock4` | NCHW         | 7×7      | —                 |
| `vit_s_16`     | `blocks[-1].norm1`     | tokens→grid  | 14×14    | Attention Rollout |
| `convnextv2_t` | `stages[-1]`           | NCHW         | 7×7      | —                 |
| `swin_v2_t`    | `norm`                 | NHWC→NCHW    | 7×7      | —                 |
| `maxvit_t`     | `blocks[-1]`           | NCHW         | 7×7      | —                 |

**Grad-CAM runs for all five models** so heatmaps are comparable across
architectures — comparing localization is meaningless if the CNNs and the
transformers are explained by different methods. Transformers do not emit NCHW
feature maps, so each architecture registers a reshape transform in
`explainability.GRADCAM_TARGETS` (Swin is NHWC; ViT is a token sequence folded
back into a grid; MaxViT ends in grid attention but still emits NCHW, so the
hybrid needs no special handling).

Note the **CAM grid** column. ViT-S/16 produces a 14×14 map where the
hierarchical models produce 7×7, because a patch-16 transformer keeps one token
per 16×16 patch while the others have downsampled four times. Upsampled to 224
this gives ViT finer predicted boxes for free, which inflates its IoU and IoBB
relative to the rest. The **pointing game** is far less sensitive to grid size
and is the fairer cross-architecture comparison.

**Attention Rollout is ViT-only by necessity, not oversight.** SwinV2 attends
inside shifted local windows and merges patches between stages, so there is no
single token set to multiply across layers; MaxViT interleaves MBConv blocks
that carry no attention at all, breaking the chain outright. Adapting rollout
to either is a research problem — which is exactly why Grad-CAM, defined for
all five, is the method the comparison rests on. Rollout is also
*class-agnostic*, showing where the model attends rather than what supported a
specific pathology.

**Rollout's localization scores will look broken, and are not.** ViTs
repurpose a few low-information background patches as global scratch space,
and those tokens carry very large activations (Darcet et al., *Vision
Transformers Need Registers*, 2024, which shows the effect on DeiT). Rollout
multiplies attention across all 12 layers, compounding exactly those tokens,
so the heatmap peak routinely lands on an image corner rather than on anatomy
— driving the pointing game to ≈0 against a ≈0.07 random baseline. This is the
method meeting the architecture, not a defect in the implementation. Report it
as a property of Attention Rollout; do not compare it against Grad-CAM's
localization as though the two were interchangeable measurements.

## Tests

```bash
pytest
```

176 tests, no dataset required — everything is synthetic, so the suite
runs with the drive unmounted, no GPU, and no checkpoint on disk. Backbones are
built with `pretrained=False`, so nothing downloads weights either. It covers
the pure functions behind the reported numbers:

| Area | What is pinned |
| ---- | -------------- |
| `metrics.py` | NaN-vs-zero handling for unscorable classes; that training's per-epoch AUROC and evaluation's macro AUROC agree on identical input |
| `dataset.py` | train/val patient-disjointness; nested subsets (5k ⊂ 15k ⊂ 30k); that a stratified prefix tracks the pool's label distribution better than the best of 20 random draws |
| preprocessing | bicubic resampling in both pipelines; that train and eval share one geometry; that no centre crop and no horizontal flip creep in; grayscale→3-channel replication |
| `localization.py` | box scaling, mask union and clipping, largest-connected-component detection, and the IoBB denominator (intersection over the *predicted* box, per Wang et al.) |
| `evaluate.py` | confusion-sweep counts against a brute-force scan; that a class scored entirely below the old grid floor still tunes and still predicts positives; the low-support and degenerate fallbacks; that fixed-sensitivity mode reaches its target and pays for it in specificity; ECE under both binning strategies and that no prediction is dropped from either; that samples-F1 credits a correctly-predicted normal study; that bootstrap intervals contain their own point estimate, widen for rare classes, widen again under patient grouping, and are withheld entirely for unscorable classes; prediction round-trips |
| `train.py` | asymmetric loss under fp16 and saturating logits; head/backbone split for all five architectures; that freezing also stops BatchNorm statistics drifting |
| `models.py` | that the factory and `SUPPORTED_MODELS` describe the same roster; that neither timm tag pulls in 21k pretraining; ViT's prefix-token count; that MaxViT is what fixes `IMAGE_SIZE` at 224 |
| `explainability.py` | ViT token→grid and Swin NHWC reshapes; that every architecture's Grad-CAM target layer still resolves; the 14×14-vs-7×7 CAM grid asymmetry; that Attention Rollout actually captures attention and restores the model afterwards; Grad-CAM under a frozen backbone; hook cleanup |

The suite was checked by mutation: deliberately switching the IoBB denominator
to the union, dropping the loss's fp32 cast, moving the ViT Grad-CAM target to
the final norm, leaving timm's fused attention enabled so rollout silently
captures nothing, leaving a stale architecture in the Grad-CAM target table,
hardcoding the wrong prefix-token count, falling back to bilinear resampling,
bootstrapping images instead of patients,
letting frozen BatchNorm keep updating, removing the low-support threshold
fallback, breaking patient grouping, and replacing stratified subsetting with a
random sample are each caught by a failing test.

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
python main.py --model vit_s_16

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
`--threshold-metric {f1,fbeta,youden,sensitivity}`, `--target-sensitivity`,
`--ece-bin-strategy {quantile,uniform}`. Run `python main.py --help` for the full list.

## Confidence intervals

Every reported AUROC and AUPRC — per class and macro — carries a bootstrap
confidence interval, written into the results JSON as `auroc_ci` / `auprc_ci`
alongside the point estimate and printed next to it in the summary table:

```
Class                     Thr    AUROC    AUROC 95% CI    AUPRC     ...
Atelectasis              0.50   0.8153  [0.803, 0.827]   0.4472     ...
Pneumonia                0.50   0.7188  [0.671, 0.764]   0.0512     ...
```

This exists because a run produces **one seed per configuration**. Five
architectures reported as five point estimates cannot support a ranking claim,
and it is the first thing a reader should push back on. Resampling the test set
costs no retraining at all — roughly 70 seconds per model against hours of
training — and turns "0.812 vs 0.818" into a statement about whether the gap is
resolved. **If two models' intervals overlap, the comparison is not settled.**

Note how much wider the interval is for the rare classes: a Pneumonia AUROC
resting on ~300 positives is a far softer number than an Atelectasis AUROC
resting on ~3,200, and the point estimate alone hides that entirely.

**Whole patients are resampled, not individual images.** ChestX-ray14 has
several studies per patient and those rows are correlated, so an image-level
bootstrap treats correlated observations as independent and returns an interval
that is too narrow — measured at ~2.3× too narrow on correlated synthetic data.
The grouping is matched to predictions positionally, so it requires an
unshuffled loader and raises rather than guessing if given a shuffled one.

Precision, recall and F1 get intervals too, written as `precision_ci` /
`recall_ci` / `f1_ci`. The thresholds behind them were frozen on validation
before the test set was touched, so resampling test rows at a fixed operating
point is an ordinary sampling interval and means the same thing the AUROC
interval does. This is where the intervals matter most — Hernia's test F1 rests
on 86 positives.

What that interval does *not* cover is uncertainty in the threshold itself. That
needs resampling **validation** and refitting, which `threshold_analysis.py`
does (see below). On the rare classes the second interval is the wider of the
two, and reading a test-set F1 interval as though it settled the operating point
would understate how loosely that point is located.

Configured in `config.py` — `BOOTSTRAP_ENABLED` (set `False` to skip it),
`BOOTSTRAP_SAMPLES`, `BOOTSTRAP_CI`, and `BOOTSTRAP_GROUP_BY_PATIENT`.

## Thresholds

Architecture comparison rests on **AUROC and AUPRC**, which are threshold-free.
Any threshold adds variance that is not about the model, and the ChestX-ray14
leaderboard lineage reports AUROC. The binary metrics are reported alongside
them, not instead of them.

When a threshold is needed, one is fitted **per class, on validation only**, and
frozen before the test set is read. Nothing else touches test.

**F1 is the default objective.** It degrades gracefully across this dataset's
prevalence range — Hernia at 0.16%, Infiltration at 15.9% — and it is what most
ChestX-ray14 papers report when they report binary metrics, so the numbers stay
comparable. Youden's J is prevalence-independent by construction, which sounds
desirable until it is applied to the tail: at 0.16% prevalence it will happily
take 50% sensitivity and 95% specificity, a PPV of about 2%. `sensitivity` is
the clinically framed alternative — it fixes recall at
`THRESHOLD_TARGET_SENSITIVITY` and reports the most specific threshold reaching
it, which puts every architecture at the same operating point.

**Candidates are the model's own scores**, not a grid. A grid cannot represent
an optimum outside its bounds, and — the failure that motivated this — a grid
floored at 0.05 scores zero at every point for a class whose scores all sit
below 0.05, then returns 0.05, labels it `tuned`, and predicts the class
negative everywhere. That reads as a result and is an artifact. With the scores
themselves as candidates the lowest candidate predicts everything positive, so
F1 is positive whenever the class has a single positive and an all-negative
"tuned" rule is unreachable by construction.

**A class needs 50 validation positives to be fitted at all**
(`THRESHOLD_MIN_SUPPORT`). Validation supports run from ~1,380 (Infiltration)
down to ~14 (Hernia); a threshold fitted on 14 positives is noise that happens
to be a float. Classes below the line stay at 0.5, are recorded with a `status`
and a `reason` in the thresholds JSON, and are marked `*` in the results table —
a fallback and a fitted-but-poor operating point look identical otherwise.

### Post-hoc analysis

Raw validation and test probabilities are saved per model
(`<model>_{val,test}_predictions.npz`, ~1 MB each), so changing the thresholding
scheme never costs another inference pass. `threshold_analysis.py` reads them:

```bash
python threshold_analysis.py --model all --experiment full_dataset \
    --metric f1 youden sensitivity
```

It fits each objective on validation, bootstraps the **threshold itself** by
refitting on validation resamples, applies the frozen thresholds to test, and
prints them side by side with intervals on both. A wide threshold CI next to a
narrow test-set CI means the operating point is poorly located rather than
precisely measured — the distinction the rare classes turn on.

Disable the saved arrays with `--no-save-predictions` if disk is a concern.

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
├── results/       # metrics JSON, tuned thresholds, raw val/test predictions (.npz),
│                  # training curves, split summary, threshold analysis,
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
