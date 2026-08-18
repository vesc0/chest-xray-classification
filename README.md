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
the HuggingFace Hub via `timm`. `densenet121_xrv` additionally pulls its
chest-X-ray checkpoint (~30 MB) from the TorchXRayVision GitHub release into
`~/.torchxrayvision/models_data/`. All are cached afterwards. The test suite
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
8. **Sanity checks**: The same explanations are recomputed with the model's weights progressively randomized, to establish that step 7 measured the model at all (`sanity_checks.py`).

The entire workflow is orchestrated natively via the `main.py` entry point.

## Python Modules

- `config.py`: Centralizes all configurations, hyperparameters, and dataset directory paths.
- `device.py`: Accelerator selection (CUDA > MPS > CPU) and synchronization around timed regions.
- `metrics.py`: Threshold-free ranking metrics (per-class and macro AUROC/AUPRC) shared by training and evaluation, so the per-epoch validation curve and the reported test numbers are computed identically.
- `dataset.py`: Handles metadata parsing, image augmentations, multi-hot label encoding, and group-aware data splitting.
- `models.py`: Defines the five roster architectures — one per family, with the pinned pretrained-weight tags and the reasoning behind each choice — plus the two off-roster models and the guards on overriding a checkpoint.
- `train.py`: Contains the training loop, optimizer, mixed precision setup, and asymmetric loss implementation.
- `evaluate.py`: Responsible for inference, computing metrics (AUROC, AUPRC, F1, Brier, ECE) with bootstrap intervals, determining class-specific decision thresholds on validation, and saving the raw probabilities behind every reported number.
- `threshold_analysis.py`: Post-hoc comparison of thresholding schemes over those saved probabilities — fits F1 / Youden / fixed-sensitivity operating points, puts a confidence interval on each threshold, and scores them on test. Runs no model.
- `explainability.py`: Implements the XAI methods — Grad-CAM for every architecture (each with its own reshape transform), plus Attention Rollout for ViT — and renders heatmap overlays.
- `localization.py`: Scores those heatmaps against the ground-truth boxes (pointing game, energy fraction, IoU/IoBB) and saves the diagnostic figures.
- `sanity_checks.py`: Runs the cascading model-parameter randomization test (Adebayo et al., 2018) over every explainer, checking that the explanations degrade as the weights they claim to explain are destroyed.
- `utils.py`: Provides helper functions for reproducible seeding, run logging, plotting training curves, and building model/experiment comparison tables.
- `main.py`: The primary command-line interface and orchestrator for running the training, evaluation, and XAI pipelines.

## Supported Models

One backbone per architecture family, so a difference between two results is
attributable to the architecture rather than to capacity, input resolution, or
pretraining data:

| `--model`      | Architecture | Family        | Params | Source | Weights |
| -------------- | ------------ | ------------- | ------ | ------ | ------- |
| `densenet121`  | DenseNet-121 | CNN baseline  | 7.0M   | torchvision | `IMAGENET1K_V1` |
| `vit_s_16`     | ViT-S/16     | pure ViT      | 21.7M  | timm   | `deit3_small_patch16_224.fb_in1k` |
| `convnextv2_t` | ConvNeXtV2-T | modern CNN    | 27.9M  | timm   | `convnextv2_tiny.fcmae_ft_in1k` |
| `swin_t`       | Swin-T       | modern ViT    | 27.5M  | torchvision | `IMAGENET1K_V1` |
| `maxvit_t`     | MaxViT-T     | hybrid CNN/ViT| 30.4M  | torchvision | `IMAGENET1K_V1` |

Two further models are buildable but **not** part of the roster:
`swin_v2_t` (SwinV2-T, 27.6M) and `densenet121_xrv` (7.0M) — both below.
`--model all` runs `SWEEP_MODELS`, which is everything in `SUPPORTED_MODELS`
except `densenet121_xrv`; run that one by name. Parameter counts are as built
here, with the 14-class head.

**What is held constant.** The last four models sit in a 22–31M band, so
"modern CNN vs modern ViT" is a comparison at matched capacity. Every roster
model is pretrained on **ImageNet-1k only** by default, and takes **224×224
input with ImageNet normalization**, so one shared transform serves the whole
roster. `--weight-tag` and `densenet121_xrv` are the two sanctioned ways past
the first of those, and both keep the second.

**What is deliberately not constant.** DenseNet-121 is not scale-matched: its
value is being the exact backbone behind the CheXNet results. Its ImageNet
weights also come from torchvision's original recipe (74.4% top-1) rather than
the modern recipes behind Swin-T (81.5%) and MaxViT-T (83.7%), so the
comparison is between architectures *as they are normally obtained*.

**Why Swin-T and not SwinV2-T.** SwinV2-T held this slot first. Its torchvision
weights are 256-native, and at 224 the final stage is a 7×7 map against a
window size of 8; SwinV2's log-spaced continuous position bias is meant to
absorb that transfer and measurably does not. Full fine-tuning reached train
AUROC 0.656 after 20 epochs against 0.79–0.81 for every other backbone, with
localization at 1.19× the random baseline. Swin-T is pretrained at 224 and
replaces it. SwinV2-T stays buildable so the failure can be reproduced, and so
it can be re-run at `--image-size 256` as a resolution study.

### Medical pretraining (`densenet121_xrv`)

```bash
python main.py --model densenet121_xrv --experiment densenet_xrv_chex
```

The roster answers "which architecture". This answers a question it cannot:
**how much of that spread is architecture at all, and how much is pretraining
domain?** It is the same DenseNet-121 graph as the baseline — same 7.0M
parameters, same Grad-CAM target, same head, same loss and schedule —
initialized from [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
chest-radiograph weights instead of ImageNet. **The `densenet121` run is its
control**, and the two differ in exactly one thing.

Default corpus is CheXpert (`densenet121-res224-chex`, 224k images, Stanford).
Swap it with `--weight-tag`:

| Tag | Corpus | |
| --- | ------ | - |
| `densenet121-res224-chex` | CheXpert (Stanford), 224k | default |
| `densenet121-res224-pc` | PadChest (Spain), 160k | |
| `densenet121-res224-mimic_ch`, `..._nb` | MIMIC-CXR (BIDMC), 377k | |
| `densenet121-res224-all` | nih-pc-chex-mimic_ch-google-openi-rsna | **blocked** |
| `densenet121-res224-nih` | ChestX-ray14 | **blocked** |
| `densenet121-res224-rsna` | RSNA Pneumonia Challenge | **blocked** |

**Why three are blocked.** Each was trained on ChestX-ray14 or on a dataset
derived from it (the RSNA challenge images are drawn from ChestX-ray8), so it
has already seen the 25,596 images this pipeline reports as held out. The
result would not be "medical pretraining wins" — it would be a model
recognizing its own training set, arriving as a plausible number several points
above every other row, with nothing downstream able to detect it. `models.py`
rejects them at construction. Note this rules out XRV's strongest checkpoint,
`-all`, on purpose.

**Preprocessing.** XRV checkpoints expect 1-channel input scaled to
[−1024, 1024], not 3-channel ImageNet-normalized tensors. Rather than fork the
shared transform, `DenseNet121XRVClassifier.forward` de-normalizes and rescales
inside the model. Both are affine maps over the same 8-bit pixel, so the
composition is exact — it reproduces `xrv.utils.normalize` to float32 rounding,
which `tests/test_models.py` pins against xrv's own function. Verified
end-to-end too: the untuned CheXpert checkpoint scores 0.53–0.84 AUROC
zero-shot on NIH test images through this path, where a broken normalization
would sit at chance across every column.

XRV's own `forward` silently resizes non-224 input back to 224, which would
override `--image-size`; the attribute driving that is removed at construction,
so this model follows the same resolution rule as the rest.

**One import guard.** torchxrayvision's vendored baseline models each run
`sys.path.insert(0, <own folder>)` when imported, and one of those folders
contains a package named `config` — ahead of the project root, so `import
config` resolves to theirs in any interpreter that has not already imported
ours. This process is unaffected; spawned DataLoader workers are not, and they
die on `module 'config' has no attribute 'SEED'` while the parent reports a
`BrokenPipeError` several frames from the cause. `models._import_torchxrayvision`
restores `sys.path` around the import, and a test asserts nothing from the
package is left on it.

### Overriding the checkpoint (`--weight-tag`)

```bash
python main.py --model vit_s_16 --weight-tag deit3_small_patch16_224.fb_in22k_ft_in1k --experiment vit_in22k
```

For asking what pretraining *data* is worth on a fixed architecture.
`deit3_small_patch16_224.fb_in1k` and `.fb_in22k_ft_in1k` share architecture,
authors, recipe and normalization, so **the default ViT-S run is already the
control**. (The 22k checkpoint carries an extra IN1k fine-tuning stage the IN1k
one does not — inherent to any 21k-vs-1k comparison, worth stating with the
result.)

Two guards, because the failure mode is a run that trains fine and reports a
wrong number: the flag is rejected for the torchvision models, which have no
tag to override; and a tag whose checkpoint expects normalization other than
ImageNet mean/std is rejected outright (`vit_small_patch16_224.augreg_*` are
JAX ports expecting mean/std = 0.5).

The same flag selects the pretraining corpus for `densenet121_xrv`, where it
takes a TorchXRayVision weight name and is guarded against test-set leakage
instead — see above.

### Resolution (`--image-size`)

Default 224. The images must already be sourced at the target resolution — see
the resize recipe below. Support differs per model:

| Model | Non-224 |
| ----- | ------- |
| `densenet121`, `convnextv2_t`, `swin_t`, `densenet121_xrv` | yes |
| `vit_s_16` | yes — timm interpolates the position embeddings |
| `swin_v2_t` | yes, and 256 is its native resolution |
| `maxvit_t` | **no** — torchvision fixes its attention partitions at 224, so it raises |

### Explainability per architecture

| `--model`      | Grad-CAM target layer  | Layout       | CAM grid | Extra XAI         |
| -------------- | ---------------------- | ------------ | -------- | ----------------- |
| `densenet121`  | `features.denseblock4` | NCHW         | 7×7      | —                 |
| `densenet121_xrv` | `features.denseblock4` | NCHW      | 7×7      | —                 |
| `vit_s_16`     | `blocks[-1].norm1`     | tokens→grid  | 14×14    | Attention Rollout |
| `convnextv2_t` | `stages[-1]`           | NCHW         | 7×7      | —                 |
| `swin_t`       | `norm`                 | NHWC→NCHW    | 7×7      | —                 |
| `swin_v2_t`    | `norm`                 | NHWC→NCHW    | 7×7      | —                 |
| `maxvit_t`     | `blocks[-1]`           | NCHW         | 7×7      | —                 |

**Grad-CAM runs for every model** so heatmaps are comparable across
architectures — comparing localization is meaningless if the CNNs and the
transformers are explained by different methods. Transformers do not emit NCHW
feature maps, so each architecture registers a reshape transform in
`explainability.GRADCAM_TARGETS` (Swin is NHWC; ViT is a token sequence folded
back into a grid; MaxViT ends in grid attention but still emits NCHW, so the
hybrid needs no special handling).

**The rule for picking those layers: explain each model at the tensor its
classifier pools.** Choosing by eye invites a different notion of "evidence" per
architecture, which is the thing the comparison is trying to hold fixed.
DenseNet-121 is the one deliberate exception. torchvision runs
`F.relu(features, inplace=True)` between `features` and the pool, so hooking
`features` — the obvious choice, and the one pytorch-grad-cam's DenseNet example
uses — captures a tensor the in-place ReLU then overwrites, silently pairing
post-ReLU activations with pre-ReLU gradients. `denseblock4` is the last tensor
before that trap; stopping one block short costs little (CAM correlation
0.80–0.99 on 7 of 8 real radiographs, peak unchanged on 5 of 8) and the
alternative's failure mode is silent.

**A second asymmetry, alongside the grid size below: what the pooled tensor has
been normalized by.** SwinV2 and ViT-S both pool straight out of a LayerNorm, so
their CAMs are computed on features that have been channel-normalized *per
spatial position*, while DenseNet, ConvNeXtV2 and MaxViT keep raw activation
magnitudes. This is not a detail with a small effect: moving SwinV2's target
from `norm` to the pre-norm stage output drops the CAM correlation to 0.25–0.69
and moves the peak on 8 of 8 images. Both targets are defensible in isolation —
the rule above is what picks one — but the honest reading is that the two
transformers measure a slightly different quantity from the three others.

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

### Do the explanations explain the model?

A heatmap that lands on the lungs is not evidence that the method is reading the
classifier. Adebayo et al., *Sanity Checks for Saliency Maps* (2018), showed that
several widely used methods produce essentially the same map after the network's
weights have been randomized — they were tracing edges in the input, and the
model was decoration. Such a method would pass straight through the localization
stage above: it would point at anatomy, beat the random baseline, and support a
completely unearned claim about what the model learned.

`sanity_checks.py` runs the **cascading model-parameter randomization test**.
Starting from the trained network it randomizes one stage at a time, from the
classifier down to the stem, re-explains the same images after each step, and
reports two things per stage:

- **rank correlation** — Spearman correlation against the trained model's own
  map. Should fall toward zero as randomization cascades; a method whose maps
  survive intact is not a function of the weights.
- **pointing game vs random baseline** — the same metric localization reports,
  recomputed at each step. Localization should collapse to baseline once the
  layers that produced it are noise.

Both are run for every explainer the architecture supports, because the question
is asked of the *method*, not the model. Two things are worth knowing before
reading the table:

- For a **class-agnostic** method the first randomized row cannot move by
  construction — Attention Rollout never reads the classifier, so randomizing
  the head is a no-op for it. Read its falloff from the rows below.
- The headline number is `stages_until_decorrelated`, not just the final row. A
  method can decorrelate eventually and still be nearly untouched by randomizing
  the layers it is nominally built from, and that is the interesting failure.

Enabled by default (`config.SANITY_CHECK_ENABLED`); runs on a seeded subsample
of the annotated instances (`config.SANITY_CHECK_SAMPLES`, default 64), since it
re-explains the set once per stage per method. Nothing is mutated: every
randomization is applied to a deep copy, and the global RNG state is saved and
restored around the run.

## Tests

```bash
pytest
```

266 tests, no dataset required — everything is synthetic, so the suite
runs with the drive unmounted, no GPU, and no checkpoint on disk. Backbones are
built with `pretrained=False`, so nothing downloads weights either. It covers
the pure functions behind the reported numbers:

| Area | What is pinned |
| ---- | -------------- |
| `metrics.py` | NaN-vs-zero handling for unscorable classes; that training's per-epoch AUROC and evaluation's macro AUROC agree on identical input |
| `dataset.py` | train/val patient-disjointness; nested subsets (5k ⊂ 15k ⊂ 30k); that a stratified prefix tracks the pool's label distribution better than the best of 20 random draws |
| preprocessing | bicubic resampling in both pipelines; that train and eval share one geometry; that no centre crop and no horizontal flip creep in; grayscale→3-channel replication |
| `localization.py` | box scaling, mask union and clipping, largest-connected-component detection, and the IoBB denominator (intersection over the *predicted* box, per Wang et al.); that the eval transform still matches the box scaling, probed behaviourally with a non-square image rather than by inspecting the transform list; that a signal-free heatmap is a pointing-game miss rather than a point at the top-left corner |
| `sanity_checks.py` | that every architecture's randomization stages cover the model exactly once, top-down; that randomization re-initializes weights *and* BatchNorm running statistics, reaches bare parameters with no `reset_parameters`, and changes the model's output; that a blank map is excluded from the rank correlation rather than scored as decorrelated |
| `evaluate.py` | confusion-sweep counts against a brute-force scan; that a class scored entirely below the old grid floor still tunes and still predicts positives; the low-support and degenerate fallbacks; that fixed-sensitivity mode reaches its target and pays for it in specificity; ECE under both binning strategies and that no prediction is dropped from either; that samples-F1 credits a correctly-predicted normal study; that bootstrap intervals contain their own point estimate, widen for rare classes, widen again under patient grouping, and are withheld entirely for unscorable classes; prediction round-trips |
| `train.py` | asymmetric loss under fp16 and saturating logits; head/backbone split for all five architectures; that freezing also stops BatchNorm statistics drifting |
| `models.py` | that the factory and `SUPPORTED_MODELS` describe the same roster; that neither timm tag pulls in 21k pretraining; ViT's prefix-token count; that MaxViT is what fixes `IMAGE_SIZE` at 224; that the XRV adapter reproduces `xrv.utils.normalize` exactly, that no XRV checkpoint trained on ChestX-ray14 can be selected, and that a newly released XRV tag fails the suite rather than defaulting to allowed |
| `explainability.py` | ViT token→grid and Swin NHWC reshapes; that every architecture's Grad-CAM target layer still resolves; the 14×14-vs-7×7 CAM grid asymmetry; that Attention Rollout actually captures attention and restores the model afterwards; Grad-CAM under a frozen backbone; hook cleanup; that both explainers expose the pre-normalization map the energy metric needs, and that rollout's floor is what makes that distinction real; that a live Grad-CAM hook stays inert outside its own `generate_batch`, including under `torch.inference_mode()` |

The suite was checked by mutation: deliberately switching the IoBB denominator
to the union, dropping the loss's fp32 cast, moving the ViT Grad-CAM target to
the final norm, leaving timm's fused attention enabled so rollout silently
captures nothing, leaving a stale architecture in the Grad-CAM target table,
hardcoding the wrong prefix-token count, falling back to bilinear resampling,
inserting a centre crop into the eval pipeline without updating the box scaling,
scoring the energy fraction on the normalized instead of the raw heatmap,
leaving a randomization stage out of the coverage,
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
resolves to, `torchxrayvision` pins the URL each XRV tag points at,
`scikit-learn` implements every metric, `Pillow` decodes the images — so two
results files written months apart can be checked for comparability rather than
assumed to be comparable. `run.weight_tag` records the checkpoint fine-tuning
started from, resolved to the real tag rather than echoing `--weight-tag`, so an
un-overridden run still identifies its own starting point (`null` for the
torchvision-backed models, which the `torchvision` version already pins).
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
│                  # <model>_localization_<method>.json,
│                  # and <model>_sanity_checks.json
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
