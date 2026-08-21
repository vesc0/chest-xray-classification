# NIH Chest X-Ray Classification

Multi-label classification of the 14 ChestX-ray14 pathologies, across five
architecture families, with calibrated thresholds, bootstrap confidence
intervals, and explainability that is itself sanity-checked.

## Contents

- [Setup](#setup)
- [Usage](#usage)
- [Models](#models)
- [Pipeline](#pipeline)
- [Modules](#modules)
- [Evaluation](#evaluation)
- [Subgroup analysis](#subgroup-analysis)
- [Explainability](#explainability)
- [Ensembling](#ensembling)
- [Outputs](#outputs)
- [Tests](#tests)
- [Notebooks](#notebooks)
- [Design notes](#design-notes)

## Setup

```bash
pip install -r requirements.txt
```

Use `requirements.lock.txt` instead to reproduce published numbers — it is
`pip freeze` output pinning the exact environment the results came from, where
`requirements.txt` only declares ranges.

Point the pipeline at your copy of the NIH dataset (the folder holding
`Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt` and
`images_001/` … `images_012/`):

```bash
export XRAY_DATASET_ROOT=/path/to/archive-chest-xrays-nih
```

Without it the default in `config.py` is used. A missing dataset, or too few
images resolving on disk, fails the run immediately rather than training on a
partial dataset. Pretrained weights download on first use per architecture and
are cached; the test suite downloads nothing.

## Usage

`--model` is required. A run trains on the full training pool by default;
`--subset N` scales the training set only, leaving validation and test fixed.

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

# Ensemble the finished runs (no training, no inference)
python ensemble.py --auto
```

Common overrides: `--epochs`, `--batch-size`, `--lr`, `--num-workers`,
`--image-size`, `--weight-tag`, `--tuning-mode {full,head_only,partial}`,
`--loss {asymmetric,weighted_bce,bce}`,
`--checkpoint-metric {val_loss,val_auroc,val_auprc}`,
`--threshold-metric {f1,fbeta,youden,sensitivity}`, `--target-sensitivity`,
`--ece-bin-strategy {quantile,uniform}`, `--xai-samples N`.
`python main.py --help` lists them all.

## Models

One backbone per architecture family, so a difference between two results is
attributable to the architecture rather than to capacity, resolution, or
pretraining data. All are ImageNet-1k pretrained at 224×224 with ImageNet
normalization.

| `--model`      | Architecture | Family         | Params | Weights |
| -------------- | ------------ | -------------- | ------ | ------- |
| `densenet121`  | DenseNet-121 | CNN baseline   | 7.0M   | torchvision `IMAGENET1K_V1` |
| `vit_s_16`     | ViT-S/16     | pure ViT       | 21.7M  | timm `deit3_small_patch16_224.fb_in1k` |
| `convnextv2_t` | ConvNeXtV2-T | modern CNN     | 27.9M  | timm `convnextv2_tiny.fcmae_ft_in1k` |
| `swin_t`       | Swin-T       | modern ViT     | 27.5M  | torchvision `IMAGENET1K_V1` |
| `maxvit_t`     | MaxViT-T     | hybrid CNN/ViT | 30.4M  | torchvision `IMAGENET1K_V1` |

Two more are buildable but off the roster: `swin_v2_t` (kept so its documented
224-transfer failure stays reproducible) and `densenet121_xrv` (DenseNet-121
initialized from [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
chest-radiograph weights, isolating pretraining domain against the `densenet121`
control). `--model all` runs everything except `densenet121_xrv`; run that one
by name.

`--weight-tag` swaps the pretrained checkpoint on the timm models and selects
the pretraining corpus for `densenet121_xrv`. It is guarded: tags expecting
non-ImageNet normalization are rejected, as are XRV checkpoints trained on
ChestX-ray14 or datasets derived from it, which would have seen this pipeline's
test images. See [design notes](docs/design-notes.md#supported-models).

## Pipeline

Data indexing (`dataset.py`) → backbone with a 14-class head (`models.py`) →
training under asymmetric loss (`train.py`) → per-class threshold calibration on
validation and evaluation on the held-out test set (`evaluate.py`) → Grad-CAM
and Attention Rollout heatmaps (`explainability.py`) → scoring those heatmaps
against ChestX-ray14's 984 ground-truth boxes (`localization.py`) → randomizing
the model's weights to confirm the explanations track them (`sanity_checks.py`).
`main.py` orchestrates the whole thing.

Evaluation data is fixed across every run: test is the full official 25,596-image
split and is never subset; validation is 8,652 images carved once from the
train_val pool before any subsetting. Both are patient-disjoint and
label-stratified, and training subsets are nested (5k ⊂ 15k ⊂ 30k).

## Modules

| Module | Role |
| ------ | ---- |
| `config.py` | Hyperparameters, paths, feature switches |
| `device.py` | Accelerator selection (CUDA > MPS > CPU) and timing synchronization |
| `dataset.py` | Metadata parsing, augmentation, multi-hot labels, group-aware splits |
| `models.py` | The architecture roster, pinned weight tags, and checkpoint guards |
| `metrics.py` | AUROC/AUPRC shared by training and evaluation, so both agree |
| `train.py` | Training loop, optimizer, mixed precision, asymmetric loss |
| `evaluate.py` | Inference, metrics with bootstrap CIs, threshold tuning, saved probabilities |
| `threshold_analysis.py` | Post-hoc comparison of thresholding schemes. Runs no model |
| `bias_analysis.py` | Sex- and age-stratified AUROC and miss rates. Runs no model |
| `ensemble.py` | Averages saved probabilities of finished runs. Runs no model |
| `explainability.py` | Grad-CAM (all architectures) and Attention Rollout (ViT) |
| `localization.py` | Pointing game, energy fraction, IoU/IoBB against the boxes |
| `sanity_checks.py` | Cascading model-parameter randomization (Adebayo et al., 2018) |
| `utils.py` | Seeding, run logging, curves, comparison tables |
| `main.py` | Command-line interface and orchestration |

## Evaluation

Architecture comparison rests on **AUROC and AUPRC**, which are threshold-free.
Every one of them — per class and macro — carries a bootstrap confidence
interval, because a run produces one seed per configuration and five point
estimates cannot support a ranking claim. **If two models' intervals overlap,
the comparison is not settled.** Whole patients are resampled, not images.

When a threshold is needed it is fitted **per class, on validation only**, and
frozen before the test set is read. F1 is the default objective; `youden` and
`sensitivity` are available. Candidates are the model's own scores rather than a
grid, and a class needs 50 validation positives to be fitted at all — classes
below that stay at 0.5 and are marked `*` in the results table.

Raw validation and test probabilities are saved per model, so re-thresholding
costs no inference:

```bash
python threshold_analysis.py --model all --experiment full_dataset --metric f1 youden sensitivity
```

That also bootstraps the **threshold itself** by refitting on validation
resamples. On rare classes it is the wider of the two intervals.

Every run additionally records wall-clock training and inference cost, together
with the device, batch size, workers, AMP setting and package versions that
produced it.

## Subgroup analysis

Macro AUROC hides who the errors land on. `bias_analysis.py` re-attaches the
patient metadata behind every saved test prediction and asks the two questions
separately:

- **Stratified AUROC** — does the model *rank* worse for a group?
- **Stratified FNR, and the share of abnormal studies where no class fires at
  all** — does the group get *missed* more at the deployed threshold?

They can disagree, and that is the useful part: equal AUROC with unequal FNR
puts the disparity in the operating point rather than the features.

```bash
python bias_analysis.py --all
```

Thresholds stay global. Refitting them per subgroup would answer "could a
group-aware model do better?", not "what does the single shipped threshold do
to each group?" — and only the second question is about a deployable system.

Every gap is reported against a reference group (the largest level on each axis)
with a **patient-level bootstrap interval on the difference**, both levels scored
inside the same resample. Two overlapping one-group intervals are not evidence of
no difference; the paired interval is what settles it.

One detail does most of the work. A macro taken over whichever classes happen to
be scorable in each group compares different diseases and invents disparities
that are not there: Hernia has a single positive among the 6,908 test images aged
20-39, and that one case moves the band's 14-class macro AUROC by five points.
Each pair is therefore restricted to the classes clearing ten positives **in both
the group and the reference**, and the reported `cls` column says how many
survived. Left uncorrected, this alone produced a spurious five-point deficit in
the 20-39 band and a four-point one in the 80-100 band, both of which vanish
under the shared class set.

Across all 22 saved runs the result is one-sided. **Sex is a bounded null**: the
median AUROC gap is -0.0014 and no run's interval excludes zero, with a median
half-width of +/-0.014 — enough to rule out sex gaps larger than about one and a
half AUROC points, not enough to call any smaller one. **The youngest band is
not**: patients aged 0-19 carry a median +0.072 FNR gap, resolved in 21 of 22
runs, while their AUROC gap stays indistinguishable from zero. The model ranks
these patients as well as anyone and still misses more of their findings, which
places the disparity squarely in the shared operating point.

Nothing in the sweep closes it. Overall AUROC and the 0-19 FNR gap correlate at
r = 0.11 across the 22 runs: the best model here (the stacked ensemble, 0.827
macro AUROC) still misses +0.056 [+0.016, +0.112] more, and twelve points of
macro AUROC bought across architecture family, pretraining source, input
resolution, training-set size and ensembling buy no measurable fairness. The
80-100 band is reported but underpowered — 31 patients and nine comparable
classes — and resolves nothing.

The framing follows Seyyed-Kalantari et al. (2021), narrowed to the two
attributes ChestX-ray14 actually carries — sex and age. Race and insurance,
which carry their largest disparities, are not in this dataset. Two limits
belong with any reading of these numbers:

- The labels are NLP-mined from reports. This measures disparity against *noisy*
  labels, and label noise that itself tracks age cannot be separated from model
  behaviour with this data alone.
- Subgroup prevalence differs, so a shared threshold produces different FNRs
  partly through calibration. That is a property of deploying one threshold, not
  an artefact — which is exactly why it is the thing worth measuring.

## Explainability

Grad-CAM runs for **every** architecture, since comparing localization across
models explained by different methods is meaningless; each registers its own
reshape transform for non-NCHW features. Attention Rollout is ViT-only by
necessity. Heatmaps are scored against ChestX-ray14's hand-drawn boxes, and
`sanity_checks.py` then re-runs each explainer with the weights progressively
randomized, checking the maps actually depend on the model.

Two asymmetries to read the numbers with: ViT-S/16's 14×14 CAM grid inflates its
IoU/IoBB against the others' 7×7 (the pointing game is the fairer comparison),
and Rollout's near-zero pointing scores are a property of the method, not a bug.
Both are explained in the [design notes](docs/design-notes.md#explainability-per-architecture).

## Ensembling

Because probabilities are saved and rows are aligned across runs by
construction, ensembling needs neither the dataset nor a GPU:

```bash
python ensemble.py --auto            # best run per family, equally weighted
python ensemble.py --auto --dry-run  # show which runs the rule picks, and why
python ensemble.py --auto --select diverse   # select for complementarity instead
python ensemble.py --auto --rule stack       # fitted per-class combiner (best measured)
python ensemble.py --members s3_swin_384:swin_t s2_maxvit_full:maxvit_t
```

Members are chosen by a stated rule rather than a search, and the run bootstraps
its margin over the best single member on shared patient resamples. The gain is
in ranking quality (macro AUPRC +0.0185 here) rather than at the operating
point. Ensembles are written as their own experiment and are never selected as
members of another ensemble.

`--select diverse` grows the set by adding the candidate maximizing
`val AUROC - w x error correlation with the members already chosen`, keeping it
only if validation AUROC improves. Correlation is measured **within each true
class**: correlating the raw residual `p - y` puts every pair above 0.9, because
at this prevalence almost every row is a negative and two models agreeing the
disease is rare tells you nothing. Every run also prints the resulting
correlation and decision-disagreement matrices — mean 0.78 and 0.44 here, with
the two Swin runs at 0.88/0.32 (near-duplicates) and the medically-pretrained
DenseNet at 0.69-0.75/0.51-0.53 (the most complementary member).

### Combining the members

Fifteen combination rules were compared under one protocol: fit on validation,
then score the half of validation the rule never saw (patient-grouped, 12
splits). Test is reported afterwards and never used to choose. The holdout
column is the one that decides:

| rule | holdout AUROC | test AUROC | test AUPRC |
|---|---|---|---|
| mean (default) | 0.8299 | 0.8254 | 0.3008 |
| logit mean / median / trimmed | 0.8284-0.8298 | 0.8237-0.8252 | 0.2953-0.3005 |
| weights ∝ val AUROC | 0.8300 | 0.8256 | 0.3009 |
| per-class AUROC weights | 0.8307 | 0.8258 | 0.3022 |
| hill-climb with replacement | 0.8296 | *0.8269* | 0.3015 |
| diversity-weighted | 0.8289-0.8298 | 0.8241-0.8254 | 0.3000-0.3010 |
| **stacking (`--rule stack`)** | **0.8325** | **0.8269** | **0.3046** |
| stacking, cross-class features | 0.8221 | 0.8150 | 0.2928 |

Two rows are worth reading twice. **Hill-climbing** posts the joint-best test
AUROC and is the *worst* weighted rule on holdout — its test number is luck, and
this is the same overfitting the half-split exposed for member selection.
**Cross-class stacking** (84 features per class instead of 6) has the second-best
in-sample validation score in the whole table and the worst holdout and test
scores: a textbook demonstration of why the fitted rules need a holdout at all.

Stacking wins, and is the only rule that improves the operating point rather
than just the ranking:

| | mean | stack | delta |
|---|---|---|---|
| macro AUROC | 0.8254 | 0.8269 | +0.0015 |
| macro AUPRC | 0.3008 | 0.3046 | +0.0038 |
| macro F1 | 0.3293 | 0.3362 | +0.0069 |
| **macro ECE** | 0.2837 | **0.0275** | **-0.2562** |
| macro Brier | 0.1412 | 0.0590 | -0.0822 |

The calibration result is the largest effect anywhere in this project, and it is
free. The members are trained with asymmetric loss, which deliberately distorts
predicted probabilities to cope with the long tail; averaging six such models
preserves the distortion, so the mean ensemble ranks well and its "0.7" means
nothing. The stacker refits a logistic model on the members' log-odds, which
optimizes log-loss and therefore recovers calibration — ECE drops by roughly a
factor of ten on every class fitted. On a task where a predicted probability is
meant to inform a decision, that matters more than the 0.0015 of AUROC.

Two implementation details are load-bearing. One combiner is fitted **per class**,
because no single global weighting can say that Swin should dominate Emphysema
while the medically-pretrained DenseNet carries Cardiomegaly — which is what the
printed coefficients show it doing. And validation predictions are produced
**out-of-fold** (5 patient-grouped folds), because a stacker's in-sample
validation scores are better than it will ever achieve again; tuning thresholds
against them calibrates to a performance the model does not have, and costs
0.006 macro F1 on test, silently. Classes below 50 validation positives are left
at the plain mean, so Hernia is never fitted from 14 examples.

Both selection rules pick the same six members on these runs. The pool is saturated:
enumerating every equal-weight subset at sizes 2-8 and keeping the best *by test
score* — a cheat no honest method can match — reaches 0.8258 against the rules'
0.8254, roughly a tenth of the bootstrap interval on the ensemble's own margin.
Greedy hill-climbing with replacement appears to reach 0.8269 by picking the
strongest run twice, but selecting on half of validation and scoring on the
unseen half gives 0.8320 against the fixed rule's 0.8333, winning in 15% of
splits — the gain is selection overfitting, which is why neither rule optimizes
weights. Going further needs a member that fails differently (a different input
representation, label policy, or objective), not another backbone.

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
true positives, false negatives, false positives, and the best/worst cases by
heatmap energy inside the box. `--xai-samples N` renders N extra unselected test
images; its one real use is the six classes with no ground-truth boxes.
`ensemble.py` writes the same layout minus `checkpoints/` and `xai/`.

## Tests

```bash
pytest
```

266 tests, no dataset, GPU, checkpoint or network required — everything is
synthetic and backbones are built with `pretrained=False`. They pin the pure
functions behind the reported numbers, and the suite was checked by mutation:
switching the IoBB denominator to the union, dropping the loss's fp32 cast,
bootstrapping images instead of patients, inserting a centre crop, and a dozen
other deliberate breakages each fail a test. Full coverage table in the
[design notes](docs/design-notes.md#tests).

## Notebooks

`notebooks/` walks through the dataflow step by step (`1-EDA`,
`2-Data-Preprocessing`, `3-Model-Training`, `4-Model-Evalutaion`, `5-XAI`).
They are explanatory, not required by the pipeline — except `1-EDA.ipynb`, which
is worth reading first.

## Design notes

[`docs/design-notes.md`](docs/design-notes.md) holds the reasoning this README
only summarizes: the experimental protocol and preprocessing choices, why each
backbone and checkpoint was picked, the Grad-CAM target-layer rule and its
exceptions, how thresholds and bootstrap intervals are constructed and what they
do and do not cover, ensemble member selection, and what the test suite pins.
