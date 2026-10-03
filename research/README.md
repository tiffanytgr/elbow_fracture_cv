# Research code: multiclass baseline and paper figures

Code supporting the manuscript, kept separate from the deployed inference app
(`pipeline/`, `elbow_grader/`, `backend/`). Nothing here is imported at
inference time.

Added in response to two reviewer requests:

1. **A comparison baseline for end-to-end multiclass classification** — given a
   paired AP and LAT radiograph, predict Normal / Grade 1 / Grade 2A /
   Grade 2B / Grade 3 directly, and compare against the hierarchical cascade on
   the same held-out patients.
2. **Paper-ready panel figures** — Figures 1–4 as composed multi-panel figures,
   with ROC curves, training curves, Grad-CAM, ablations and failure cases
   moved to supplementary figures.

```
research/
├── multiclass_baseline/   training + evaluation of the end-to-end baseline
└── figures/               the four main figures and the supplements
```

## Verify the setup first

```bash
bash research/smoke_test.sh
```

Runs the whole chain — manifest, both architectures, evaluation with coverage
filtering, the paired comparison, and two figures — on 45 synthetic noise
images, in about a minute on CPU. It checks that the code runs and that the
artefacts connect to each other, not that anything learns. Worth running before
committing GPU time.

## Install

The two halves have different requirements on purpose: the figure scripts have
no torch dependency, so they run on a laptop with no GPU stack.

```bash
pip install numpy scipy matplotlib pillow          # figures only
pip install torch torchvision                      # also needed to train
```

## 1. End-to-end multiclass baseline

### Build the paired manifest

The cascade trains from per-view manifests (`train.csv`, `LAT_train.csv`, each
`Filepath,Label`). The baseline needs one row per patient with both views, so
the manifest step pairs them on a case ID extracted from the filename, and
takes the split from the existing manifests — the baseline must be evaluated on
exactly the same patients as the cascade, or the comparison means nothing.

```bash
python -m research.multiclass_baseline.manifest \
    --ap-csv  train=experiments/train.csv \
    --lat-csv train=experiments/LAT_train.csv \
    --ap-csv  val=experiments/val.csv \
    --lat-csv val=experiments/LAT_val.csv \
    --ap-csv  test=experiments/test.csv \
    --lat-csv test=experiments/LAT_test.csv \
    --out research/outputs/paired_manifest.csv
```

Or walk the dataset tree directly, taking the grade from the folder names and
only the split from the manifests:

```bash
python -m research.multiclass_baseline.manifest \
    --data-root "/path/to/Supracondylar Fracture Data Set" \
    --split-csv train=experiments/train.csv \
    --split-csv val=experiments/val.csv \
    --split-csv test=experiments/test.csv \
    --out research/outputs/paired_manifest.csv
```

**Check the run log before going further.** It prints the split × grade table
(compare it against Table 1) and names every case it dropped: AP without LAT,
grade disagreement between the two views, split disagreement, or more than one
file per view. These are data-quality findings, not noise.

**Case IDs.** The default pattern is built from the filename grammar in the KKH
manifests and keeps a trailing letter and a laterality suffix:

| filename | case ID |
|---|---|
| `A3517.png`, `A326 AP.png`, `A162 AP2.png` | `A3517`, `A326`, `A162` |
| `A241A AP.png`, `A594B AP.png` | `A241A`, `A594B` |
| `A118 (L).png`, `A118 (R).png` | `A118(L)`, `A118(R)` |

That trailing letter matters: `A241` and `A241A` are different patients with
different Gartland grades. A pattern that stopped at the digits would merge
them. Override with `--id-pattern` if the convention changes; group 1 is the ID.

### Train

```bash
python -m research.multiclass_baseline.train \
    --manifest research/outputs/paired_manifest.csv \
    --encoder-checkpoint experiments/checkpoints/autoencoder/encoder.pth \
    --out-dir research/outputs/multiclass_dual_seed0 \
    --seed 0
```

Architecture is a **dual-encoder ResNet-18**: one encoder per view, features
concatenated, a single 5-way head. Both encoders start from the same
autoencoder-pretrained weights the cascade nodes use — a baseline denied that
initialisation would be a strawman. `--architecture two-channel` trains the
cheaper single-encoder variant for the appendix.

Training follows report section 2.7.1 exactly, so the baseline differs from the
cascade in decision structure only:

| | setting |
|---|---|
| Stage 1 | encoders frozen (batch-norm in eval mode), head only, Adam, 10 epochs |
| Stage 2 | all parameters, backbone LR = 0.1 × head LR, cosine over 20 epochs |
| Loss | class-weighted cross-entropy, inverse-frequency weights |
| Augmentation | rotation ±15°, translate 5%, scale 0.95–1.05, sharpness, autocontrast, colour jitter |
| Selection | best validation accuracy |

Run three seeds (`--seed 0 1 2`) to match the cascade's protocol.

> The checkpoint is selected on validation **accuracy** while recall is the
> primary clinical metric — the objective mismatch the report already notes in
> section 4.6. The default reproduces the cascade's behaviour so the comparison
> is like-for-like; `--select-on macro_recall` selects on balanced recall
> instead. If you report that variant, report it for both arms.

### Evaluate

```bash
python -m research.multiclass_baseline.evaluate \
    --manifest research/outputs/paired_manifest.csv \
    --checkpoint research/outputs/multiclass_dual_seed0/best_model.pth \
    --split test \
    --coverage-from research/outputs/drue_retained_cases.txt \
    --out-dir research/outputs/multiclass_eval
```

Writes `predictions.csv` (one row per patient) and `metrics.json`: the 5×5
confusion matrix, per-grade recall and precision with Wilson CIs, overall and
macro accuracy, quadratic-weighted κ with a bootstrap CI, the adjacent-versus-
distant error split, the 2A↔2B boundary error count, the operative collapse,
and the risk–coverage and calibration curves.

`--coverage-from` takes the DRUE-retained patient list (one case ID per line,
or a CSV with a `case_id` column) so the filtered row covers the same 178
patients as the cascade's filtered row. Without it, metrics are unfiltered.

### Compare

```bash
python -m research.multiclass_baseline.compare \
    --cascade research/outputs/cascade_predictions.csv \
    --multiclass research/outputs/multiclass_eval/predictions.csv \
    --out research/outputs/comparison.json
```

Restricted to the patients both files contain, so every test is paired: exact
McNemar on per-patient correctness, the κ difference with a paired bootstrap
CI, per-grade recall differences, and boundary-error counts. It refuses to run
if the two files disagree on the reference standard for any shared case.

The cascade file needs only `case_id,true_idx,pred_idx` (or `true_grade` /
`pred_grade` as text). Rows where the cascade abstained — `pred_idx` of `-1` or
empty — are dropped with a count in the log rather than scored as wrong.

## 2. Figures

```bash
python -m research.figures.fig1_pipeline       --results research/outputs
python -m research.figures.fig2_classification --results research/outputs
python -m research.figures.fig3_uncertainty    --results research/outputs
python -m research.figures.fig4_anatomy        --results research/outputs
python -m research.figures.appendix_figures    --results research/outputs
```

Each writes PDF (submission), TIFF at 600 dpi (production) and PNG (drafts) to
`--out-dir`. Restrict with `--formats pdf`.

| | content |
|---|---|
| **Figure 1** | pipeline schematic; representative AP and LAT processing strips |
| **Figure 2** | 5×5 confusion matrices (cascade, baseline); per-grade recall comparison; error composition |
| **Figure 3** | risk–coverage; uncertainty/error discrimination; calibration; four-condition ablation |
| **Figure 4** | AHL, cortical-width and Baumann examples with their distributions |
| **S1–S5** | ROC, training curves, Grad-CAM, full ablation, failure cases |

### Checking layout before the results exist

```bash
python -m research.figures.fig2_classification --demo
```

`--demo` renders from data synthesised to reproduce the published summary
statistics (65.9% overall, 56.8% macro, κ 0.755 — Table 3), so composition can
be reviewed now. **Demo output is stamped "SYNTHETIC DEMO DATA" across the
figure and suffixed `_DEMO`.** It is for layout only; the numbers are not
results. The supplementary figures have no demo mode — they skip panels whose
inputs are absent and say which.

### Input files

All under `--results`. Each is optional; a figure that cannot find its input
says so and exits non-zero (the supplements skip and continue).

| file | columns |
|---|---|
| `cascade_predictions.csv` | `case_id,true_idx,pred_idx[,confidence]` |
| `multiclass_predictions.csv` | written by `evaluate.py` |
| `comparison.json` | written by `compare.py` |
| `uncertainty.csv` | `case_id,drue_score[,correct,confidence]` |
| `ablation.csv` | `condition,node,recall[,weighted_f1,coverage]` |
| `cortical_width.csv` | `case_id,grade,match_ratio` |
| `cortical_profiles.csv` | `case_id,grade,position,normalised_width` |
| `baumann.csv` | `case_id,grade,baumann_angle` |
| `ahl.csv` | `case_id,grade,bisects_capitellum` |
| `roc_curves.csv` | `node,fpr,tpr[,auc]` |
| `training_curve.csv` | written by `train.py` |
| `autoencoder_curve.csv` | `epoch,train_loss,val_loss` |
| `ablation_full.csv` | `configuration,node,metric_value[,seed]` |
| `failure_cases.csv` | `case_id,true_grade,pred_grade[,confidence,note]` |
| `images/` | representative PNGs — see each figure's module docstring for filenames |

Missing images render as obvious labelled placeholders, so composition can be
reviewed before the representative cases are chosen.

### Style

`figures/style.py` holds the whole look: RSNA column widths, sans-serif type,
600-dpi TIFF, bold uppercase panel labels.

Colour follows two rules that are kept apart. **Ordinal severity is magnitude**,
so Normal → Grade 3 uses a single hue light-to-dark; a categorical rainbow over
an ordered scale would hide the ordering quadratic-weighted κ depends on.
**Model identity is categorical**, so it uses a fixed hue order assigned by
entity and never cycled — the cascade is blue and the baseline orange in every
panel. The categorical set passes colorblind-safe checks under all-pairs
comparison, with one pair in the 6–8 ΔE band that is always backed by hatch
texture and direct labels. Print only; there is no dark mode.

## Reproducing everything

```bash
python -m research.multiclass_baseline.manifest --ap-csv ... --out research/outputs/paired_manifest.csv
for seed in 0 1 2; do
  python -m research.multiclass_baseline.train \
      --manifest research/outputs/paired_manifest.csv \
      --encoder-checkpoint experiments/checkpoints/autoencoder/encoder.pth \
      --out-dir "research/outputs/multiclass_dual_seed${seed}" --seed "$seed"
done
python -m research.multiclass_baseline.evaluate \
    --manifest research/outputs/paired_manifest.csv \
    --checkpoint research/outputs/multiclass_dual_seed0/best_model.pth \
    --coverage-from research/outputs/drue_retained_cases.txt \
    --out-dir research/outputs/multiclass_eval
cp research/outputs/multiclass_eval/predictions.csv research/outputs/multiclass_predictions.csv
python -m research.multiclass_baseline.compare \
    --cascade research/outputs/cascade_predictions.csv \
    --multiclass research/outputs/multiclass_predictions.csv \
    --out research/outputs/comparison.json
for f in fig1_pipeline fig2_classification fig3_uncertainty fig4_anatomy appendix_figures; do
  python -m "research.figures.$f" --results research/outputs
done
```

`research/outputs/` is git-ignored; nothing patient-derived is committed.
