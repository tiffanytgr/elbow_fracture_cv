# Master Manuscript Outline
### An Anatomically Grounded AI Framework for Automated Gartland Classification of Pediatric Supracondylar Humerus Fractures
*Target: Radiology: Artificial Intelligence (ORIGINAL RESEARCH). Format and section order follow the RSNA house style.*

> **Reading guide.** Prose is written toward submission quality. `[brackets]` mark values not
> yet available (measurement-module metrics, cohort counts, IRB details). Real classifier
> numbers are filled from the ablation grid (`ablation_results_exp1–4.csv`, post-marker-removal
> + alignment, DRUE-filtered, 3 seeds). All pre-submission actions and citations-to-verify are
> collected at the end (Parts F–G) rather than scattered inline.

---

## PART A — Related work and positioning *(condensed; grounds the Introduction and Discussion)*

**The verified gap.** Published AI for pediatric elbow radiographs performs fracture
*detection* / normal-versus-abnormal triage, or at most binary operative-versus-nonoperative
triage. No verified prior system produces a full ordinal Gartland grade *together with*
automated AP- and lateral-view anatomic measurements and a learned-versus-anatomic consistency
check. That triad is this paper's contribution; the individual components are all off-the-shelf.

| Prior work | Task | Data | Result | Relation to this study |
|---|---|---|---|---|
| Tan MB et al., *Singapore Med J* 2025 (PMID 40258236) ⚠ | Lateral: normal vs abnormal | 1,314 radiographs | Acc 80.4%, AUROC 0.872 | Institutional comparator; **detection only** |
| DL meta-analysis, PMID 39976732 ⚠ | Detection (synthesis) | multiple | — | Confirms field = detection, not grading |
| Radiomics SCHF, *Medicine* 2024 (PMID 38847664) ⚠ | Fracture vs normal | 411+190 ROIs | AUC 0.65–0.72 | Needs **manual ROIs**; weak; still binary |
| **AI triage, *Injury* 2026 (S0020138326001440) ⚠** | **Operative (IIB/III/flexion) vs non-operative** | **1,811 AP+lat, single centre 2010–2017** | **Acc 77%, AUROC 0.84, sens 69%** | **Closest prior art — same boundary; black-box binary, no measurement** |
| Hierarchical femur cascade, *Eur J Radiol* 2020 ⚠ | AO/OTA cascade | — | — | Cascade precedent (not elbow) |
| SAM-family zero-shot bone, arXiv:2411.08629 ⚠ | Bone segmentation | CT/MRI | — | SAM2 precedent; unvalidated on pediatric elbow x-ray |

**Positioning rules (load-bearing):**
1. Do **not** claim novelty of operative/non-operative triage — *Injury* 2026 reported it. Claim
   ordinal grading + interpretable anatomic measurement + cross-module consistency.
2. The *Injury* 2026 model still reached only **69% sensitivity** at this boundary — external
   evidence that IIB recall is the genuinely hard, unsolved metric (supports the primary-endpoint
   choice).
3. Confirm the *Injury* 2026 / Tan 2025 cohorts are not your own group's data; if they are, state
   the provenance relationship explicitly.

---

## PART B — Front matter

**ORIGINAL RESEARCH**

### Title
An Anatomically Grounded Deep Learning Framework for Automated Gartland Classification of
Pediatric Supracondylar Humerus Fractures Using Hierarchical Classification and Anatomic
Measurement

**Authors** *(Name, degree, affiliation superscripts; confirm order + equal-contribution before submission.)*

### Structured Abstract

**Purpose:** To develop and evaluate an anatomically grounded artificial intelligence framework
for automated Gartland classification of pediatric supracondylar humerus fractures (SCHF) from
AP and lateral elbow radiographs, and to characterize its performance at the surgically decisive
Grade IIB boundary.

**Materials and Methods:** This retrospective, IRB-approved study included [n] patients ([date
range], [institution]). Paired AP and lateral radiographs were graded by [two blinded pediatric
orthopedic surgeons with third-reader adjudication / single expert — state]. All partitioning was
patient-level. The framework comprised (a) anatomic image standardization (marker removal,
lateral-view alignment); (b) an autoencoder-pretrained hierarchical ResNet-18 classifier cascade
(four binary nodes along the Gartland decision pathway); (c) view-specific anatomic measurement
(automated Baumann angle on AP; anterior humeral line, capitellum localization, and cortical-
width profiling on lateral); and (d) cross-module consistency verification. Out-of-distribution
cases were removed with a 95th-percentile DRUE threshold. The primary endpoint was Grade IIB
recall. Proportions are reported with Clopper–Pearson 95% CIs; results are 3-seed mean ± SD.

**Results:** [n] patients ([n] male; mean age [X] ± [SD] years) with [n] radiographs were
included. Node-conditional recall (3-seed mean ± SD) was 96% ± 1 for fracture detection, 49% ± 5
for Grade III, and 82% ± 5 for Grade II. Grade IIB recall — the primary endpoint — was 33% ± 19
under weighted-F1 model selection versus 60% ± 14 under recall-oriented selection (pooled 95% CI
[17, 53] and [41, 77], respectively), reflecting the small IIB denominator ([n]/seed). Image
standardization succeeded in 95% of images; anatomic standardization and DRUE filtering improved
weighted F1 at every node. Automated Baumann angle showed a mean absolute error of [X]° (ICC
[X]; Bland–Altman bias [X]°). [Segmentation and consistency-flag results.]

**Conclusion:** In this feasibility evaluation, an anatomically grounded framework produced
interpretable Gartland grade predictions with auditable anatomic evidence and flagged unreliable
predictions for review. It is positioned as interpretable decision support rather than autonomous
grading; the wide confidence interval on the primary endpoint reflects cohort size, and
prospective and external validation are required before clinical use.

*Supplemental material is available for this article. © RSNA, 2026*

### Abbreviations
AHL = anterior humeral line, AP = anteroposterior, AUC = area under the ROC curve, CI =
confidence interval, CRPP = closed reduction and percutaneous pinning, DRUE = dual reconstruction
uncertainty estimation, Grad-CAM = gradient-weighted class activation mapping, ICC = intraclass
correlation coefficient, IoU = intersection over union, PCA = principal component analysis, SAM2
= Segment Anything Model 2, SCHF = supracondylar humerus fracture, YOLO = you only look once

### Summary
An anatomically grounded AI framework integrating hierarchical deep learning with automated AP-
and lateral-view anatomic measurements provided interpretable decision support for Gartland
classification of pediatric supracondylar humerus fractures, evaluated at the surgically decisive
Grade IIB boundary as the primary endpoint.

### Key Points *(each carries a number)*
- ■ An autoencoder-pretrained hierarchical ResNet-18 cascade classified Gartland grade from paired
  AP and lateral radiographs; node-conditional Grade IIB recall at the surgical boundary was [final]% (95% CI [X, X]).
- ■ View-specific measurement modules produced clinically recognized quantities — automated
  Baumann angle (MAE [X]°; ICC [X]) on AP and anterior humeral line and cortical-width analysis on
  lateral — as auditable evidence for each prediction.
- ■ Cross-module disagreement between the learned classifier and independent anatomic measurement
  flagged [X]% of cases and detected misclassifications with [AUC/sensitivity X].

### Keywords
Pediatric, Elbow, Supracondylar Humerus Fracture, Gartland Classification, Deep Learning,
Hierarchical Classification, Segmentation, Anatomic Measurement, Explainability, Baumann Angle

---

## PART C — Introduction *(untitled in the journal; ends on the purpose statement)*

**¶1 Clinical burden.** SCHF are the most common operatively managed pediatric elbow injury, with
peak incidence at [age]. [Epidemiology + operative importance.]

**¶2 The decision the system supports.** The modified Gartland classification directs management:
type I is immobilized; types IIB and III typically require closed reduction and percutaneous
pinning. The IIA/IIB and Grade III boundaries thus separate nonoperative from operative care.

**¶3 Why interpretation is hard, exactly at the boundary.** Grading integrates AP alignment and
Baumann angle, the lateral AHL relationship, and cortical continuity. Aggregate interobserver
agreement is substantial (κ ≈ 0.77–0.85) but degrades at the type II sub-grades, with reported κ
as low as [0.26–0.43] for IIB — the distinction that determines surgery. Manual Baumann-angle
measurement itself carries observer variability (inter-reader ICC ≈ 0.74–0.77).

**¶4 What AI has and has not done.** Prior deep learning detects fractures or triages normal
versus abnormal (Tan 2025; meta-analysis PMID 39976732), and one study reported binary operative-
versus-nonoperative triage (*Injury* 2026); radiomics grading needs manual ROIs and is weak (AUC
0.65–0.72). No verified system produces a full ordinal Gartland grade together with automated AP-
and lateral-view anatomic measurements and a learned-versus-anatomic consistency check.

**¶5 Technical rationale.** A single CNN can exploit markers or positioning shortcuts; the
framework therefore combines standardized input, radiograph-domain representation learning,
hierarchical grading that mirrors the clinical pathway, explicit anatomic measurement, and
consistency verification that surfaces disagreement rather than resolving it silently.

**¶6 Purpose.** Therefore, this study aimed to develop and evaluate an anatomically grounded
artificial intelligence framework for automated Gartland classification using hierarchical deep
learning and complementary AP- and lateral-view anatomic measurement modules, evaluated with the
surgically decisive Grade IIB boundary as the primary endpoint.

---

## PART D — Materials and Methods

### Patient Cohort
Retrospective study approved by [IRB], protocol [no.], [consent waived/obtained]. Consecutive
patients [date range] from [institution] with paired AP and lateral elbow radiographs were
included. Inclusion: [age threshold; paired views]. Exclusion: [missing view, poor quality,
cast/splint obscuring anatomy, prior fixation]. Inclusion is shown in Figure 2. [State the
external/supplementary "Pedia XHF" dataset and its role: it contributed to training/retraining;
whether it also contributed test cases must be stated — see Part F.] Acquisition and preprocessing
detail is in Appendix S1.

### Reference Standard
[Two] blinded pediatric orthopedic surgeons ([X], [Y] years) independently assigned the Gartland
grade; discordance was adjudicated by a third senior surgeon. Manual Baumann-angle measurements
were obtained independently by [two] readers to validate the AP measurement pipeline.
Interobserver agreement is reported as the reference ceiling.

### Study Design and Partitioning
All splits were patient-level; every image from a patient remained in one partition.
Partitioning preceded autoencoder pretraining, YOLO training, capitellum-regressor training,
classifier training, threshold selection, and geometric-pipeline development. Autoencoder
pretraining used training-partition images only. The test partition was not used for model,
hyperparameter, checkpoint, or threshold selection.

### AI Framework Overview
Four modules (Fig 1): image standardization; hierarchical classification; view-specific anatomic
measurement; and cross-module verification. Architecture and equations are in Appendix S1.

### Module 1 — Image Standardization
Radiographic text markers were detected and removed before training and inference, identically
across partitions, to reduce shortcut learning. Lateral radiographs were standardized by Otsu
thresholding, principal-component identification, left–right normalization by horizontal
flipping, PCA long-axis rotation, and elbow-region cropping; flipping precedes PCA rotation, and
the anterior edge is derived from the standardized orientation. Standardization succeeded in 95%
of images.

### Module 2 — Autoencoder-pretrained Hierarchical ResNet-18 Classification
A convolutional autoencoder (ResNet-18 encoder; residual decoder discarded after pretraining)
was pretrained on training-partition radiographs ([exact reconstruction loss]). Four binary
classifiers were arranged along the Gartland pathway: **Exp1** (AP, fracture vs normal), **Exp2**
(AP, Grade III vs I+II), **Exp3** (lateral, Grade II vs I), **Exp4** (lateral, Grade IIB vs IIA).
Each node was trained on the reference-standard-defined subset with two-stage transfer (Stage 1:
encoder frozen, head trained; Stage 2: encoder fine-tuned at a reduced backbone learning rate
[ratio]). A regularization factorial (label smoothing, MixUp, dropout) was evaluated over three
seeds (42/123/456) with class weights held constant. Within each run the epoch checkpoint was
selected by validation accuracy; the per-node operating configuration and the Exp4 threshold are
selected on the **validation** partition using a recall-oriented criterion (positive-class recall
or positive-class F1), with the full precision–recall curve reported. Grad-CAM from `layer4[-1]`
was used post hoc for attention assessment and as input to Module 4.

### Module 3 — View-specific Anatomic Measurement
**AP — automated Baumann angle.** A YOLO instance-segmentation model segmented the distal humerus
and shaft; the humeral component was isolated (largest connected component, forearm excluded by
[rule]); PCA on the proximal [fraction] of the mask gave the shaft axis; a robust iterative
least-squares fit (2.5-SD clipping, three iterations) gave the distal physeal line; the Baumann
angle is [the acute angle between them / its complement — state convention matching the readers].
Angles were computed on isotropically scaled images (letterbox padding; no anisotropic scaling in
the measurement path). **Lateral.** Zero-shot SAM2 (sam2_hiera_large) segmented the humerus and
forearm; a ResNet-18 coordinate regressor localized the capitellum; the AHL was constructed from
the distal anterior cortex and assessed for capitellar bisection; cortical width was profiled
across [n] cross-sections.

### Module 4 — Cross-module Consistency Verification
(a) *Attention–anatomy consistency:* Grad-CAM activation was compared with the anatomic
segmentation by IoU, giving a per-case spatial consistency score (metric adapted from Saporta et
al ⚠; applied here as a cross-prong agreement test). (b) *Rule-level discordance:* classifier
output was compared with the anatomic measurement expected under the institutional pathway
[enumerate rules]. Thresholds, the IoU cut-point, weighting, and missing-measurement handling
were prespecified. Flagged cases were surfaced for review, not overridden.

### Out-of-distribution Filtering
A DRUE filter (shallow decoder from `layer1`, deep decoder from `layer4`, on a frozen backbone)
produced a per-image uncertainty score = mean |shallow − deep reconstruction|. Cases above the
95th percentile of the in-distribution score distribution [state calibration partition, disjoint
from test] were removed. Full-cohort and OOD-filtered performance are both reported.

### Model Comparison and Ablations
Evaluated: (a) representation learning (random vs ImageNet vs autoencoder initialization); (b)
preprocessing (marker removal; lateral alignment); (c) OOD filtering; (d) external-dataset
retraining; and (e) a flat four-class ResNet-18 baseline versus the cascade, on identical splits.

### Evaluation Metrics
Primary endpoint: **Grade IIB recall (sensitivity)**, chosen because costs at the surgical
boundary are asymmetric (a missed IIB is a surgical fracture managed in a cast; a false-positive
IIB is a theatre referral resolved by review), so the headline metric is one-sided. Secondary:
Grade III recall; per-node and end-to-end cascade performance; positive-class and weighted F1
(comparability only); weighted κ (with prevalence/bias indices or PABAK); AUROC and AUPRC at Exp2
and Exp4 (with prevalence baselines); the Exp4 precision–recall curve; calibration (intercept,
slope, Brier); and, for measurement, MAE/median AE, ICC(2,1), Bland–Altman, %within ±3°/±5°, and
Dice/IoU. Selective-prediction mechanisms are evaluated by coverage and selective risk against
random flagging. Overall accuracy and weighted F1 are not primary endpoints (prevalence-dependent
and symmetric, contradicting the clinical cost asymmetry). The unit of analysis is stated per
metric.

### Statistical Analysis
Proportions use Clopper–Pearson 95% CIs; other intervals use patient-clustered bootstrap. Paired
classifier comparisons use McNemar; paired AUC comparisons use DeLong. Headline metrics are 3-seed
mean ± SD. The expected 95% CI half-width on the primary endpoint was computed a priori (Riley et
al ⚠). Reporting follows CLAIM 2024, TRIPOD+AI, and STARD. [Software/library versions.]

### Code and Data Availability
Code at [repo]; model weights [hosting]. Patient radiographs are not publicly shared; [what is
available on request].

---

## PART E — Results *(mirrors Methods; real classifier numbers filled)*

### Patient Characteristics
Figure 2 outlines selection: [n] enrolled, [n] excluded ([reasons]), [n] analyzed, split
patient-level into training (n=[n]), validation (n=[n]), test (n=[n]). Grade distribution and
baseline characteristics are in Table 1. Reference-standard interobserver agreement was κ = [X].

### Node-conditional Classification *(post-alignment, DRUE-filtered; 3-seed mean ± SD; pooled recall 95% CI)*

| Node | Positive class | n₊ (test) | Config | Weighted F1 | **Positive-class recall** | Pooled recall 95% CI | Precision |
|---|---|---|---|---|---|---|---|
| Exp1 | Fracture | ~83 | LS | 0.878 ± 0.023 | **96.4% ± 1.0** | [93.2, 98.3] | 90.6% |
| Exp2 | Grade III | ~19 | Baseline | 0.835 ± 0.015 | **49.1% ± 5.0** | [35.6, 62.7] | 75.5% |
| Exp3 | Grade II | ~36 | LS+Dropout | 0.742 ± 0.040 | **81.5% ± 5.2** | [72.9, 88.3] | 74.5% |
| **Exp4** | **Grade IIB** | **~10** | LS+MixUp *(max-F1)* | 0.699 ± 0.084 | **33.3% ± 18.9** | **[17.3, 52.8]** | 52.8% |
| **Exp4** | **Grade IIB** | **~10** | Baseline *(max-recall)* | 0.686 ± 0.010 | **60.0% ± 14.1** | **[40.6, 77.3]** | 43.9% |

Two findings are load-bearing for the paper: (a) at the primary node, selecting the configuration
by weighted F1 (LS+MixUp) yields IIB recall of 33.3% ± 18.9, whereas a recall-oriented selection
(Baseline) yields 60.0% ± 14.1 at statistically indistinguishable F1 — model selection must be
recall-based; (b) the IIB confidence interval is wide by design at this denominator. The full
regularization factorial is in Appendix S2. *(End-to-end cascade recall — routing each case by
predicted rather than reference upstream labels — is reported in Table [n]; see Part F.)*

### Effect of Pipeline Quality Control and External Data *(weighted F1, single run)*

| Node | Baseline | +Alignment | +Alignment+OOD | +External dataset |
|---|---|---|---|---|
| Exp2 (Gr III vs I+II) | 0.82 | 0.83 | 0.88 | 0.89 |
| Exp3 (Gr I vs II) | 0.74 | 0.76 | 0.82 | 0.85 |
| Exp4 (Gr IIA vs IIB) | 0.72 | 0.72 | 0.76 | 0.78 |

Anatomic standardization and DRUE filtering improved weighted F1 at every node; external-dataset
retraining added further gains. *(Caveat, Part F: hold the test set constant across columns and
report the same progression in positive-class recall with seed variance, since the Exp4 gains are
within the seed SD.)*

### AP Segmentation, Baumann Angle, Lateral Pipeline
[YOLO mask mAP/Dice — reported separately from angle error.] [Baumann MAE, median, ICC(2,1),
Bland–Altman bias + limits, %within ±3°/±5°, failure rate, inter-reader ceiling — Fig 3.] [SAM2
Dice; capitellum median localization error; AHL bisection agreement; cortical-profile agreement —
Fig 4.]

### Cross-module Consistency and OOD
[IoU distribution by correct/incorrect; AUC of IoU for misclassification.] [Rule-discordance flag
rate; sensitivity for misclassifications; review burden; vs random at equal coverage.] At the
95th-percentile DRUE threshold, [X]% of test cases were removed; retained-subset vs full-cohort
performance [X] vs [X].

### Ablations, Failure Analysis, Downstream Management
[Representation-learning ablation; flat multiclass vs cascade.] [Grade III / IIB false negatives;
segmentation and alignment failures — Fig 5, Fig 6.] [Predicted vs recorded management.]

---

## PART F — Discussion

**¶1 Principal findings.** In this feasibility evaluation, node-conditional recall was 96% (fracture),
49% (Grade III), 82% (Grade II), and 33–60% (Grade IIB, depending on selection), with a wide IIB CI.
The contribution is an interpretable, auditable framework — not a high-accuracy autonomous grader.

**¶2 Hierarchical contribution.** The cascade mirrors the clinical pathway and produces a full
ordinal grade, where prior work is binary; report node-conditional and end-to-end results, and the
flat-multiclass comparison.

**¶3 AP measurement.** YOLO segmentation, PCA shaft-axis estimation, and robust distal-boundary
fitting convert an AP radiograph into an explicit, auditable Baumann angle; the contribution is the
integration, not the components.

**¶4 Lateral measurement.** AHL, capitellum localization, and cortical-width analysis as
complementary evidence at the Grade I/II and IIA/IIB boundaries.

**¶5 Consistency and safety.** Disagreement between learned and anatomic evidence is an interpretable
review trigger; surfacing discordance rather than resolving it silently is the design principle.
Address the Grad-CAM reliability critique directly and cite it.

**¶6 Relation to prior work.** Prior systems detect fractures (Tan 2025) or triage operatively
(*Injury* 2026); this framework adds an ordinal grade, anatomic measurement, and a consistency
check. [State cohort-provenance relationship if applicable.]

**¶7 Limitations.** Single-institution retrospective cohort of limited size; YOLO developed on 90
images; segmentation metrics do not guarantee angle accuracy; Baumann angle sensitivity to
positioning; physeal-line dependence on distal segmentation; reference-measurement observer
variability; wide CIs at the IIA/IIB node; the DRUE threshold is a single in-distribution
percentile; no prospective or external workflow validation.

**¶8 Conclusion.** In conclusion, an anatomically grounded framework integrating autoencoder-
pretrained hierarchical ResNet-18 classification with automated AP and lateral anatomic measurements
produced interpretable Gartland grade predictions with auditable anatomic evidence and surfaced
unreliable predictions for review. It is intended as interpretable decision support at the surgical
boundaries, not as an autonomous grader; prospective and external validation are required before
clinical use.

### Back matter
Author affiliations; received/revised/accepted dates; corresponding author; supplemental-material
line; funding; author contributions (CRediT); acknowledgments; disclosures (ICMJE). Ethics, data-
availability, code-availability, conflicts, generative-AI-use, and prior-presentation statements.

### Figures / Tables
Fig 1 framework · Fig 2 inclusion · Fig 3 Baumann pipeline + agreement · Fig 4 lateral pipeline ·
Fig 5 Grad-CAM + anatomy · Fig 6 consistency + failures. Table 1 cohort/grade distribution · Table 2
node-conditional + end-to-end classification · Table 3 measurement agreement.

---

## Supplement — Appendix S1 (Methods) / Appendix S2 (Results)
*(Every page: "Supplemental materials undergo peer review but are not copyedited…©RSNA, 2026, DOI".)*

**Appendix S1 — Supplementary Methods.** A.1 problem definition; A.2 architecture + equations
(standardization; autoencoder loss; per-node classifier; AP Baumann pipeline; lateral pipeline;
IoU/discordance) with Figs S1–S3; A.3 experimental settings (A.3.1 inclusion flow Fig S4; A.3.2
external/Pedia XHF dataset Table S2; A.3.3 acquisition + isotropic-scaling policy; A.3.4
implementation — optimizer, LRs, epochs, batch, seeds, versions Table S3; A.3.5 reference-standard
+ manual Baumann protocol; A.3.6 ablation configs; A.3.7 DRUE score equation + per-node 95th-
percentile thresholds).

**Appendix S2 — Supplementary Results.** B.1 full regularization factorial (all nodes, 3-seed
mean ± SD) Table S4; B.2 representation-learning ablation Table S5; B.3 cascade vs flat multiclass
Table S6; B.4 Exp4 PR-curve data + validation selection Fig S5/Table S7; B.5 Baumann agreement
(complete-case + intention-to-measure) Fig S6/Table S8; B.6 segmentation Dice/IoU Table S9; B.7
selective prediction (DRUE 95th-pct vs random) Fig S7; B.8 calibration Fig S8; B.9 computational
cost Table S10; B.10 precision statement; B.11 CLAIM/TRIPOD+AI checklists. Supplement-only
references renumbered after the main-text list.

---

## PART G — Pre-submission checklist *(all outstanding actions in one place)*

**Methodology fixes (from the training notebooks):**
1. Move final per-node **config selection to the validation partition** — the notebooks select by
   `test_acc.idxmax()` (test-set leakage). Config choice can be redone today from the CSVs' `val_acc`.
2. Re-report **positive-class recall** (primary) and positive-class F1; the ablation used *weighted* F1.
3. Export **per-case probabilities** (`case_id, node, true_label, prob_positive`) for validation and
   test → enables PR/AUROC/AUPRC, calibration, recall-target thresholds, and end-to-end cascade.
4. Compute **end-to-end cascade** IIB recall (route by predicted upstream labels); it will be below
   the node-conditional figure and is the number reviewers ask for first.
5. Hold the **test set constant** across the alignment/OOD/external ablation columns; the external
   column currently changes denominators. Report the progression in recall with seed variance.
6. Confirm the **patient-level split** (LAT_* / AP CSVs are built upstream; not verifiable from notebooks).
7. Reconcile the **slide vs CSV discrepancy** — single-seed slide confusion matrices do not match the
   ablation CSVs; regenerate all reported numbers from one consistent, final run.

**Data / analysis still needed:**
8. Baumann-angle agreement (MAE, ICC, Bland–Altman, %±3/±5°) and YOLO segmentation metrics.
9. Lateral pipeline (SAM2 Dice, capitellum localization, AHL bisection, cortical width).
10. Cross-module consistency (IoU AUC; discordance flag rate + sensitivity + review burden).
11. Agreement κ (model–expert, quadratic-weighted ordinal, expert–expert ceiling) — needs the second reader.
12. Calibration, decision-curve analysis, downstream management agreement.

**Executed-code facts to pull:** exact losses (autoencoder, classifiers, YOLO, capitellum), PCA
proximal-mask fraction, Exp4 threshold value, per-node DRUE 95th-percentile thresholds, isotropic-
scaling confirmation, software/library versions.

**Design / governance:** IRB number + consent; reference-standard reader count (drop κ endpoints if
single-reader); Pedia XHF role (training vs any test contribution) fixed before partitioning; target-
journal prior-submission (coursework) policy.

---

## PART H — References to verify *(⚠ = re-check against PubMed/DOI before use)*

1. ⚠ SCHF epidemiology / Gartland classification review.
2. ⚠ Gartland concordance among PEM/radiology/orthopedics. *Pediatr Radiol* 2024. DOI 10.1007/s00247-024-05935-3 / PMID 38693251.
3. ⚠ Modified Gartland reliability (PMID 11176349; PMC10779671, incl. Baumann/AHL).
4. ⚠ Tan MB et al. Paediatric elbow radiograph binomial classification. *Singapore Med J* 2025;66(4):208–214. PMID 40258236.
5. ⚠ DL pediatric elbow fracture detection — systematic review/meta-analysis. PMID 39976732.
6. ⚠ AI-assisted triage of pediatric SCHF (operative vs non-operative). *Injury* 2026. S0020138326001440.
7. ⚠ Radiomics SCHF diagnosis. *Medicine (Baltimore)* 2024;103(23). PMID 38847664 / PMC11155539.
8. ⚠ Rayan JC et al. Multiview pediatric elbow classification. *Radiol Artif Intell* 2019. DOI 10.1148/ryai.2019180015.
9. ⚠ Hierarchical proximal-femur cascade. *Eur J Radiol* 2020. S0720048X20305635.
10. ⚠ Zero-shot SAM-family bone segmentation. arXiv:2411.08629.
11. ⚠ Saporta et al. saliency-vs-segmentation localization. *Nat Mach Intell* 2022.
12. Method citations to insert where cited: Davis & Goadrich 2006; Saito & Rehmsmeier 2015; Byrt 1993;
    Koo & Li 2016; Vickers & Elkin 2006; Riley 2021 (PMID 34031906); CLAIM 2024 (DOI 10.1148/ryai.240300); TRIPOD+AI 2024.
