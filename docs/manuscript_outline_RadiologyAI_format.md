# Manuscript Outline — *Radiology: Artificial Intelligence* ORIGINAL RESEARCH format
### An Anatomically Grounded AI Framework for Automated Gartland Classification of Pediatric Supracondylar Humerus Fractures

> **How to use this document.** The structure, section order, and house style below mirror
> two *Radiology: AI* 2026 ORIGINAL RESEARCH articles (Han et al., ryai.250789; Zhang & Geng
> et al., ryai.250914) and the Han et al. supplement. Fill the `[bracketed]` placeholders
> with executed-code facts and final numbers. `⚠` marks a citation to re-verify (see
> `literature_landscape_and_outline.md`, Part D). Main text targets ~3000–3500 words; deep
> architecture detail, equations, acquisition parameters, and secondary analyses go to the
> Supplement (Appendix S1/S2), exactly as in the exemplars.

---

# FRONT MATTER

**ORIGINAL RESEARCH**

### Title
An Anatomically Grounded Deep Learning Framework for Automated Gartland Classification of
Pediatric Supracondylar Humerus Fractures Using Hierarchical Classification and Anatomic
Measurement

**Authors** *(Name, degree¹ — assign superscripts to affiliations; confirm order and
equal-contribution asterisks in writing before submission)*
[Author 1, degree*¹] • [Author 2, degree*¹] • [… ] • [Senior/corresponding author, degree ⁿ]
\* [X.X. and Y.Y. contributed equally to this work.]

*Author affiliations, funding, and conflicts of interest are listed at the end of this article.*
Radiology: Artificial Intelligence 2026; [vol]:[eLocator] • https://doi.org/[DOI] • Content codes: [ ]

---

### Structured Abstract
*(Four labeled paragraphs, matching the exemplars. Keep each tight; put the headline
numbers with 95% CIs in Results.)*

**Purpose:** To develop and evaluate an anatomically grounded artificial intelligence
framework for automated Gartland classification of pediatric supracondylar humerus fractures
(SCHF) from AP and lateral elbow radiographs.

**Materials and Methods:** This retrospective, IRB-approved study included [n] patients
([date range]) from [institution(s)]. Paired AP and lateral radiographs were graded by [two
blinded pediatric orthopedic surgeons with adjudication / single expert — state]. The
framework comprised four modules: (a) anatomic image standardization (marker removal,
lateral-view alignment); (b) an autoencoder-pretrained hierarchical ResNet-18 classifier
cascade (four binary nodes along the Gartland decision pathway); (c) view-specific anatomic
measurement — automated Baumann angle on AP views via YOLO segmentation and PCA shaft-axis
estimation, and anterior humeral line, capitellum localization, and cortical-width profiling
on lateral views via zero-shot SAM2; and (d) cross-module consistency verification. All
splits were performed at the patient level. The primary endpoint was Grade IIB recall, the
surgical decision boundary. [State statistical approach: Clopper–Pearson/Wilson CIs;
patient-clustered bootstrap; weighted κ; ICC and Bland–Altman for Baumann.]

**Results:** [n] patients ([n male]; mean age, [X] years ± [SD]) with [n] radiographs were
included. Grade IIB recall was [X]% (95% CI: [X], [X]); Grade III recall was [X]% (95% CI:
[X], [X]). Model–expert agreement at the IIA/IIB boundary was κ = [X] versus expert–expert κ
= [X]. End-to-end cascade [and node-conditional] performance are reported. Automated Baumann
angle showed a mean absolute error of [X]° (ICC, [X]; Bland–Altman bias, [X]°; 95% limits of
agreement, [X], [X]). [Segmentation Dice, ablation, and consistency-flag results summarized.]

**Conclusion:** In this feasibility evaluation, an anatomically grounded framework combining
hierarchical deep learning with automated AP and lateral anatomic measurements produced
interpretable Gartland grade predictions with auditable anatomic evidence and flagged
unreliable predictions for review. Grade IIB recall at the surgical boundary was [X]% (95% CI:
[X], [X]). The framework is positioned as interpretable decision support rather than autonomous
grading; prospective and external validation are required before clinical use.

*Supplemental material is available for this article.*
*© RSNA, 2026*

---

### Abbreviations
*(Alphabetical, as in exemplars.)*
AHL = anterior humeral line, AP = anteroposterior, AUC = area under the receiver operating
characteristic curve, CI = confidence interval, CNN = convolutional neural network, CRPP =
closed reduction and percutaneous pinning, DRUE = dual reconstruction uncertainty estimation,
Grad-CAM = gradient-weighted class activation mapping, ICC = intraclass correlation
coefficient, IoU = intersection over union, PCA = principal component analysis, SAM2 = Segment
Anything Model 2, SCHF = supracondylar humerus fracture, YOLO = you only look once

### Summary
An anatomically grounded AI framework integrating hierarchical deep learning with automated
AP- and lateral-view anatomic measurements provided interpretable decision support for Gartland
classification of pediatric supracondylar humerus fractures, evaluated at the surgically
decisive Grade IIB boundary as the primary endpoint.

### Key Points
*(Three bullets; every bullet must carry a number in the final manuscript — see exemplars.)*
- ■ An autoencoder-pretrained hierarchical ResNet-18 cascade classified Gartland grade from
  paired AP and lateral radiographs, achieving Grade IIB recall of [X]% (95% CI: [X], [X])
  at the surgical decision boundary.
- ■ View-specific measurement modules produced clinically recognized quantities — automated
  Baumann angle (mean absolute error, [X]°; ICC, [X]) on AP views and anterior humeral line
  and cortical-width analysis on lateral views — as auditable evidence for each prediction.
- ■ Cross-module disagreement between the learned classifier and independent anatomic
  measurements flagged [X]% of cases for review and detected misclassifications with [AUC/
  sensitivity X].

### Keywords
Pediatric, Elbow, Supracondylar Humerus Fracture, Gartland Classification, Deep Learning,
Convolutional Neural Networks, Hierarchical Classification, Segmentation, Anatomic Measurement,
Explainability, Baumann Angle

---

# INTRODUCTION
*(No section heading in the journal — this is the flowing text before "Materials and Methods."
Target 5–6 paragraphs, ending with the purpose statement. Citations are the verified set from
the landscape document.)*

**¶1 — Clinical burden.** Supracondylar humerus fractures are the most common operatively
managed elbow injury in children, accounting for a substantial share of pediatric fractures,
with a peak incidence at [age] years. [Cite epidemiology.]

**¶2 — The clinical decision the system supports.** The modified Gartland classification
directs management: type I is immobilized, whereas types IIB and III typically require closed
reduction and percutaneous pinning. The IIA/IIB and Grade III boundaries therefore separate
nonoperative from operative care. ⚠[Gartland classification review; management references]

**¶3 — Why interpretation is hard, precisely at the decision boundary.** Grading requires
integrating AP alignment and Baumann angle, the lateral anterior humeral line relationship,
cortical continuity, and displacement. Although aggregate interobserver agreement is
substantial (κ ≈ 0.77–0.85), it degrades at the type II sub-grades, with reported κ as low as
[0.26–0.43] for IIB — the very distinction that determines surgery. ⚠[concordance study, Ped
Radiol 2024; modified-Gartland reliability studies] Manual Baumann-angle measurement itself
carries observer variability (inter-reader ICC ≈ 0.74–0.77). ⚠

**¶4 — What AI has and has not done (the gap).** Prior deep learning on pediatric elbow
radiographs has largely addressed fracture detection or normal-versus-abnormal triage
⚠[Tan 2025, Singapore Med J; systematic review 2025], and one recent single-center study
reported binary operative-versus-nonoperative triage ⚠[Injury 2026]. Radiomics approaches to
SCHF remain limited to detection, require manual region delineation, and report modest
discrimination (AUC 0.65–0.72) ⚠[Medicine 2024]. To our knowledge, no verified prior framework
produces a full ordinal Gartland grade together with automated AP- and lateral-view anatomic
measurements and a cross-check between learned and anatomic evidence. *(Positioning note: do
NOT claim novelty of operative/nonoperative triage — that boundary has been reported; claim
ordinal grading + anatomic measurement + consistency verification.)*

**¶5 — Technical rationale.** A single convolutional network can learn global image patterns
but may also exploit acquisition markers or positioning shortcuts. We therefore combined
standardized input, radiograph-domain representation learning, a hierarchical classifier that
mirrors the clinical decision pathway, explicit anatomic measurement, and consistency
verification that surfaces disagreement rather than resolving it silently.

**¶6 — Purpose.** Therefore, this study aimed to develop and evaluate an anatomically grounded
artificial intelligence framework for automated Gartland classification using hierarchical
deep learning and complementary AP- and lateral-view anatomic measurement modules, and to
evaluate its performance with the surgically decisive Grade IIB boundary as the primary
endpoint.

---

# MATERIALS AND METHODS
*(Subheadings follow the exemplars. Keep architecture equations and acquisition detail in the
Supplement; name and summarize here.)*

### Patient Cohort
Retrospective study approved by [IRB name], protocol [no.], with [informed consent
waived/obtained]. Consecutive patients [date range] from [institution] with paired AP and
lateral elbow radiographs were [included/identified]. Inclusion: [age threshold; paired
views]. Exclusion: [missing view, poor quality, cast/splint obscuring anatomy, prior
fixation]. The patient inclusion process is shown in Figure 2 [and Fig S[n]]. [State
supplementary/auxiliary data source and its role — training vs external validation — as a
design choice.] Additional acquisition and preprocessing details are provided in Appendix S1.

### Reference Standard
[Two] blinded pediatric orthopedic surgeons ([X] and [Y] years of experience) independently
assigned the Gartland grade; discordance was adjudicated by a third senior surgeon. Manual
Baumann-angle measurements were obtained independently by [two] readers to validate the AP
measurement pipeline. [If single-reader: state so plainly and remove κ endpoints — see Open
Item.] Interobserver agreement is reported as the reference ceiling.

### Study Design and Partitioning
All splits were performed at the patient level; every image from a patient (AP and lateral
views, repeated examinations) remained in one partition. Partitioning preceded autoencoder
pretraining, YOLO segmentation training, capitellum-regressor training, classifier training,
threshold selection, and geometric-pipeline development. The test cohort was not used for
model, hyperparameter, checkpoint, or threshold selection. Autoencoder pretraining used
training-partition images only.

### AI Framework Overview
The framework comprised four modules (Fig 1): (a) anatomic image standardization; (b) an
autoencoder-pretrained hierarchical ResNet-18 classifier cascade; (c) view-specific anatomic
measurement; and (d) cross-module consistency verification. Full architectural specifications
and equations are provided in Appendix S1.

### Module 1: Image Standardization
Radiographic text markers (eg, L/R/SUPINE) were detected and removed before training and
inference to reduce shortcut learning; the identical procedure was applied to all partitions.
Lateral radiographs were standardized by Otsu thresholding, principal-component identification,
left–right orientation normalization by horizontal flipping, PCA-based long-axis rotation, and
elbow-region cropping (flipping precedes PCA rotation; anterior-edge determination is derived
from the standardized orientation). Detailed steps are in Appendix S1.

### Module 2: Autoencoder-pretrained Hierarchical ResNet-18 Classification
A convolutional autoencoder (ResNet-18 encoder; residual decoder discarded after pretraining)
was pretrained on training-partition pediatric elbow radiographs using [exact reconstruction
loss — L1/MSE/weighted]. Four binary classifiers were arranged along the Gartland decision
pathway (Table [1/Fig 1]): Exp1 (AP, fracture vs normal), Exp2 (AP, Grade III vs I+II), Exp3
(lateral, Grade II vs I), Exp4 (lateral, Grade IIB vs IIA). Each node was trained on the
subset defined by the reference standard. Two-stage transfer (Stage 1: encoder frozen,
head trained; Stage 2: encoder unfrozen, fine-tuned with reduced backbone learning rate
[ratio]) was used. The classification objective was [exact loss as executed — eg, two-logit
weighted cross-entropy] with class weights [state numerically], held constant across
regularization arms. Within each run, the epoch checkpoint was selected by validation accuracy.
The per-node operating configuration is selected on the **validation** partition using a
recall-oriented criterion (positive-class recall or positive-class F1 at the surgical boundary);
the Exp4 operating threshold is then selected on validation to meet a prespecified recall target,
with the full precision–recall operating curve reported. ⚠ Implementation note (fix before
submission): the development ablation selected the final per-node configuration by *test*
accuracy and reported *weighted* F1 — both must be re-based on the validation partition with
positive-class metrics, because test-based selection biases the reported performance. Grad-CAM
from `layer4[-1]` was used post hoc for attention assessment and as
input to §Module 4. Full training schedules and hyperparameters are in Appendix S1.

### Module 3: View-specific Anatomic Measurement
**AP — automated Baumann angle.** A YOLO instance-segmentation model [version/size] segmented
the distal humerus and shaft. The humeral component (largest connected component; forearm
excluded by [rule]) was isolated; PCA on the proximal [fraction] of the mask estimated the
shaft axis; a robust iterative least-squares fit (2.5-SD clipping, three iterations) estimated
the distal physeal line. The Baumann angle was defined as [the acute angle between the shaft
axis and the physeal line / its complement — state convention matching the reference readers].
Measurements were computed on images preserving the original pixel aspect ratio (isotropic
scaling with letterbox padding; no anisotropic scaling in the measurement pathway).
**Lateral — AHL, capitellum, cortical width.** Zero-shot SAM2 (sam2_hiera_large) segmented the
humerus and forearm; a ResNet-18 coordinate regressor localized the capitellum; the anterior
humeral line was constructed from the distal anterior cortex and assessed for capitellar
bisection; cortical width was profiled across [n] cross-sections. Details and equations are in
Appendix S1.

### Module 4: Cross-module Consistency Verification
Two mechanisms were applied. (a) *Attention–anatomy spatial consistency:* Grad-CAM activation
was compared with the anatomic segmentation mask by intersection-over-union, yielding a
per-case spatial consistency score (metric adapted from Saporta et al ⚠; applied here as a
cross-prong agreement test, not as a saliency benchmark). (b) *Rule-level discordance:*
classifier output was compared with anatomic measurement under the institutional decision
pathway [enumerate rules]. Discordance definitions, thresholds, the IoU cut-point, weighting,
and missing-measurement handling were prespecified before test-set evaluation. Flagged cases
were surfaced for review, not silently overridden.

### Out-of-distribution Filtering
A dual reconstruction uncertainty estimation (DRUE) filter was applied per node. Two
reconstruction decoders (shallow, from ResNet-18 `layer1`; deep, from `layer4`) attached to a
frozen classifier backbone produced a per-image uncertainty score defined as the mean absolute
difference between the shallow and deep reconstructions; higher scores indicate greater
out-of-distribution deviation. The rejection threshold for each node was set at the **95th
percentile of the in-distribution DRUE score distribution** [state calibration partition —
training or validation, and confirm it is disjoint from the test set]; images with a score
above this threshold were flagged as out-of-distribution and removed, retaining the lower 95%
of in-distribution cases by construction. Full-cohort and OOD-filtered performance are both
reported; filtered performance does not replace full-cohort performance. Score computation and
threshold derivation are detailed in Appendix S1.

### Model Comparison and Ablations
The framework was compared against [a flat four-class ResNet-18 baseline on the same splits]
and evaluated in ablations isolating (a) representation learning (random vs ImageNet vs
autoencoder initialization), (b) preprocessing (marker removal; lateral alignment), and (c)
module contribution (classifier alone; + AP Baumann; + lateral module; full framework with
verification). Detailed ablation configurations are in Appendix S1.

### Evaluation Metrics
The primary endpoint was Grade IIB recall (sensitivity), chosen because the misclassification
costs at the surgical boundary are asymmetric (a missed IIB is a surgical fracture managed in a
cast; a false-positive IIB is a theatre referral resolved by review), so the headline metric is
one-sided. Positive-class (Grade IIB) F1 is reported as a secondary metric for comparability;
weighted F1 (the metric used in the development ablation) is reported for comparability only and
is not a primary endpoint, as it is symmetric in the two error types and prevalence-dependent,
contradicting the clinical cost asymmetry. Overall accuracy is likewise not used as a primary
endpoint. ⚠ Model selection must be performed on the validation partition using positive-class
recall/F1 rather than test accuracy (see Supervised training). Secondary
endpoints included Grade III recall; per-node and end-to-end cascade performance; weighted κ
(with prevalence/bias indices
or PABAK); AUROC and AUPRC at Exp2 and Exp4 (with no-skill baselines); the Exp4 precision–
recall operating curve; calibration (intercept, slope, Brier); and, for measurement modules,
mean and median absolute error, ICC(2,1, absolute agreement), Bland–Altman bias and limits of
agreement, proportion within ±3°/±5°, and Dice/IoU for segmentation. Selective-prediction
mechanisms were evaluated by coverage and selective risk (risk–coverage curve) against random
flagging. The unit of analysis is stated for every metric.

### Statistical Analysis
Sample size and the expected 95% CI half-width on the primary endpoint were computed a priori
[Clopper–Pearson; cite Riley et al ⚠]. Proportions used [Clopper–Pearson/Wilson] intervals;
all other intervals used patient-clustered bootstrap resampling. Paired classifier comparisons
used the McNemar test; paired AUC comparisons used the DeLong test. Multiplicity was handled by
[correction / labeling secondary comparisons exploratory]. Headline metrics are reported as
mean ± SD across [≥3] seeds [or stated single-run]. Analyses used [Python version; R version];
P < .05 indicated significance. [Reporting standards: CLAIM 2024, TRIPOD+AI, STARD.]

### Code Availability
The code used for model development is available at [repository URL]; model weights are
[hosting arrangement].

---

# RESULTS
*(Mirror the Methods order; reference Tables and Figures inline as in exemplars.)*

### Patient Characteristics
Patient selection is outlined in Figure 2. Overall, [n] patients were [enrolled/identified];
[n] were excluded because of [reasons], leaving [n] for analysis, allocated to training
(n = [n]), validation (n = [n]), and test (n = [n]) sets. Baseline characteristics and grade
distribution are summarized in Table 1. [Reference-standard interobserver agreement: κ = [X].]

### Image-Standardization Performance
[X]% of images were successfully standardized; [X]% required manual review. [Alignment failure
criteria and rates.]

### Hierarchical Classification: Node-conditional and End-to-end
Table 2 reports per-node performance under the reference standard and end-to-end cascade
performance with error propagation. Grade IIB recall (primary endpoint) was [X]% (95% CI: [X],
[X]) node-conditional and [X]% (95% CI: [X], [X]) end-to-end. Grade III recall was [X]%. The
Exp4 precision–recall operating curve with the selected threshold is shown in Figure [n].

### AP Segmentation Performance
YOLO mask [mAP@0.5:0.95, precision, recall] on the [partition] set. *(Report separately from
angle accuracy — do not tabulate together.)*

### Automated Baumann Angle Agreement
Mean absolute error, [X]°; median, [X]°; ICC(2,1), [X]; Bland–Altman bias, [X]° (95% limits of
agreement: [X], [X]); within ±3°, [X]%; within ±5°, [X]%; measurement failure rate, [X]%.
Inter-reader manual agreement (ceiling) was [X] (Fig 3).

### Lateral Measurement Pipeline Performance
SAM2 segmentation Dice, [X]; capitellum localization median error, [X] px ([X] mm); AHL
bisection agreement, [X]; cortical-profile agreement, [X]; measurement failure rate, [X]%
(Fig 4).

### Cross-module Consistency
IoU distribution overall and stratified by correct/incorrect classification; discrimination
(AUC) of IoU for misclassification, [X]. Rule-level discordance flag rate, [X]%; sensitivity
for actual misclassifications, [X]; review burden, [X] cases per [period]; comparison against
random flagging at equal coverage (Fig [n]).

### Out-of-distribution Filtering
At the 95th-percentile DRUE threshold, [X]% of test cases were flagged as out-of-distribution
and removed. Classification performance on the retained (in-distribution) subset was [X] versus
[X] on the full cohort; both are reported so filtered performance does not stand in for
full-cohort performance. [Report coverage and selective risk; compare against random rejection
at equal coverage — see Appendix S2, B.7.]

### Ablation Results
Representation-learning ablation (Table [n]): autoencoder [X] vs ImageNet [X] vs random [X].
Flat multiclass baseline vs cascade: [X]. Preprocessing effects: [X].

### Failure Analysis
[Grade III false negatives; IIB false negatives; segmentation failures; Baumann outliers; AHL/
capitellum failures; residual marker attention; alignment errors] with paired figures (Fig 5,
Fig 6).

### Downstream Management Agreement
Predicted management (immobilization vs reduction+CRPP) versus recorded management: [X].

---

# DISCUSSION
*(Flowing paragraphs; the exemplars devote ¶1 to a numeric recap and close with a limitations
paragraph then "In conclusion.")*

**¶1 — Principal findings (numeric).** We developed an anatomically grounded framework that
[recap: IIB recall X% (95% CI wide, given the small IIB denominator), Grade III recall X%,
IIA/IIB κ = X vs expert–expert κ = X, Baumann MAE X°]. We report these as a feasibility result:
the confidence interval on the primary endpoint is wide by design at this cohort size, and the
contribution is an interpretable, auditable framework rather than a high-accuracy autonomous
grader.

**¶2 — Hierarchical contribution.** The cascade mirrors the clinical decision pathway; [node-
conditional vs end-to-end interpretation; comparison with a flat multiclass baseline].

**¶3 — AP measurement contribution.** YOLO segmentation, PCA shaft-axis estimation, and robust
distal-boundary fitting convert an AP radiograph into an explicit, auditable Baumann angle; the
contribution is the integration into an interpretable Gartland framework, not the individual
components.

**¶4 — Lateral measurement contribution.** AHL, capitellum localization, and cortical-width
analysis as complementary evidence at the Grade I/II and IIA/IIB boundaries.

**¶5 — Consistency and safety.** Disagreement between learned and anatomic evidence provides an
interpretable review trigger; surfacing discordance rather than resolving it silently is the
design principle. [Address Grad-CAM reliability critique directly and cite it.]

**¶6 — Relation to prior work.** Prior systems detect fractures ⚠[Tan 2025] or perform binary
operative/nonoperative triage ⚠[Injury 2026]; this framework additionally produces an ordinal
grade, anatomic measurements, and a consistency check. [If cohort overlaps a prior study, state
the relationship explicitly.]

**¶7 — Limitations.** This study had several limitations. [Single-institution retrospective
cohort of limited size; YOLO developed on 90 images; segmentation metrics do not guarantee
angle accuracy; Baumann angle sensitivity to positioning and rotation; physeal-line dependence
on distal segmentation; reference-measurement observer variability; wide CIs at the IIA/IIB
node; the DRUE out-of-distribution threshold was set at a single in-distribution percentile
(95th) rather than tuned against a labeled OOD set [and — if applicable — the calibration
partition's disjointness from the test set]; no prospective or external workflow validation.]

**¶8 — Conclusion.** In conclusion, an anatomically grounded framework integrating autoencoder-
pretrained hierarchical ResNet-18 classification with automated AP and lateral anatomic
measurements produced interpretable Gartland grade predictions with auditable anatomic evidence
and surfaced unreliable predictions for review. The framework is intended as interpretable
decision support at the surgical decision boundaries, not as an autonomous grader; prospective
and external validation are required before clinical use.

---

# BACK MATTER

**Author affiliations:** [numbered list with full department/institution addresses]
**Received** [date]; **revision requested** [date]; **revision received** [date]; **accepted** [date].
**Address correspondence to:** [initials] (email: [ ]).
**Supplemental material:** Supplemental material is available at Radiology: Artificial Intelligence online.
**Funding:** [grants].
**Author contributions:** [CRediT taxonomy per author — Conceptualization, Data curation, …]
**Acknowledgments:** [computing/clinical support].
**Disclosures of conflicts of interest:** [per author; ICMJE forms provided as supplemental materials].

### References
*(Vancouver style, numbered in citation order. Starter set from the landscape review — verify
every ⚠ against PubMed/DOI before submission. Method-citations from the v2 outline —
Saporta 2022, Davis & Goadrich 2006, Byrt 1993, Koo & Li 2016, Vickers & Elkin 2006, Riley
2021, CLAIM, TRIPOD+AI — inserted where cited above.)*
1. ⚠ [SCHF epidemiology / Gartland classification review]
2. ⚠ Gartland classification concordance among PEM physicians, radiologists, and orthopedic surgeons. Pediatr Radiol 2024. DOI 10.1007/s00247-024-05935-3 / PMID 38693251.
3. ⚠ Modified Gartland classification reliability. [PMID 11176349; PMC10779671]
4. ⚠ Tan MB, et al. Use of deep learning model for paediatric elbow radiograph binomial classification. Singapore Med J 2025;66(4):208–214. PMID 40258236.
5. ⚠ Impact of deep learning on pediatric elbow fracture detection: a systematic review and meta-analysis. PMID 39976732 (2025).
6. ⚠ Artificial Intelligence–Assisted Triage of Pediatric SCHF in Emergency Departments: A Single-Centre Validation Study. Injury 2026. [S0020138326001440]
7. ⚠ Automatic diagnosis of pediatric supracondylar humerus fractures using radiomics-based machine learning. Medicine (Baltimore) 2024;103(23). PMID 38847664 / PMC11155539.
8. ⚠ Rayan JC, et al. Binomial classification of pediatric elbow fractures using a deep learning multiview approach. Radiol Artif Intell 2019. DOI 10.1148/ryai.2019180015.
9. ⚠ Hierarchical fracture classification of proximal femur X-ray images using a multistage deep learning approach. Eur J Radiol 2020. [S0720048X20305635]
10. ⚠ Zero-shot capability of SAM-family models for bone segmentation. arXiv:2411.08629.
11. Saporta A, et al. [Grad-CAM/segmentation IoU localization]. Nat Mach Intell 2022. ⚠
12. [Additional method citations as inserted]

---
---

# SUPPLEMENTAL MATERIAL
*(Every page carries: "Supplemental materials undergo peer review but are not copyedited, and
may include typographical errors. ©RSNA, 2026, DOI." Structure follows Han et al.: Appendix S1
= Supplementary Methods; Appendix S2 = Supplementary Results; supplement-only references
renumbered to continue from the main-text list.)*

## Appendix S1. Supplementary Methods

### A.1. Problem Definition and Notation
Formal statement of the classification and measurement tasks; notation for AP/lateral inputs,
Gartland grade label space {Normal, I, IIA, IIB, III}, cascade routing, and the measurement
outputs (Baumann angle, AHL bisection, cortical-width profile).

### A.2. Framework Architecture (with equations and Figures S1–S3)
- **A.2.1. Image standardization pipeline.** Marker-detection/removal method; the six-step
  lateral standardization with the flip-before-PCA ordering; anterior-edge derivation. *(Fig S1:
  standardization stages.)*
- **A.2.2. Autoencoder pretraining.** Encoder/decoder architecture; exact reconstruction loss
  (equation); pretraining corpus (training partition only; AP/lateral pooled or not); epochs;
  final reconstruction loss. *(Fig S2: reconstruction examples.)*
- **A.2.3. Hierarchical classifier.** Per-node architecture; two-stage transfer schedule; exact
  classification loss (equation) and class weights; threshold-selection procedure. *(Table S1:
  per-node training configuration.)*
- **A.2.4. AP Baumann pipeline.** YOLO configuration; connected-component/forearm-exclusion
  rule; PCA on proximal-mask fraction (equation); robust iterative line fit (equations, 2.5-SD
  clipping, 3 iterations); angle definition and convention (equation). *(Fig S3: pipeline
  stages.)*
- **A.2.5. Lateral pipeline.** SAM2 prompting; capitellum coordinate-regression loss (equation);
  AHL construction; cortical-width sampling convention.
- **A.2.6. Cross-module verification.** IoU computation between Grad-CAM and mask (equation);
  rule-discordance definitions and thresholds.

### A.3. Experimental Settings
- **A.3.1. Patient inclusion.** Full flow diagram. *(Fig S4: inclusion flow.)*
- **A.3.2. Supplementary / auxiliary dataset.** Source, size, and role (training vs external);
  explicit statement it did/did not touch the test set. *(Table S2: auxiliary cohort
  characteristics.)*
- **A.3.3. Image acquisition and preprocessing.** Radiograph sources, isotropic-scaling policy
  for the measurement pathway, letterbox padding, normalization.
- **A.3.4. Implementation details.** Optimizer, learning rates (stage 1/stage 2), schedule,
  batch size, epochs, early stopping, checkpoint criterion, hardware, software/library versions,
  seeds. *(Table S3: hyperparameters per component.)*
- **A.3.5. Reference standard and manual measurement protocol.** Reader experience, blinding,
  adjudication; manual Baumann-angle protocol and inter-reader setup.
- **A.3.6. Ablation and comparison configurations.** Exact definitions of each ablation arm and
  the flat multiclass baseline.
- **A.3.7. DRUE out-of-distribution filter.** Decoder architecture (shallow from `layer1`, deep
  from `layer4`); uncertainty score = mean |shallow_recon − deep_recon| (equation); threshold
  set at the 95th percentile of the in-distribution DRUE scores on the [calibration partition];
  confirmation the calibration partition is disjoint from the test set. *(Table S[n]: per-node
  95th-percentile threshold and resulting coverage.)*

## Appendix S2. Supplementary Results

- **B.1. Full regularization factorial.** Label smoothing × MixUp × dropout at Exp4 (class
  weighting fixed). *(Table S4.)*
- **B.2. Representation-learning ablation (full).** Random / ImageNet / autoencoder across all
  nodes with multi-seed mean ± SD. *(Table S5.)*
- **B.3. Cascade vs flat multiclass baseline.** Node-conditional and end-to-end. *(Table S6.)*
- **B.4. Threshold / operating-point analysis.** Exp4 precision–recall curve data; validation
  selection criterion. *(Fig S5; Table S7.)*
- **B.5. Baumann-angle agreement, extended.** Complete-case and intention-to-measure Bland–
  Altman; proportional-bias check; MAE–median gap. *(Fig S6; Table S8.)*
- **B.6. Segmentation quality.** YOLO and SAM2 Dice/IoU against manual masks by region.
  *(Table S9.)*
- **B.7. Selective prediction.** Risk–coverage curves for cross-module flagging and DRUE
  (95th-percentile threshold) vs random rejection at equal coverage; coverage, selective risk,
  and errors captured at the operating threshold. *(Fig S7.)*
- **B.8. Calibration.** Calibration curves, intercept/slope, Brier decomposition. *(Fig S8.)*
- **B.9. Computational cost.** Params, GPU memory, FLOPs, training/inference time per component
  (as in Han et al. Table S8). *(Table S10.)*
- **B.10. Precision statement.** A-priori expected CI half-width on IIB recall at the observed
  denominator.
- **B.11. Reporting checklists.** Completed CLAIM 2024 and TRIPOD+AI checklists.

### Supplement-only References
[Renumbered to continue from the last main-text reference number, per RSNA convention — eg,
SAM2 checkpoint, YOLO/Ultralytics version, Otsu, PCA references not cited in the main text.]

---

## Mapping notes (delete before submission)
- **What moved to Supplement vs main text:** following the exemplars, the main text *names* each
  loss and module and reports headline numbers; every equation, architecture diagram, acquisition
  parameter, hyperparameter table, and secondary ablation lives in Appendix S1/S2. This keeps the
  main text near the journal's length norm.
- **Figures (main text):** Fig 1 framework overview (like Han Fig 1 / Zhang Fig 1); Fig 2 patient
  inclusion (like Zhang Fig 2); Fig 3 Baumann pipeline + agreement; Fig 4 lateral pipeline; Fig 5
  Grad-CAM + anatomy (correct/incorrect); Fig 6 consistency + failures.
- **Tables (main text):** Table 1 baseline characteristics/grade distribution; Table 2 node-
  conditional + end-to-end classification with 95% CIs; Table 3 measurement agreement. Additional
  tables → Supplement (S1–S10).
- **Still blocked on executed-code facts:** exact losses (autoencoder, classifiers, YOLO,
  capitellum), PCA proximal-mask fraction, Exp4 threshold criterion, isotropic-scaling status,
  and all final metrics. Pull these from the pipeline code before filling. (DRUE OOD filtering
  was performed with a 95th-percentile in-distribution threshold; the committed pipeline config
  ships thresholds as `None`, so record the exact per-node threshold values from your analysis.)
- **Pre-submission methodology fixes (from the training notebooks):**
  (1) the ablation reported *weighted* F1 (`average='weighted'`) — re-report positive-class
  recall (primary) and positive-class F1;
  (2) the final per-node model was selected by *test* accuracy (`test_acc.idxmax()`) — re-select
  on the validation partition by a recall-oriented criterion (this is test-set leakage as run);
  (3) per-case probabilities are not saved (eval uses `argmax`) — export `case_id, node,
  true_label, prob_positive` for val and test to enable PR/AUROC/AUPRC, calibration,
  recall-target thresholds, and the end-to-end cascade;
  (4) confirm the train/val/test split (read from upstream `LAT_*` / AP CSVs) is patient-level
  with no patient spanning partitions — not verifiable from the notebooks alone.
