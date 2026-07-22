# Literature Landscape, Technology Analysis, and Journal-Ready Outline
### Automated Gartland Classification of Pediatric Supracondylar Humerus Fractures

> **Status of this document.** Prepared as background research for the manuscript. All
> citations below were retrieved through web search summaries, **not** from full text
> (journal full-text hosts were unreachable from the working environment). Every
> author list, metric, and identifier marked with ⚠ must be re-verified against the
> actual paper (PubMed/DOI) before it enters the manuscript. This is exactly the
> citation-hygiene discipline your own outline (§5.16, Appendix item 18) requires.

---

## Part A — What already exists

The published work clusters into four bands, moving from "furthest from your claim" to
"closest to your claim." The important finding is at the boundary between bands 2 and 3.

### Band 1 — Fracture *detection* / normal-vs-abnormal triage (the bulk of the field)

This is where almost all pediatric-elbow AI lives. The task is binary: is there a
fracture / abnormality or not. None of these produce a Gartland grade or an anatomic
measurement.

| Study | Task | Data | Method | Headline result |
|---|---|---|---|---|
| **Tan MB et al., *Singapore Med J* 2025;66(4):208–214** (PMID 40258236) — your institutional comparator | Lateral radiograph: normal vs abnormal | 1,314 pediatric lateral elbow radiographs, mean age 8.2 y | EfficientNet-B1 | Acc 80.4% (95% CI 71.8–87.3), AUROC 0.872 ⚠ |
| **Rayan et al., "Binomial … Multiview … Emulating Radiologist Decision Making," *Radiology: AI* 2019** (DOI 10.1148/ryai.2019180015) | Binary abnormality, multi-view | Pediatric elbow | Multi-view CNN that mimics the radiologist's view-integration | Precedent for **multi-view** design ⚠ |
| **"Elbow trauma in children: development and evaluation of radiological AI models"** (PMC11265386) | Detection / triage | Pediatric elbow | CNN | Model 1 ext. test acc 82.5%, AUROC 0.916; improved reader sensitivity ⚠ |
| **"Assessing deep learning AI support for detecting elbow fractures in the pediatric ED," *Eur J Radiol* 2025** (S0720048X25005844) | Detection support in ED | Pediatric ED elbow | Commercial/DL aid | Reader-assist study ⚠ |
| **Systematic review & meta-analysis, PMID 39976732 (2025)** | Evidence synthesis | Pediatric elbow fracture detection | — | Confirms the field is dominated by **detection**, not grading ⚠ |

**Takeaway:** the entire first band answers "is it broken," a question your framework
treats as solved by node Exp1 alone. Cite this band to establish that detection is
mature and that the open problem is *grading and measurement*.

### Band 2 — Radiomics diagnosis of supracondylar fracture

| Study | Task | Data | Method | Result |
|---|---|---|---|---|
| **"Automatic diagnosis of pediatric supracondylar humerus fractures using radiomics-based machine learning," *Medicine* (Baltimore) 2024;103(23)** (PMID 38847664 / PMC11155539) | Fracture vs normal (still detection, not Gartland grade) | 411 fracture + 190 normal ROI samples; hand-delineated humeral-condyle contours on AP **and** lateral | 1,561 pyradiomics features → t-test + LASSO → logistic regression | **AUC only 0.65 (AP), 0.72 (lateral)** ⚠ |

**Takeaway:** this is the closest thing to "structured analysis of the supracondylar
region," yet (a) it is still binary detection, not grading; (b) it needs **manual ROI
delineation**, so it is not automated end-to-end; and (c) its AUCs are weak. This is a
useful contrast: hand-crafted radiomics on this problem underperforms, which motivates a
learned + anatomically-grounded approach.

### Band 3 — Operative-vs-non-operative severity triage ⭐ **your closest prior art**

| Study | Task | Data | Method | Result |
|---|---|---|---|---|
| **"Artificial Intelligence–Assisted Triage of Pediatric Supracondylar Humerus Fractures in Emergency Departments: A Single-Centre Validation Study," *Injury* (Elsevier, 2026)** (article S0020138326001440; ResearchGate 402848162) | **Binary: operative (Gartland IIB, III, flexion) vs non-operative (normal, I, IIA)** | **1,811 AP + lateral radiographs, single tertiary pediatric orthopaedic centre, 2010–2017** | Deep learning classifier | **Acc 77% (71–82), AUROC 0.84 (0.78–0.89), sensitivity 69% (58–78), specificity 82% (74–88)** ⚠ |

**Why this one matters most.** It draws its decision boundary at *exactly the same place
your paper claims as the clinically decisive line* — the operative/non-operative split
that runs through the IIA/IIB boundary. Its cohort (single tertiary pediatric orthopaedic
centre, 2010–2017, ~1,800 paired AP/lateral) is strikingly similar in shape to yours, and
its institutional lineage (alongside the Tan et al. *Singapore Med J* paper) may be the
same or a sibling group. **You cannot present the operative/non-operative boundary as an
open problem — this paper has already reported a binary triage model for it.**

What it does **not** do, and where your contribution still stands:
- It produces a **single binary label**, not a full ordinal Gartland grade (I / IIA / IIB / III).
- It produces **no anatomic measurement** — no Baumann angle, no anterior humeral line, no cortical profile.
- It offers **no interpretable evidence** tying the prediction to the anatomy a surgeon reasons over.
- It has **no cross-check** between a learned classifier and an independent geometric measurement.

Its own reported sensitivity (69%) also underlines your primary-endpoint framing: even a
purpose-built binary triage model misses ~31% of operative cases, so recall at this
boundary is genuinely the hard, unsolved metric.

### Band 4 — Adjacent methods you are assembling (precedent exists, none on the pediatric elbow)

These establish that each *component* is off-the-shelf (Appendix Tier-1 objection #3), and
that your novelty must be the integration + the cross-module check, not the parts.

- **Hierarchical / cascade CNN grading:** demonstrated for proximal-femur AO/OTA fractures
  ("Hierarchical fracture classification of proximal femur X-ray images using a multistage
  deep learning approach," *Eur J Radiol* 2020, S0720048X20305635 ⚠). Precedent for a
  cascade that mirrors a clinical classification tree — **but not for Gartland**.
- **Automated anatomic-angle measurement by landmark/line fitting:** demonstrated for the
  calcaneus (Böhler's angle, Gissane's angle) via rotation-invariant landmark regression
  (arXiv:1912.04536 ⚠). Precedent for "segment → fit line → compute angle" — **but not for
  the Baumann angle**.
- **Zero-shot SAM / SAM2 bone segmentation:** demonstrated on CT and MRI bone
  (SAM-family zero-shot bone CT, arXiv:2411.08629; SegmentAnyBone on MRI ⚠). Establishes
  SAM2 zero-shot bone is *plausible* but **explicitly unvalidated on pediatric elbow
  radiographs** — which is why your §5.8 must report Dice/IoU against manual masks.
- **Grad-CAM interpretability in fracture CNNs:** routine, and routinely criticized as
  unreliable in radiology (your Appendix Tier-2 #5). Precedent, plus the reason you must
  frame Grad-CAM as one signal in an agreement test, not as an explanation.

### Cross-cutting: the clinical gap your Introduction should lean on

Interobserver reliability of Gartland grading is *good in aggregate but breaks down exactly
at your boundary* — the perfect motivating gap:

- Radiologist–orthopaedic agreement overall: **κ ≈ 0.85** (90% raw agreement) — Gartland
  concordance study, *Pediatric Radiology* 2024 (DOI 10.1007/s00247-024-05935-3, PMID
  38693251) ⚠.
- Tri-specialty (ED / radiology / orthopaedics) consensus: **κ ≈ 0.77** ⚠.
- **But at the type II sub-grades the agreement collapses: reported κ as low as 0.26–0.43
  for IIB** in the modified-Gartland reliability literature (e.g. reliability studies incl.
  PMID 11176349; PMC10779671, which also covers Baumann angle and AHL) ⚠.
- Baumann-angle inter-observer ICC ≈ 0.74–0.77 ⚠ — i.e. even the *manual* measurement has a
  real error floor, which is your ceiling for the automated pipeline (§5.7.7, §5.12.9).

This is the strongest, most honest framing of your gap: **the decision that matters most
(IIA vs IIB) is precisely the one humans agree on least, and no existing AI produces the
anatomic measurements clinicians use to resolve it.**

---

## Part B — Technology inventory (field vs. this project)

| Capability | State of the field | This project |
|---|---|---|
| Fracture detection (normal vs abnormal) | **Mature** — multiple CNNs, AUROC 0.87–0.92 | Exp1 node (not the contribution) |
| Operative vs non-operative triage | **Demonstrated once** (Injury 2026, AUROC 0.84, sens 69%) as a **black-box binary** | Full ordinal cascade + interpretable evidence + recall-tuned Exp4 |
| Full ordinal Gartland grade (I/IIA/IIB/III) | **Not demonstrated** in verified literature | Hierarchical Exp1→Exp2→Exp3→Exp4 cascade |
| Automated Baumann angle | **Not demonstrated** on elbow (only Cobb/foot/calcaneus angles elsewhere) | YOLO seg → PCA shaft axis → robust physeal-line fit |
| Automated anterior humeral line / capitellum | **Not demonstrated** as an automated measurement | SAM2 + capitellum regressor + AHL bisection |
| Cortical-width profiling for IIA/IIB | **Not demonstrated** | Lateral cortical-width module |
| Learned-vs-geometric cross-check | **Not demonstrated** in this domain | Module 4 (Grad-CAM↔mask IoU + rule discordance) |
| Radiograph-domain self-supervised pretraining | Common in general MSK; **not shown to matter on a cohort this small** | Autoencoder-pretrained ResNet-18 (ablation §5.11A must justify it) |
| Zero-shot foundation segmentation (SAM2) | Validated on CT/MRI bone; **unvalidated on pediatric elbow x-ray** | Used zero-shot; must report Dice/IoU |

**Reading of the table:** three rows are genuinely empty in the verified literature —
(1) full ordinal Gartland grading, (2) automated Baumann angle on the elbow, (3) the
learned-vs-anatomic consistency check. Those three are the defensible core of the paper.
Everything else is assembly, and the manuscript should say so plainly.

---

## Part C — Journal-ready outline (literature-grounded)

This restructures your v2 working outline into the shape a clinical-AI journal (Radiology:
AI, or a pediatric-orthopaedics venue) expects, with the related-work grounding wired in.
Method-level detail already nailed in your v2 (§5.5–5.18) is referenced, not repeated.

### Title
*An Anatomically Grounded AI Framework for Automated Gartland Classification of Pediatric
Supracondylar Humerus Fractures* (keep; it names the two real contributions — Gartland
grading + anatomic grounding).

### 1. Abstract (structured: Purpose / Materials & Methods / Results / Conclusion)
Lead the Results with the **primary endpoint (IIB recall, with 95% CI)** and the
**expert–expert κ ceiling**, then the Baumann-angle agreement. Conclusion: interpretable
grading with surgical-grade recall demonstrated; no claim of clinical readiness.

### 2. Introduction (six paragraphs, now anchored to real citations)
1. **Clinical burden** — SCHF incidence, age, operative importance.
2. **The decision** — Gartland grade drives cast vs reduction+CRPP; frame the **IIA/IIB and
   Grade III boundaries** as the target.
3. **Why it is hard for humans** — aggregate κ is high (~0.77–0.85) **but collapses at type
   II sub-grades (κ 0.26–0.43 for IIB)**; Baumann/AHL carry their own observer variability.
   *[cite the reliability + concordance studies]*
4. **What AI has and has not done** — detection is mature (Tan 2025; meta-analysis PMID
   39976732); severity triage has been shown once as a **black-box binary** (Injury 2026);
   radiomics grading is weak and needs manual ROIs (Medicine 2024, AUC 0.65–0.72). **State
   the gap precisely:** no verified system produces a full ordinal Gartland grade *together
   with* automated AP/lateral anatomic measurements and a learned-vs-anatomic consistency
   check. *(This wording survives the Injury 2026 paper — see Part E.)*
5. **Technical rationale** — a single CNN can shortcut on markers/positioning; hence
   standardization + domain pretraining + hierarchical grading + explicit measurement +
   verification.
6. **Purpose statement** — as in your v2 §4¶6.

### 3. Materials and Methods
- **3.1 Cohort & reference standard** — institution, IRB, inclusion/exclusion, paired
  AP+lateral, patient-level splits; **two blinded readers + adjudicator** (or restate as
  single-reader and drop κ — Open Item #3).
- **3.2 Framework overview** (Figure 1) — four modules.
- **3.3 Module 1 — anatomic standardization** (marker removal; 6-step lateral alignment;
  flip-before-PCA).
- **3.4 Module 2 — autoencoder-pretrained hierarchical ResNet-18** (Exp1–Exp4; node-
  conditional **and** end-to-end reporting; exact losses/thresholds from code).
- **3.5 Module 3A — automated Baumann angle** (isotropic-scaling precondition; YOLO seg;
  PCA on proximal-mask fraction; robust physeal-line fit; angle convention).
- **3.6 Module 3B — lateral measurement** (SAM2; capitellum regressor; AHL; cortical width).
- **3.7 Module 4 — cross-module verification** (Grad-CAM↔mask IoU; rule discordance;
  prespecified thresholds).
- **3.8 OOD filtering (DRUE)** — performed; cases above the 95th-percentile in-distribution DRUE
  score were removed. Report full-cohort and OOD-filtered performance (both, never one for the other).
- **3.9 Ablations** — representation-learning (random/ImageNet/autoencoder); preprocessing;
  **flat multiclass baseline** (Appendix Tier-2 #6); component ablation.
- **3.10 Endpoints & statistics** — primary IIB recall; Clopper–Pearson/Wilson CIs; patient-
  clustered bootstrap; κ (weighted + PABAK); ICC + Bland–Altman for Baumann; Dice/IoU for
  segmentation; risk–coverage for flagging; multi-seed variance; precision statement.
- **3.11 Reporting standards** — CLAIM 2024, TRIPOD+AI, STARD.

### 4. Results (mirror Methods order)
Cohort → standardization → **classification: node-conditional AND end-to-end** → YOLO seg →
**Baumann agreement (separate from mAP)** → lateral pipeline → cross-module consistency →
ablations → OOD filtering (95th-percentile DRUE; full-cohort vs filtered) → failure analysis →
downstream management agreement.

### 5. Discussion
Main findings (numeric) → hierarchical contribution → AP measurement contribution → lateral
measurement contribution → **consistency/safety as the methodological novelty** →
positioning **relative to Injury 2026 and Tan 2025** (you grade + measure + verify; they
triage) → limitations → conclusion.

### Figures
1 framework · 2 inclusion diagram · 3 Baumann pipeline · 4 lateral pipeline · 5 Grad-CAM +
anatomy (correct/incorrect) · 6 cross-module consistency & failures.

---

## Part D — Reference list with verification status

Re-verify each ⚠ against PubMed/DOI before use. Ordered by relevance to the paper.

1. ⚠ *Artificial Intelligence–Assisted Triage of Pediatric Supracondylar Humerus Fractures
   in Emergency Departments: A Single-Centre Validation Study.* **Injury** (Elsevier), 2026.
   Article S0020138326001440. **[Closest prior art — must cite and distinguish.]**
2. ✔/⚠ Tan MB, et al. *Use of deep learning model for paediatric elbow radiograph binomial
   classification.* **Singapore Med J** 2025;66(4):208–214. **PMID 40258236.** (Your stated
   institutional comparator — verify author list & metrics.)
3. ⚠ *Automatic diagnosis of pediatric supracondylar humerus fractures using radiomics-based
   machine learning.* **Medicine (Baltimore)** 2024;103(23). **PMID 38847664 / PMC11155539.**
4. ⚠ *Impact of deep learning on pediatric elbow fracture detection: a systematic review and
   meta-analysis.* **PMID 39976732** (2025).
5. ⚠ Rayan JC, et al. *Binomial Classification of Pediatric Elbow Fractures Using a Deep
   Learning Multiview Approach Emulating Radiologist Decision Making.* **Radiology: AI** 2019.
   DOI 10.1148/ryai.2019180015.
6. ⚠ *Elbow trauma in children: development and evaluation of radiological AI models.*
   **PMC11265386** (2023–24).
7. ⚠ *Assessing deep learning AI support for detecting elbow fractures in the pediatric ED.*
   **Eur J Radiol** 2025. Article S0720048X25005844.
8. ⚠ *Gartland classification concordance … among PEM physicians, radiologists, and orthopedic
   surgeons.* **Pediatric Radiology** 2024. DOI 10.1007/s00247-024-05935-3 / **PMID 38693251.**
9. ⚠ *The Gartland classification for expediting SCHF triage … structured reporting.*
   **Clinical Imaging** 2024. **PMID 38520814.**
10. ⚠ *Evaluation of Gartland Classification, Baumann Angle and Anterior Humeral Line …
    Inter/Intra-Observer Reliability Study.* **PMC10779671.**
11. ⚠ *Reliability of a modified Gartland classification of supracondylar humerus fractures.*
    **PMID 11176349** (2001).
12. ⚠ *Hierarchical fracture classification of proximal femur X-ray images using a multistage
    deep learning approach.* **Eur J Radiol** 2020. Article S0720048X20305635. (Cascade precedent.)
13. ⚠ *Automatic Analysis System of Calcaneus Radiograph: Rotation-Invariant Landmark
    Detection for Calcaneal Angle Measurement …* **arXiv:1912.04536.** (Angle-measurement precedent.)
14. ⚠ *Zero-shot capability of SAM-family models for bone segmentation in CT scans.*
    **arXiv:2411.08629.** (SAM2 zero-shot bone precedent.)
15. ⚠ *Artificial Intelligence in Pediatric Orthopedics: A Comprehensive Review.* **Medicina**
    (MDPI) 2025;61(6):954. (Review to situate the field.)

*Methodological citations already in your v2 (Saporta 2022; Davis & Goadrich 2006; Saito &
Rehmsmeier 2015; Byrt 1993; Koo & Li 2016; Vickers & Elkin 2006; Riley 2021; CLAIM/TRIPOD+AI)
are retained and were not re-checked here — verify separately.*

---

## Part E — Strategic implications (read this before you touch the manuscript)

1. **The Injury 2026 paper changes your positioning, not your novelty.** Your defensible
   gap statement becomes: *"Prior work performs fracture detection (Tan 2025; meta-analysis
   2025) or binary operative/non-operative triage (Injury 2026); no verified system produces
   a full ordinal Gartland grade combined with automated AP and lateral anatomic measurements
   and a learned-vs-anatomic consistency check."* Every clause there is true against the
   verified literature. **Do not** claim you are first to triage at the operative boundary —
   you are not.

2. **It also arms your primary-endpoint argument.** A purpose-built binary triage model still
   only reached 69% sensitivity at this boundary. That is external evidence that IIB recall is
   the genuinely hard, unsolved metric — use it to justify the single-primary-endpoint choice.

3. **Confirm the institutional relationship.** The Injury 2026 cohort (single tertiary
   pediatric orthopaedic centre, 2010–2017, ~1,800 paired films) and the Tan 2025 SMJ paper
   may be from your own group. If so, you must (a) cite them as prior work from the same group,
   (b) state clearly how your cohort relates to theirs (overlap? superset?), and (c) avoid any
   appearance of self-plagiarism or of reusing a test set that touched earlier training. This
   is a data-provenance question that a reviewer will ask directly.

4. **The three empty rows in Part B are your paper.** Full ordinal grading, automated Baumann
   angle on the elbow, and the cross-module consistency check are unoccupied in the verified
   literature. Concentrate the novelty framing there; treat ResNet/YOLO/PCA/SAM2/Grad-CAM as
   acknowledged off-the-shelf assembly.

5. **Verification debt.** I could not open full text (egress blocked), so treat Part D as a
   to-verify list, not a finished bibliography. In particular confirm: the Injury 2026 author
   list/journal/exact metrics; the radiomics paper's authors; and whether any Band-1 paper
   already reports a *graded* (not binary) output that would narrow your gap.
