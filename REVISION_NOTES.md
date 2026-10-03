# Revision notes — `elbow_paper_revised_2.docx`

Changes made to `elbow_paper_revised_1.docx` in response to the two reviewer
requests. Nothing was deleted: every figure and every image in the original is
still in the document, renumbered or relocated.

**How to read the document:**

- **Yellow highlighted `[[FILL: ... ]]`** — a value, sentence or figure file
  that still has to be supplied. 43 in the body, 37 in tables. Search for
  `[[FILL` in Word to step through them.
- **Word comments (39)** — one on every section, table and figure that was
  added, edited, renumbered or moved, saying what changed and why. Turn on
  *Review → All Markup* to see them in the margin.

---

## 1. End-to-end multiclass baseline (request 1)

### New and edited text

| Location | Change |
|---|---|
| **Abstract — Materials and Methods** | Added: a baseline was trained to predict the five grades directly from a paired AP/LAT study, with the same backbone, initialization, schedule and splits. |
| **Abstract — Results** | Added the baseline result and the paired comparison. *Placeholders.* |
| **Abstract — Conclusion** | One sentence needed on the comparison. *Placeholder.* |
| **Abstract — Key Points** | New third key point on the comparison. *Placeholders.* RSNA allows three key points, so you may want to cut one. |
| **§2.8 End-to-end Multiclass Comparison Baseline** *(new)* | Rationale, architecture (2.8.1), training (2.8.2), evaluation (2.8.3). Complete — no placeholders. |
| **§2.10 Quantitative Metrics** | New paragraph defining the error-composition and operative-collapse measures used by Figure 2D and Table 4. |
| **§2.11 Statistical Analysis** | Added the paired tests: exact McNemar on per-patient correctness, and a patient-level bootstrap for the κ difference. |
| **§2.12 Code Availability** | Points at the `research/` directory. |
| **§3.6 End-to-end Multiclass Baseline** *(new)* | Three paragraphs: headline metrics, per-grade comparison and paired tests, error composition. *All numbers are placeholders.* |
| **§4.1 Restatement and principal findings** | States the baseline result. *Placeholders, including a three-way choice of wording.* |
| **§4.7 Hierarchical cascade versus end-to-end multiclass** *(new)* | Three paragraphs. The middle one is deliberately left as a decision rather than prose — see below. |
| **§4.8 Limitations** | New limitation: the baseline was matched to the cascade rather than independently tuned, which bounds what the comparison can claim. |
| **Chapter 5 Conclusion** | Names the comparison. *Placeholder with a three-way choice.* |

### New and edited tables

- **Table 3** — two rows added for the baseline (unfiltered, DRUE-filtered), so
  the comparison sits beside the cascade result. Caption extended.
- **Table 4** *(new)* — cascade versus baseline: per-grade recall with Wilson
  CIs for both arms and the difference, then overall accuracy, macro accuracy,
  κ, 2A/2B boundary errors, adjacent-error share and operative-collapse
  accuracy. Footnote carries the McNemar P and the bootstrap description.
- **Table 5** — the old Table 4 (standardisation and quality control),
  renumbered. Three in-text references were updated.

### §4.7's middle paragraph

Left as an instruction, not prose, because which argument is honest depends on
how the numbers come out. Three cases, with the paragraph to write for each:

- **(a) The two are statistically indistinguishable.** The claim is that the
  cascade's contribution is auditability and explicit modelling of the
  operative boundary, not accuracy. This is already the position §4.6 takes,
  and it is *strengthened* by having a baseline in hand rather than weakened.
  Your prof said he does not expect to beat the baseline — if that holds, this
  is the paragraph, and it is a defensible result, not a concession.
- **(b) The cascade is better.** Say by how much, and attribute it to the
  structure — the per-node class balance is more favourable than the
  five-class balance, which is a mechanism, not just a correlation.
- **(c) The baseline is better.** Report it plainly and argue the cascade's
  value on interpretability: per-decision audit, node-level abstention, and
  criterion-level explanation are things the baseline cannot provide.

### Producing the numbers

```bash
python -m research.multiclass_baseline.manifest  ...   # paired_manifest.csv
python -m research.multiclass_baseline.train     ...   # best_model.pth
python -m research.multiclass_baseline.evaluate  ...   # metrics.json
python -m research.multiclass_baseline.compare   ...   # comparison.json
```

`comparison.json` carries every placeholder in §3.6 and Table 4, under the keys
`cascade`, `multiclass`, `per_grade_recall` and `paired_tests`. `metrics.json`
carries the Table 3 rows. Full instructions in `research/README.md`.

---

## 2. Panel figures (request 2)

### Main text — four composite figures

| Figure | Content | Built from | Script |
|---|---|---|---|
| **1** | Clinical pipeline (A), representative AP processing (B), lateral standardisation (C) | old Figures 2 and 3 | `fig1_pipeline.py` |
| **2** | 5×5 confusion matrix, cascade (A) and multiclass (B); per-grade recall comparison (C); error composition incl. operative-boundary crossings (D) | new | `fig2_classification.py` |
| **3** | Risk–coverage (A), uncertainty/error discrimination (B), calibration (C), four-condition ablation (D) | partly old Figure 9 | `fig3_uncertainty.py` |
| **4** | AHL, cortical-width and Baumann examples (A–C) with their distributions (D–F) | old Figures 6, 7, 8 | `fig4_anatomy.py` |

Figure 3 is placed after Figure 2 so the figures are cited in order; it is cited
from §3.7, whose subject (standardisation and quality control) is what it shows.

### Supplement

| New | Was | Status |
|---|---|---|
| Figure S1 | Figure 11 (ROC) | image moved, renumbered |
| Figure S2 | Figure 10 (AE training curves) | moved; panel B for the baseline still to add |
| Figure S3 | Figure 12 (Grad-CAM) | moved; add more examples as the reviewer asked |
| Figure S4 | — | new, a figure form of Table S2 |
| Figure S5A–C | Figures S1–S3 (failure cases) | renumbered in place in §S1.4 |
| Figure S6 | Figure 1 (Wilkins–Gartland scheme) | moved |
| Figure S7 | Figures 4 + 5 (SAM2, capitellum) | moved and combined |
| Figure S8 | Figure 9 (QC trade-off) | moved |
| Figure S9 | Figures 13 + 14 (cortical profiles) | moved and combined |

**Appendix S2** (new, at the end) holds the images from old Figures 2, 6, 7 and
8. They are the source panels the composite figures are assembled from — export
each to `research/outputs/images/` under the filename in its note, then run the
scripts. Delete the appendix once the composites are final.

LIST OF TABLES and LIST OF FIGURES were rebuilt. All in-text figure and table
cross-references were updated; no stale references remain.

---

## Three things that need a decision

1. **Figure 3 asks for analyses the paper does not yet contain.** Panels A–C
   need a per-radiograph DRUE score and the per-case winning-class probability
   for the held-out cohort (`uncertainty.csv`). Calibration in particular is
   currently listed as a *limitation* in the Discussion — if you add panel C,
   that sentence needs revising. If you would rather not add them, delete the
   sentence I added to §3.3 and drop the panels.

2. **Figure 3D shows four conditions; Table 5 has three.** The table has raw,
   +std, and +std+QC. The reviewer asked for a four-condition ablation, which
   needs the QC-only condition run. Alternatively I can reduce panel D to three
   — say which and I will adjust the script.

3. **Figure 4D is new.** The AHL bisection rate by grade with confidence
   intervals needs per-case bisection outcomes (`ahl.csv`); §3.9.1 currently
   reports only a single summary figure.

## One judgement call you may want to reverse

The Wilkins–Gartland classification diagram was Figure 1 and is now Figure S6,
because the reviewer asked for Figure 1 to be the clinical pipeline. It is the
clinical scheme the whole paper rests on, and a clinician reader may expect it
in the main text. It would also work as a panel A of the new Figure 1 — say the
word and I will move it.

---

## Checks run

- All 24 images in the original are present in the revised document.
- 39 Word comments, all correctly anchored (ranges, references and definitions
  all match); every image relationship resolves; every XML part is well-formed.
- No stale numeric figure references remain in the body text.
- Figures and tables are cited in ascending order.

LibreOffice in this container cannot open either the original or the revised
file, so the document was validated structurally rather than visually. Please
open it in Word and skim the figure placements before sending it on.

---

## Figure inventory — nothing is missing

Audited after the edit: all 24 images from the original are present and each
sits under the right caption. Five slots are empty by design, because they are
the figures the scripts generate.

| Slot | Image | Note |
|---|---|---|
| Figure 1 | **to generate** | `fig1_pipeline.py`. Panel A is drawn in code; panels B and C need exported radiographs. |
| Figure 2 | **to generate** | `fig2_classification.py`. Needs the two prediction CSVs. |
| Figure 3 | **to generate** | `fig3_uncertainty.py`. Needs `uncertainty.csv` and `ablation.csv`. |
| Figure 4 | **to generate** | `fig4_anatomy.py`. Panels A–C reuse the Appendix S2 images. |
| Figure S1 | present | old Figure 11 |
| Figure S2 | present | old Figure 10; panel B for the baseline still to add |
| Figure S3 | present | old Figure 12 |
| Figure S4 | **to generate** | `appendix_figures.py --only s4` |
| Figure S5A–C | present (6 images) | the three failure cases, renumbered in §S1.4 |
| Figure S6 | present | old Figure 1 |
| Figure S7 | present (2) | old Figures 4 + 5 |
| Figure S8 | present | old Figure 9 |
| Figure S9 | present (3) | old Figures 13 + 14 |
| Appendix S2 | present (4) | old Figures 2, 6, 7, 8 — source panels |

The lateral standardisation strip (old Figure 3, four images) stays in §2.4,
where the standardisation it illustrates is described. Its caption is in a Word
text box, so it was relabelled in place rather than moved. Export its five
sub-images for Figure 1C, then delete the strip once Figure 1 is final,
otherwise it duplicates that panel.

### Verification run on the final file

- 24 of 24 original images present; none left loose in Chapter 3.
- No body prose lost. The 44 original text blocks that do not match verbatim
  are all headings, captions and figure-list rows that were deliberately
  renumbered or rewritten.
- 39 Word comments, all correctly anchored; every XML part well-formed.
