# SAR-driven target prioritisation from proteomics screens

**Created:** 2026-05-04
**Context:** discussion record for the MS-proteomics × SERAC-library workflow
implemented in [vignettes/MS_TargetML.ipynb](../vignettes/MS_TargetML.ipynb).
This document captures the methodology, known pitfalls, and a recommended
workflow for using ML-predictability of `logfc` as a target-prioritisation
signal beyond classical hit counting.

---

## The idea

For each gene/protein measured in a proteomics screen, train a regression
model that predicts the per-compound `logfc` from molecular features
(Morgan FPs + physchem). Use 5-fold cross-validated R² as a proxy for
**how much SAR (structure–activity relationship) signal exists** for that
target.

**Why this matters:** classical screening counts hits (binary
`|logfc| > threshold`) and discards dose-response. A target with subtle but
consistent compound-trend signal — i.e., the chemistry is modulating it in a
structure-dependent way — won't show up as a strong hit but *will* have
predictable SAR. That predictability is itself a fingerprint of
mechanism: if the chemistry can systematically tune a target's abundance,
something real is going on.

**Output:** a shortlist of targets with above-noise predictability, handed to
chemists to expand the corresponding chemical series.

---

## Why this is a sound concept

Not novel — pharma has run variants of this for several years (sometimes
called "SAR-directed target prioritisation" or "chemical-genetics-via-ML").
The core inference is well-grounded:

- Random compound effects → no SAR → low CV R².
- Specific target engagement → SAR → high CV R².
- Therefore, predictability is a *necessary* condition for direct
  engagement. (Not sufficient — see caveats below.)

It's complementary to, not a replacement for, classical hit counting. It
catches the targets that classical methods miss because their effects are
distributed rather than concentrated.

---

## Five concrete improvements (ordered by impact on conclusions)

### 1. Build a shuffled-label null distribution before declaring "reasonable" R²

This is the single most important addition. With ~1.6 k compounds × 2 k
features, even pure noise can produce small positive R² in 5-fold CV. Without
a null, "reasonable" is hand-wavy and you'll flag random fluctuation as
signal.

```python
def null_r2(ML_data, n_perm=20):
    nulls = []
    for seed in range(n_perm):
        ML_perm = ML_data.copy()
        ML_perm['label'] = ML_perm['label'].sample(frac=1, random_state=seed).values
        _, df_pred = ML_Reg.run_K_Fold_Xval_Regression(
            ML_perm, model=model, col_to_rm=['compound','label'],
            ID='compound', v=False, to_impute=None, rm_empty_cols=False,
        )
        nulls.append(ML_Reg.get_reg_metrics_from_preddf(df_pred, verbose=False)['r2'])
    return np.array(nulls)

# real_r2 vs null distribution → empirical p-value per gene
# then BH-correct across all 11 k genes
```

Without this you'll find ~5 % of all 11 k genes (~550) above any reasonable
threshold by chance alone. With it, you can FDR-control to a real shortlist.

### 2. Use scaffold-stratified CV in parallel with random CV

If the SERAC library is dominated by a few PROTAC scaffolds (likely),
**high random-CV R² can mean "the model memorised scaffolds"** rather than
"the chemistry engages the target".

The signature of *real* SAR is **R² preserved across scaffold-CV**. Genes
where random R² is high but scaffold R² collapses to ~0 are scaffold-bias
artefacts — not deprioritised targets, but the prediction isn't reflecting
target engagement either. Filter accordingly.

Implementation: Bemis–Murcko scaffolds + `sklearn.model_selection.GroupKFold`.
Pattern is the same one used in
[autoresearch.md](autoresearch.md).

### 3. Don't rely on R² alone

`logfc` is dominated by zeros for most genes. R² rewards predicting "≈ 0
for everyone", so a model can score acceptably without ever calling a real
hit. Three complementary metrics tell different stories:

| Metric | What it answers |
|---|---|
| **Pearson r / R²** | "Are my predictions calibrated to the labels?" |
| **Spearman ρ** | "Are predictions in the right rank order?" (robust to outliers / zero-mass) |
| **PR-AUC on `\|logfc\| > 1` binarised** | "Can the model distinguish active compounds from the bulk?" |

A gene where R² is mediocre but PR-AUC is high is *exactly* the
case the workflow exists to catch: a few real hits with predictable
structure, swamped by inactive bulk.

Report all three per gene; let them argue with each other.

### 4. Multi-task model instead of N independent models

Most genes in proteomics co-vary (same pathway, same complex, same
regulation). A single multi-task model predicting all 11 k genes
simultaneously typically beats per-gene models by ~10–20 % R² because it
borrows strength. Cheaper too — one fit instead of 11 k.

Implementation options:
- `sklearn.multioutput.MultiOutputRegressor(XGBRegressor(...))`
- ChemProp with multi-task regression head
- A simple shared-trunk MLP if you want pure PyTorch

After the multi-task fit, per-gene R² is computed on the model's per-output
predictions. The first-pass screen across all 11 k targets is essentially
free; you then run dedicated single-target models only on the shortlist.

### 5. Distinguish "predictable abundance" from "direct target engagement"

Predictable `logfc` could mean the compound modulates a *regulator* of that
protein, not the protein itself. To strengthen the inference on shortlisted
hits:

- **Cross-check with thermal proteome profiling (TPP / 2DTPP)** if Serac
  runs it. Direct binders shift thermal stability; abundance-only changes
  don't.
- **Look at consistency with known biology**: a high-R² gene whose top
  compounds also score for known binders of related targets is more
  credible than one with no biological precedent.
- **Cross-reference against the OpenTargets disease-association table**
  (already cached at
  `output/MS/opentargets_target_disease.parquet`). Therapeutically relevant
  × predictable SAR is the actual prioritisation signal — either alone is
  weaker.

---

## Other risks / second-order concerns

### Proteomics measurement noise floor

Each gene's `logfc` has its own measurement error. If a gene's `logfc` SD
across technical replicates is 0.3, you can't get R² above some ceiling —
measurement noise dominates. If you have replicate runs, estimate
per-gene measurement variance and use it to **cap interpretation** (an R² of
0.3 is impressive for a noisy gene, mediocre for a quiet one).

### Multiple testing

With ~11 k genes × however many models / metrics, naive thresholding
multiplies false positives. Always apply Benjamini–Hochberg or
permutation-based q-values across the gene axis.

### Tail-vs-bulk

Most compounds will have `logfc ≈ 0` for any given gene. The interesting
targets have a few outlier compounds with large `|logfc|`. The
PR-AUC-on-binarised-label metric (suggestion #3) is the best single guard
against R² being driven by predicting the bulk correctly while missing
every real hit.

### Compound chemistry diversity

If all 1.6 k compounds are similar, even strong SAR will be hard to detect
because there's not enough variance in features. Conversely,
scaffold-clustered SAR can give artificially high R². Scaffold-CV
(suggestion #2) addresses both.

### Sparse fingerprint features

RandomForest / XGB on 2 k binary FP bits + 1.6 k samples is fine in aggregate,
but individual bits with very few non-zero compounds make CV unstable
(test-set fold composition matters more than it should). For genes that
look "interesting", verify by repeating the CV with different
`random_state` seeds and checking R² stability.

---

## Recommended workflow

1. **Pre-filter** to ~2 k genes with at least 5 compounds at
   `|logfc| > 1` — anything sparser is below the SAR-detectability floor.
2. **Multi-task regression** on those genes, 5-fold CV (random + scaffold).
3. For each gene compute:
   - `real_r2`, `null_r2_p95` (from 20 label permutations)
   - `scaffold_r2`
   - `spearman_rho`
   - `pr_auc_on_active` (label binarised at `|logfc| > 1`)
4. Filter to genes where:
   - `real_r2 > null_r2_p95` (above noise)
   - `scaffold_r2 > 0.1` (not just memorising scaffolds)
   - `pr_auc_on_active > 0.6` (can distinguish hits from bulk)
5. **Cross-reference with OpenTargets** to pull the therapeutically
   relevant subset (drop targets with no disease-association signal).
6. That filtered list is the chemist-ready shortlist for series expansion.

---

## What turns this from "interesting heuristic" into "actionable / publishable result"

The two controls in suggestions #1 and #2 — null-distribution and
scaffold-CV. Without them you can't tell predictability from overfit, and
you'll over-prioritise. With them, the same workflow becomes a defensible
target-discovery tool rather than a fishing expedition.

---

## File pointers

- ML harness: [python/ML_Reg.py](../python/ML_Reg.py) (`run_K_Fold_Xval_Regression`,
  `get_reg_metrics_from_preddf`)
- OpenTargets cache:
  [output/MS/opentargets_target_disease.parquet](../output/MS/opentargets_target_disease.parquet)
- OT helper: [python/Statistics_tools.py](../python/Statistics_tools.py)
  → `get_opentarget_disease_score(df, gene_col, top_n, ot_root)`
- Scaffold-CV reference: [autoresearch.md](autoresearch.md)
- Notebook with the per-gene-loop scaffolding (current state):
  [vignettes/MS_TargetML.ipynb](../vignettes/MS_TargetML.ipynb)
