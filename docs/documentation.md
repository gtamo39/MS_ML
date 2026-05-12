# Notebook documentation

Three Jupyter notebooks in [`vignettes/`](../vignettes/) make up the
end-to-end MS-proteomics SAR analysis. They share the same upstream data
([`data/MS/`](../data/MS)) and helper modules (`python/`, `Scripts/`),
but each one answers a different question:

| Notebook                                                                 | Question it answers                                                       |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`MS_exploratory.ipynb`](../vignettes/MS_exploratory.ipynb)              | "Can I predict whether a new compound will be *active vs silent* at all?" |
| [`MS_Plate_analysis.ipynb`](../vignettes/MS_Plate_analysis.ipynb)        | "Which plates carry usable signal vs scale-compressed noise?"             |
| [`MS_TargetML.ipynb`](../vignettes/MS_TargetML.ipynb)                    | "Which target genes have predictable SAR *and* pharma/disease relevance?" |

All three follow the project conventions in [`CLAUDE.md`](../CLAUDE.md):
heavy intermediates persisted under `data/`/`output/`, parameters in
config files where applicable, no project data sent to any cloud service.

---

## 1. `MS_exploratory.ipynb` — compound-level active/silent classifier

A binary classifier that predicts, from chemistry alone, whether a
compound will produce *any* down-modulation in the proteomics screen
(`Nr. Down > 0` → label = 1). Used as the prioritisation gate for the
weekly enumeration screen.

**Inputs**
- `data/MS/CDD CSV Export - 2026-04-29 06h13m33s.csv` — clean MS table (CDD export).
- Per-compound features built from SMILES on the fly: Morgan FP 2048 bits + 6 physchem descriptors via `rdkit_tools.get_MF_bits_from_df` and `rdkit_tools.compute_properties_from_smiles`.
- `data/20260423_UNC45aproteomics.csv` — raw signal table (vector-length analysis, optional).

**Pipeline**
1. **Data formatting** (cells 7-9) — restrict to `SERAC` source rows, binarise `MSData - Proteomics activities: Nr. Down` into `label ∈ {0,1}`, compute baseline active rate.
2. **Feature build** (cell 12) — Morgan FPs + physchem properties → unified `ML_data`. Sanity-checked with `stats_tools.check_ML_data` (cell 13).
3. **Optional autoresearch loop** (cells 14-15) — cache `ML_data` to `autoresearch/logs/ML_data.pkl` so the autoresearch harness ([`autoresearch.md`](autoresearch.md)) can iterate on it without re-featurising.
4. **Property-vs-label diagnostics** (cells 16-19) — violin/box plots for MW, etc., and `Largest_Vector_Length` analysis (vector descriptors from a separate CDD export).
5. **Chemical-space view** (cells 20-22) — Tanimoto distance matrix + t-SNE coloured by active/silent label. Numba-jitted Tanimoto for speed.
6. **Grid search → final classifier** (cells 23-27) — RF and XGB tuning on the four most-impactful knobs; ROC + PPV-vs-probability curves via `ML_Class.plot_roc_curve` / `ML_Class.get_PPV_vs_proba`.
7. **Feature importance + SHAP** (cells 31-32) — XGB feature importance and per-feature SHAP contributions (`ML_Class.shap_analysis_xgb`).
8. **Enumeration scoring** (cells 28-42) — applies the trained classifier to every SDF file under `output/enumeration/<date>/sdf/`, writes predicted probabilities and a combined SDF; t-SNE of the enumeration set overlaid with predicted-probability heat-map.
9. **Forward-looking validation** (cells 43-49) — splits by `date`, retrains on past tranches, evaluates on the most-recent tranche with conformal-prediction credibility filtering (`pred_icp['cred']`).

**Outputs**
- `output/enumeration/<date>/{csv,sdf}/` — scored enumeration files.
- `output/enumeration/<date>/<combined>.sdf` — single combined SDF for review.
- Plots (saved in-line; nothing else persisted).

---

## 2. `MS_Plate_analysis.ipynb` — plate-quality scan & drop validation

Decides **globally** which proteomics plates should be dropped before any
downstream SAR modelling. Per-gene plate drops are explicitly avoided
(that's model-selection bias against the 6 000-gene production set —
see [`CLAUDE.md`](../CLAUDE.md)).

**Inputs**
- `RAW_PROTEOMICS_PATH` — raw `df_raw` (compound × gene × plate).
- `CHEMLIB_PATH` — chemical library for SMILES + features.
- ~65 highlighted genes (`genes_highlighted`, cell 20) defined in-notebook as the calibration cohort.

**Pipeline**
1. **Load + format** (cells 4-6) — `df_raw`, `serac_df`, Morgan-FP `MF_features`.
2. **Within-compound × cross-plate variance** (cell 8) — for compounds measured on ≥2 plates, compute each (compound, plate) deviation from the compound's mean. Surfaces plates with abnormally large signed bias.
3. **Single-gene ML reference** (cells 10-15) — idempotent bias-correction + plate-drop snapshot, classifier on one gene at a time (ARG1, ANXA13, KDM1B, …) to build intuition and to act as a unit test before the multi-gene scan.
4. **A/B/C/D matrix** (cell 17) — same RF, same fold seed; only the label column (`logfc` vs `logfc_corrected`) and plate-drop status vary. Isolates the contribution of bias-correction from plate-drop.
5. **LOPO CV for a single gene** (cell 18) — leave-one-plate-out per the spec: train on plates ≠ P (compound-mean of `logfc_corrected`), predict the P-plate measurement, R² per held-out plate.
6. **Global plate-quality scan** (cell 21) — `fn.assess_plates_globally` runs LOPO across all ~65 highlighted genes, builds a `(gene × plate)` R² matrix, aggregates per plate:
   - `frac_genes_negative_r2` — fraction of genes for which dropping that plate would *hurt* predictions.
   - `median_r2` across genes.
   - A plate is *recommended for drop* when both metrics cross thresholds.
7. **Drop validation** (cell 22) — `fn.validate_plate_drop` runs 5-fold per-gene CV with vs without the drop set, reports mean ΔR² and per-gene losers so any single-gene regressions are visible.
8. **Cumulative ablation** (cell 23) — `fn.cumulative_plate_ablation` peels plates off one-at-a-time in the order suggested by step 6, producing a marginal-utility curve. Tells you where to stop.

**Outputs**
- Recommended-drop list (currently `['Plate12', 'Plate15', 'Plate23']`, applied downstream in `MS_TargetML.ipynb` and `python/compute_R2_for_all_genes.py`).
- Diagnostic plots only; no persisted artifacts.

---

## 3. `MS_TargetML.ipynb` — target prioritisation via per-gene SAR

The shortlister: which target genes have **both** (a) chemistry that
predicts logfc well (R² above noise floor) and (b) clear pharma /
disease relevance. The output drives the 3D prioritisation viz that
goes to chemistry meetings.

**Inputs**
- `RAW_PROTEOMICS_PATH`, `CHEMLIB_PATH` — same as Plate analysis.
- `OT_ROOT` / `OT_CACHE` — OpenTargets target-disease scores, cached under `output/MS/opentargets_target_disease.parquet`.
- [`data/patent/20260511_pharma_sm.csv`](../data/patent/20260511_pharma_sm.csv) — big-pharma small-molecule patent targets, used for the `pharma` disease-area override.
- `data/srb_png/<compound>.png` — pre-rendered compound thumbnails for the 3D viz hover/pin panel. See [§ 4. Building `data/srb_png/`](#4-building-datasrb_png--cdd-vault-png-export) below for the download script.

**Pipeline**

### Section 0 — Imports & data prep (cells 1-9)
- `sys.path` includes `../Scripts` (shared helpers) and `python/` (project helpers).
- Loads `df_raw`, applies the global plate-drop `['Plate12', 'Plate15', 'Plate23']` (from `MS_Plate_analysis.ipynb`).
- Loads chemical library, computes `MF_features` (multi-FP champion: Morgan + physchem + MACCS + AtomPair via `rdkit_tools.compute_H236_features`).
- Pulls OpenTargets target-disease table, filters to big-pharma priority franchises.

### Section 1 — Single-gene SAR exploration (cells 10-23)
Same one-gene-at-a-time pattern as `MS_Plate_analysis`: pick a gene
(KDM1B, UNC45A, …), aggregate `logfc` per compound, fit RF/XGB, plot
ROC and PPV. Used as a sanity check before launching the full screen.

### Section 2 — Per-gene SAR screen (cells 24-27)
- **Target list** (cell 25) — OpenTargets-ranked genes with sufficient compound coverage.
- **Resumable screen** (cell 26) — for every gene in `target_list`, run `fn.compute_gene_sar_r2` with the H236 production-model params (single RF, n=200, max_depth=20, max_features=0.3, leaf=2, split=4). Per-gene winsorize labels (`logfc_clipped`) with raw-`logfc` override for 9 curated genes (PARP4, TGFBR3, MERTK, PIK3CA, MCL1, PIK3CD, MDM2, ROCK1, UNC45A). Output appended row-by-row to `GENE_SAR_OUT` (`output/MS/20260509_geneSAR_R2_full_genome.csv`) and flushed each gene — kernel kills lose at most one gene.

### Section 3 — Enrichment + visualisation (cells 28-37)
1. **R² shortlist** (cell 29) — `target_final = target2R2_df[n > 400]` sorted by R².
2. **Per-gene MCS enrichment** (cell 30) — for each shortlisted gene, the Maximum Common Substructure of the top-K most-active compounds (parallelised, resumable).
3. **Top-N down-modulators per target** (cell 32) — pulls top-5 compounds (lowest mean `logfc`) per gene with SMILES from `serac_df`. Pivoted wide into `top1..top5_{compound,logfc,smiles}` columns on `target_final`.
4. **Disease-area assignment + pharma override** (cell 32, continued):
   - Each gene gets a `disease_area` from OpenTargets using the `PRIORITY` ranking (cancer → hematology → cardiovascular → immune → … — flagship-pharma franchises).
   - Genes that appear in the pharma patent file *and* clear R² > 0.1 are **relabelled `'pharma'`** — overriding any disease tag. Surfaces the genes where (a) SAR is modellable and (b) pharma is already chasing them.
5. **3D prioritisation viz** (cell 33) — `fn.plot_target_3d` writes a standalone HTML to `~/Downloads/20260505_R2_vs_disease_vs_fold.html`:
   - X = R², Y = OpenTargets `overall_score`, Z = MCS fold-enrichment (log-scale).
   - Highlight set: top-20 closest to the (↑,↑,↑) corner ∪ all genes with `overall_score > 1.5` ∪ all `must_include` (always includes the 'pharma' genes, see cell 33 `_pharma_show`).
   - Colour by `disease_area` using `DISEASE_AREA_COLORS`. Pharma override gets navy `#1D3557`.
   - **Hover** the dot → top-right floating panel shows up to 5 compound thumbnails per gene. Thumbnails come from `data/srb_png/<compound>.png` when available, else fall back to RDKit rendering from SMILES.
   - **Click** the dot → the panel pins, compound IDs become triple-click-selectable (copyable). Escape or × to unpin.
6. **Top-compound contrast extraction** (cell 34) — for each (gene, top-K compound) pair, look up the matching `uniquecontrast` row in `df_raw` (Mascot / MaxQuant proxy diagnostics).
7. **MCS sweep over K** (cell 35) — sweeps the top-K cutoff (5, 10, 20, …) to see how the consensus substructure degrades — low K → cleanest MCS, high K → broader scaffold.
8. **Compound grid + Bemis–Murcko scaffold enrichment** (cells 36-37) — labelled grid of top-K compounds per gene, plus scaffold-level enrichment of top-K vs the rest.

**Outputs**
- `output/MS/20260509_geneSAR_R2_full_genome.csv` — per-gene R²/nullR²/n (≈11k rows).
- `output/MS/20260505_target_final_mcs.csv` — per-gene MCS-enrichment scores.
- `~/Downloads/20260505_R2_vs_disease_vs_fold.html` — interactive 3D prioritisation viz with PNG thumbnails.

---

## 4. Building `data/srb_png/` — CDD Vault PNG export

`MS_TargetML.ipynb`'s 3D prioritisation viz (cell 33) prefers
pre-rendered structure thumbnails from `data/srb_png/<compound>.png`
and falls back to RDKit-on-the-fly only when a PNG is missing. The
PNGs come from CDD Vault via [`python/download_cdd_structures.py`](../python/download_cdd_structures.py),
a local-only script that streams structure images straight to disk —
no chemistry data leaves the machine.

### One-time setup

1. **Get a CDD Vault API token.** In CDD Vault: click your name (top-right)
   → *My Account* → *API Tokens* tab → *Generate New Token*. Copy it.
2. **Save it locally.** Easiest is a plain text file containing just the
   token on one line:
   ```bash
   echo "<paste-token-here>" > ~/.cdd_token
   chmod 600 ~/.cdd_token            # owner-only read
   ```
3. **Install the dependency** (one package):
   ```bash
   pip install requests
   ```

### Verify before running (no files written)

```bash
python python/download_cdd_structures.py \
    --vault 7108 \
    --search 23196193 \
    --token-file ~/.cdd_token \
    --output data/srb_png/ \
    --discover
```
`--discover` probes the API endpoints and dumps 3 sample molecules so
you can confirm your token, vault ID, and saved-search ID are correct
before committing to a full pull.

### Run the full export

```bash
python python/download_cdd_structures.py \
    --vault 7108 \
    --search 23196193 \
    --token-file ~/.cdd_token \
    --output data/srb_png/ \
    --workers 8 --delay 0.05
```
- `--vault 7108` — Serac's CDD vault ID (numeric).
- `--search 23196193` — saved-search ID for the full SERAC library (numeric portion of the search URL — e.g. for `.../searches/23196193-gdca...` use `23196193`).
- `--workers 8 --delay 0.05` — 8 parallel workers with a 50 ms inter-batch nap. Gentle on the CDD API; finishes the full ~10 350-compound library in a few minutes.

### Behaviour

- **Resumable** — files already in `--output` are skipped (`skip=N` in the progress line). Stop and restart freely.
- **Output naming** — `<MoleculeName>.png` (so `SRB-1234567` → `SRB-1234567.png`), matching the `compound` column convention the notebooks use. Add `--strip-prefix` if you'd rather drop the `SRB-` prefix from filenames.
- **Default PNG size** — 600 × 600. Override with `--size 800` if you want sharper hovers in the 3D viz.
- **Error handling** — transient HTTP errors are counted in the breakdown at the end (`submit_http_400=1`, etc.). A handful of errors over a 10k-row pull is normal — re-run to pick them up.
- **Test the wiring first** — use `--limit 50` to download just 50 compounds before launching a full run.

### Typical successful run

```text
[main] resolving search 23196193 in vault 7108...
  listing... 10350/10350
[main] 10350 molecules to fetch
    25/10350  ok=1 skip=24 err=0    9.6/s  ETA  18.0 min
   …
 10350/10350  ok=13 skip=10336 err=1  1748.6/s  ETA   0.0 min  submit_http_400=1
Done.  ok=13  skipped(already-existed)=10336  errors=1
Files in: /home/gtamo/MS_ML/data/srb_png
```
Here `ok=13` means 13 new PNGs were written and `skip=10336` confirms the
folder was already mostly up-to-date — exactly the resumable behaviour
you want.

### Maintenance cadence

Re-run the same command whenever new compounds are added to the SERAC
library; the resume logic ensures only the new ones get fetched. The
`MS_TargetML.ipynb` viz cell doesn't need any change — it picks up the
new PNGs automatically on next render.

---

## How the three notebooks fit together

```
        ┌─────────────────────────────┐
        │  MS_Plate_analysis.ipynb    │   decides global plate-drop list
        └──────────────┬──────────────┘   ['Plate12','Plate15','Plate23']
                       │
                       ▼  (applied at load time)
        ┌─────────────────────────────┐
        │  MS_TargetML.ipynb          │   per-gene R² screen + 3D viz
        │  (+ python/compute_R2_…)   │   over the full proteome
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │  MS_exploratory.ipynb       │   compound-level "will this hit
        │                             │   anything at all?" gate for
        │                             │   the weekly enumeration
        └─────────────────────────────┘
```

Plate analysis is upstream of target ML (provides the drop list).
Compound-level exploratory modelling is orthogonal — same underlying
features, different prediction target (any-down vs target-specific
logfc).

## Related docs

- [`autoresearch.md`](autoresearch.md) — the iterative SAR optimisation harness referenced from `MS_exploratory` / `MS_TargetML` (cell `run_one` / `update_best_if_improved`).
- [`SAR_prioritization.md`](SAR_prioritization.md) — the deployable SAR-prioritisation policy (single-RF H236 champion config).
- [`findings_FBXO31_2026-04-28.md`](findings_FBXO31_2026-04-28.md) — early case study that motivated the multi-gene generalisation.
