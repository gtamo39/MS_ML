# Notebook documentation

Six Jupyter notebooks in [`vignettes/`](../vignettes/) make up the
end-to-end MS-proteomics SAR analysis. They share the same upstream data
([`data/MS/`](../data/MS)) and helper modules (`python/`, `Scripts/`),
but each one answers a different question:

| Notebook                                                                       | Question it answers                                                                |
|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`MS_exploratory.ipynb`](../vignettes/MS_exploratory.ipynb)                    | "Can I predict whether a new compound will be *active vs silent* at all?"          |
| [`MS_Plate_analysis.ipynb`](../vignettes/MS_Plate_analysis.ipynb)              | "Which plates carry usable signal vs scale-compressed noise?"                      |
| [`MS_TargetML.ipynb`](../vignettes/MS_TargetML.ipynb)                          | "Which target genes have predictable SAR *and* pharma/disease relevance?"          |
| [`MS_ML_Prioritization.ipynb`](../vignettes/MS_ML_Prioritization.ipynb)        | "Given a trained gene-model, which library / virtual compounds should we screen?"  |
| [`MS_Interface.ipynb`](../vignettes/MS_Interface.ipynb)                        | "Interactively browse FBX targets by SAR/disease/MS score, their hit compounds, volcanoes, and degradation research." |
| [`MS_cytotox.ipynb`](../vignettes/MS_cytotox.ipynb)                            | "How does compound cytotoxicity across cell lines relate to gene expression (e.g. FBXO31) — and can expression predict sensitivity?" |

All follow the project conventions in [`CLAUDE.md`](../CLAUDE.md):
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
- [`data/patent/20260512_pharma_sm.csv`](../data/patent/20260512_pharma_sm.csv) — big-pharma small-molecule patent targets, used for the `pharma` disease-area override. Built from the Dropbox-side master xlsx (`PATENTS_RAW`) via the resolver in `data/patent/`; bump the dated filename and re-run when the master updates.
- `data/srb_png/<compound>.png` — pre-rendered compound thumbnails for the 3D viz hover/pin panel. See [§ 6. Building `data/srb_png/`](#6-building-datasrb_png--cdd-vault-png-export) below for the download script.

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

Includes a **single-gene parity plot cell** (`6f986240`) that mirrors the
production screen bit-for-bit:
- Per-gene 1-99% winsorize on `logfc` (raw passthrough for the 9-gene
  `RAW_LOGFC_GENES` override set — same list as the production screen).
- Loads `MF_features` from `autoresearch/optimizeMS_genes_R2/logs/inputs_multifp.pkl`
  to match the screen's compound ordering exactly (live `compute_H236_features`
  emits compounds in a different order and shifts K-fold splits).
- Applies the Morgan-bit **prevalence-cut (mean > 0.02)** that the production
  screen uses ([compute_R2_for_all_genes.py](../python/compute_R2_for_all_genes.py)).
- Runs 5-fold CV via `ML_Reg.run_K_Fold_Xval_Regression` with H236 params
  (single RF, n=200, depth=20, max_features=0.3, leaf=2, split=4).
- Scatters `df_pred['real_y']` vs `df_pred['pred_y']`; tunable `PLOT_XLIM`
  / `PLOT_YLIM` knobs let you zoom into the central blob vs the active tail.

Use this cell to (a) confirm a single gene's screen R² matches its parity
plot, and (b) diagnose whether the per-gene R² is genuine SAR or
outlier-anchored (separated tail of strong actives → big drop when you
restrict to the central |logfc|<0.5 band).

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
   - **Per-point tooltip** (small box near the cursor): shows `gene`, R², `overall_score`, `fold`, `fisher_p` (from the MCS enrichment in cell 30, formatted as `< 0.0001` below the floor and `0.NNNN` otherwise), `n`, and `disease_area`.
   - **Hover** the dot → top-right floating panel shows up to 5 compound thumbnails per gene + the gene-level `fisher_p`. Thumbnails come from `data/srb_png/<compound>.png` when available (see [§ 6](#6-building-datasrb_png--cdd-vault-png-export)), else fall back to RDKit rendering from SMILES.
   - **Click** the dot → the panel pins. Compound IDs become triple-click-selectable (copyable). Escape or × to unpin.
   - **In pinned mode, hover a compound thumbnail** → a per-(gene, compound) volcano plot appears below the panel (logfc × −log10 p, target gene ringed). Pre-rendered when `df_raw=` is passed to `plot_target_3d`; opt-in because it adds ~12-60 s render time depending on `volcano_n_jobs` (set `volcano_n_jobs=8` for the 8-CPU parallel path, default 1 = serial).
6. **Top-compound contrast extraction** (cell 34) — for each (gene, top-K compound) pair, look up the matching `uniquecontrast` row in `df_raw` (Mascot / MaxQuant proxy diagnostics).
7. **MCS sweep over K** (cell 35) — sweeps the top-K cutoff (5, 10, 20, …) to see how the consensus substructure degrades — low K → cleanest MCS, high K → broader scaffold.
8. **Compound grid + Bemis–Murcko scaffold enrichment** (cells 36-37) — labelled grid of top-K compounds per gene, plus scaffold-level enrichment of top-K vs the rest.

**Outputs**
- `output/MS/20260509_geneSAR_R2_full_genome.csv` — per-gene CV results (≈11k rows). Schema:
  `gene, R2, nullR2, n, pearson_r, pearson_p, spearman_r, spearman_p`.
  `R2` is squared Pearson (project convention via [`stats_tools.rsquared`](../python/old/Rdkit_tools.py)). `pearson_r` is the *signed* version (catches sign-flips), `spearman_r` is the outlier-robust rank correlation (a Pearson²/Spearman² gap flags outlier-anchored fits). Both `pearson_p` and `spearman_p` are two-sided.
- `output/MS/20260505_target_final_mcs.csv` — per-gene MCS-enrichment scores.
- `~/Downloads/20260505_R2_vs_disease_vs_fold.html` — interactive 3D prioritisation viz with PNG thumbnails.

### Live featurization (production screen)

[`python/compute_R2_for_all_genes.py`](../python/compute_R2_for_all_genes.py) reproduces this notebook's per-gene screen at proteome scale (config: [`config/compute_R2_for_all_genes.yaml`](../config/compute_R2_for_all_genes.yaml)). Two featurization modes via `data.features_source`:

- `pickle` *(default)* — load pre-computed `MF_features` from `autoresearch/optimizeMS_genes_R2/logs/inputs_multifp.pkl`. Fast (~0 s) but assumes the pickle's compound population matches the current chemical library snapshot.
- `compute` — run `rdkit_tools.compute_H236_features(serac_df)` live from `data.chemlib_csv` (default `data/chemical_libs/20260430_SERAC_lib.csv`). Slower (~30 s on 10 k cpds) but matches the `FEATURES_TYPE='prevalence'` branch in the notebook bit-for-bit, including the per-cohort prevalence cut.

---

## 4. `MS_ML_Prioritization.ipynb` — score libraries with a trained model

The downstream "use" notebook: given a per-gene model trained in
`MS_TargetML.ipynb`, score any compound source (Enamine virtual
enumerations, unscreened parts of the SERAC library, custom CSVs)
and write a prioritised SDF for chemistry to triage.

**Inputs**
- A trained joblib bundle (e.g. [`output/ML/trained_models/20260513/PCSK9_RF_H236.joblib`](../output/ML/trained_models/20260513/PCSK9_RF_H236.joblib)) — dict with at least `model`, `feature_cols`, optionally `gene`, `cv_r2`, `features`.
- One or more compound sources:
  - **Virtual enumerations** — Enamine SDFs with `R1_Code` / `R2_Code` tags (`rdkit_tools.get_smiles_df_from_enum`, supports `v=True` tqdm progress).
  - **Unscreened library subset** — CSV / parquet of compounds not yet measured for the target.
  - **Already-screened library** — used as a set-difference baseline (compounds we've tested are excluded from prioritisation).
- A target SDF directory under `DROPBOX_ML + 'virtual libraries'` (when the prioritised compounds need round-trip vendor metadata preserved).

**Pipeline**

1. **Load compound source** (cells 5-9) — read SDF / CSV, canonicalise SMILES, attach `compound`, `R1_Code`, `R2_Code`, `filename` columns. The `filename` column is required by `write_filtered_enum_sdf` when preserving vendor metadata.
2. **Exclude already-screened compounds** (cells 6-7, 10) — set-difference vs the SERAC library subset already measured for the target.
3. **Score with the trained model** (cell 11) — `ML_Reg.MLReg_prioritize_compounds(ori_data, model, top, features_n)`. Loads the bundle, computes the full H236 feature universe on the input, lets `feature_cols` selection apply the training-time prevalence cut, predicts, sorts ascending by `predicted_label`, returns the top-N. Prints a one-line diagnostic: `> GENE  R²=…  features=…  (N cols)  predicted X/Y compounds; top N`.
4. **Write prioritised SDF** (cell 12) — `rdkit_tools.write_filtered_enum_sdf(pred_df, source_dir, out_path, pred_col='predicted_label')`. Two modes:
   - **`source_dir` set** — copy each kept record verbatim from `<source_dir>/<filename>.sdf` (preserves vendor tags, molblock, 3D coords).
   - **`source_dir=None`** — build a new SDF from `df[smiles_col]` directly (`embed_2d=True` writes 2D coords). All non-`smiles` columns become SDF tags by default.

**Outputs**
- `DROPBOX_ML + 'predictions/<date>_<library>_<gene>_pred.sdf'` — prioritised SDF for chemistry.
- (Optional) `pred_df` in-memory — the full scored frame for downstream filtering / spot-checks.

**Key helpers (all in `Scripts/ML_Reg.py`)**
- `ML_Reg.MLReg_prioritize_compounds(...)` — the end-to-end "load model → score → top-N" entry point.
- `ML_Reg.plot_pred_intervals_from_df_pred(pred_df, kind='caterpillar'|'parity'|'fan', ...)` — diagnostic viz from a (compound, pred_y, real_y, low, up) frame produced by `pred_ints`.

---

## 5. `MS_Interface.ipynb` — interactive FBX target browser

A standalone, fully interactive HTML "cockpit" for triaging the FBX
target set. Unlike `MS_TargetML`'s `plot_target_3d` (static highlight,
PNG hovers), this notebook builds `fn.plot_3d_interface` — a 3D scatter
whose colouring is driven live by range sliders + checkbox filters, with
interactive volcanoes and per-gene degradation research.

**Inputs** (the `advantedge`/FBX tranche under `data/advanteidge/`)
- `FBX_MSSCORE` — one row per gene after cell-8 processing (drop noisy
  plates → keep max `ms_score_percent` per gene → inner-join the per-gene
  R² table `GENE_SAR_OUT`). These are the **dots**, with axes
  `R2` (x), `association_score` (y), `ms_score` (z). ~161 plottable genes
  (need all three coordinates).
- `FBX_MEASURE` — per (gene × experiment `uniquecontrast`) `logfc` /
  `pvalue` / `significant`. The **volcano source**.
- `FBX_REPORT` — maps `uniquecontrast → srbnumber` (+ plate, concentration,
  `activity`). Used to link experiments to compounds (batch suffix stripped
  to the `SRB-XXXXXXX` chem-lib key).
- chem library (`CHEMLIB_PATH`) — SMILES for compound thumbnails.
- `GENE_RESEARCH` — a JSON list (one record/gene) of degradation
  research (loaded from a local path; see the load cell). Fields:
  `target_class, lof_therapeutic_benefit, degrader_vs_inhibitor_rationale,
  degrader_feasibility, depmap_dependency, opentargets_top_indications,
  existing_degraders, safety_flags, confidence, biology_rationale, sources`.
  Several of these drive client-side filters (see "Filter panel" below).

**Whole-Px variant (step 4).** Besides the FBX cockpit, the notebook builds a
**whole-proteome** interface from the unified `measure`/`mscore`/`report` tables
(same `fn.plot_3d_interface`): ~4,600 plottable genes, written to
`GTLOCAL/interfaces/` (or `DROPBOX_ML/interfaces/`). It carries the full filter
panel and per-plate volcanoes for every significant-down hit.

**Compound ↔ gene association.** A compound appears under a gene only
where that gene is **significantly down-modulated** in the experiment —
i.e. FBX's `significant == 1 & logfc < 0` (the dataset's own flag, *not* a
`logfc`/p threshold). This matches the volcano colouring one-to-one. One
panel entry per (gene, compound), carrying **all** its passing plates.

**`fn.plot_3d_interface(...)`** writes a self-contained HTML to
`DROPBOX_ML/interfaces/` with these layers (all verified in a headless
browser):

- **Dots** coloured by `disease_area` (pharma/BMS override as in
  `MS_TargetML`). `customdata` is just the gene name; the heavy per-gene
  compound entries live in an injected `__GENE_COMPOUNDS__` map (so slider
  restyles only move tiny arrays — Plotly chokes on restyling jagged
  nested customdata).
- **Range sliders** (`range_sliders=True`) — dual-handle R² / association /
  MS-score sliders, flattened into a 3-column bar bottom-left. A gene is
  coloured iff it's in all three ranges **and** has a visible compound (see
  filters). Defaults to a focused "high-on-all-three" box (~the corner
  subset; binary-searched percentile), with the association handle starting
  at `0.35` (`range_defaults={'y': 0.35}`). A handle within one step of its
  limit is treated as unbounded so axis-extreme genes aren't dropped.
- **Filter panel** (top-left, collapsible groups under two section headers,
  each header carrying a one-line `title=` tooltip):
  - *Compound filters* — **Plates**, **Activity** (`activity`/nr-down levels),
    and **Other** (toggle whole compound classes: **Controls** and
    **Contaminants**, both off by default). **Option B**: a plate-row shows only
    if its plate *and* activity are ticked; a compound lists only if it has a
    visible row; a **gene greys out of the 3D plot** if it has no visible
    compound (`geneHasVisibleCompound` + `cmpAllowed`).
  - *Target filters* — gene-level masks from `GENE_RESEARCH`: **DepMap
    dependency** (Pan-essential / Selective / Non-essential / Other),
    **Confidence** (High / Med / Low), **LoF benefit** (Yes / No / Maybe), and
    **Validation** (a user-supplied validated/devalidated gene split, e.g.
    "FBXO31 dependent" / "FBXO31 independent"). Each greys out genes whose
    category is unticked; a `(no data)` bucket appears only when some plotted
    gene lacks a value.
  - Each group's **default ticked set** is configurable (`activity_defaults`,
    `depmap_defaults`, `conf_defaults`, `lof_defaults`, `validation_defaults`,
    …) so the view can open focused (e.g. Activity = Low + Single, DepMap =
    Selective + Non-essential, Confidence = High + Med, LoF = Yes). `None` = all
    ticked; `[]` = none. An empty group / section auto-hides.
- **Compound panel** (top-right, click a dot to pin) — paginated 5/page
  (◀ ▶ / ←→), structure thumbnail + per-plate volcanoes; click a compound
  to pin its volcano(s). Patents panel sits to its left. Each volcano's caption
  shows the experiment's **MoleculeBatchID** (`SRB-XXXXXXX-NNN`, batch-specific)
  rather than the bare compound id — supplied via an optional `molecule_batch_id`
  column on `compounds_df` (falls back to the compound id when absent).
- **Volcanoes** — `fn.plot_volcano_significant` colours **only significant
  targets** (up/down by logfc sign, rest grey; or **by biological function**
  via `gene_category=`, see below), keyed by `uniquecontrast`,
  `xlim=(-8, 8)`. Rendered as **interactive SVG** (`volcano_significant=True`):
  the dense ~8k-point cloud is rasterised; each significant point is a vector
  marker with a native `<title>` (gene name) → **hover shows the gene**, like
  the 3D dots. Cached to `<interfaces>/volcanoes_px/<hash>.svg` and referenced via
  `<object>` (tiny HTML, lazy-loaded, cached re-runs skip rendering).
- **Control targets** (`control_genes`) — genes whose only significant
  compound(s) are control compounds (`control_compounds`, e.g. GAK →
  `SRB-0000692`) render as **grey diamonds** in a dedicated **"control"**
  legend entry, pulled out of their disease-area colour.
- **Research box** (bottom-right, on hover) — formatted degradation research
  for the gene from `GENE_RESEARCH`: confidence badge, LoF benefit,
  degrader rationale/feasibility, DepMap dependency, indications, safety,
  biology, source links.
- **Invisible backdrop.** The grey "all genes" trace is rendered at
  `opacity=0` (not removed): it stays in the scene so its full data extent
  pins the 3D autorange, keeping the coloured dots anchored as the sliders
  filter them — but nothing grey is drawn, and it's dropped from hover/legend.
- **Axis legend** (bottom-left, above the sliders; auto-positioned via a
  `ResizeObserver`) — hover a row for the full explanation of each axis
  (overridable via `axis_help=`).

**Why most significant-down genes aren't on the plot.** ~2,250 genes are
significantly down-modulated by ≥1 compound, but only the ~161 with FBX
`ms_score` *and* an R² can be placed on the three axes — the rest are
whole-proteome genes the FBX pipeline never scored. The plot already shows
every *plottable* hit gene.

**Key helpers** (in [`python/functions.py`](../python/functions.py))
- `plot_3d_interface(target_df, *, x_col, y_col, z_col, compounds_df,
  volcano_source, volcano_key, range_sliders, range_defaults, control_genes,
  gene_research, volcano_significant, volcano_dir, axis_help, …)` — the
  interface renderer. Long-format `compounds_df` (one row per
  gene/compound/plate) drives the paginated, plate-aware panel.
- `plot_volcano_significant(df, uniquecontrast, gene, …)` — significant-only
  volcano keyed on `uniquecontrast`. Pass `gene_category=` (a `{gene: category}`
  map from `fn.categorize_genes`) + `category_colors=` to colour significant
  points by **biological function** instead of red/blue up/down — direction is
  then shown by marker shape (▲ up, ▼ down).
- `categorize_genes(gene2term, genes=None)` — map each gene to one coarse
  functional category (`CATEGORY_KEYWORDS` / `CATEGORY_COLORS`) from its
  GO/Reactome annotations by **specificity-weighted consensus** (each category
  scores `Σ 1/√term_size` over the gene's terms, so many specific terms beat one
  broad ancestor or incidental tiny term). ~90% of the Px universe gets a
  specific label; heuristic, so multifunctional genes can be debatable (use
  `explain_gene` for authoritative per-gene detail). Built on
  `output/cell_signature/gene2term.parquet`.
- `ora_enrichment(gene_set, background, gene2term, …)` — over-representation
  analysis (**hypergeometric / one-tailed Fisher**) on a *thresholded* set
  (e.g. significant-down genes), measured proteome as background, BH-FDR.
  Maps 1:1 onto the coloured volcano points.
- `gene_category_long(gene_category)` — reshape a `{gene: function}` map into a
  `gene2term`-shaped frame so the coarse **functions** can be used as gene sets
  in `ora_enrichment`/`gsea_preranked` (one enrichment score per function rather
  than per GO term). Relax the size caps — categories are large.
- `gsea_preranked(ranks, gene2term, …)` — **GSEA-preranked** (threshold-free):
  ranks all measured proteins by a signed statistic and tests concentration at
  the top/bottom via a size-matched permutation null (cached by size). Catches
  coordinated subtle shifts ORA misses; processes called by *both* are the
  trustworthy signal.
- `signature_matrix_from_enrichment(func_enrich_all)` — pivot the per-compound
  enrichment into a **compound × function fingerprint** (NES vector per compound).
- `compound_distance_matrix(features, metric='cosine')` — pairwise compound ×
  compound **distance** matrix (smaller = more similar; diagonal NaN), same layout
  as `Rdkit_tools.get_*_distance_matrix`, so it feeds straight into
  `Rdkit_tools.get_NN_from_dist_matrix(d, top=N)` for top-N nearest neighbours.
  Cosine on the signature fingerprint = "same cell signature"; `metric='correlation'`
  on a gene-level logfc table = CMap-style connectivity.
- `plot_function_enrichment(df, sig=0.05, …)` — diverging **lollipop** of one
  compound's per-function GSEA NES (suppressed/induced colour, significant rows
  highlighted + `*`); readable replacement for the flat enrichment table.
- `function_enrichment_all(df_raw, gene_category, n_perm=1000, n_jobs=8, …)` —
  per-compound enrichment of the 15 functions for **every** compound, fanned out
  across compounds with joblib (~25–35 min for 2,277 compounds; cache to
  parquet). Returns tidy `compound × function` with `n_down/n_up`, `ora_down_fdr`,
  `ora_up_fdr`, `gsea_NES`, `gsea_fdr`, `gsea_direction`. Built in MS_TargetML →
  `output/cell_signature/compound_function_enrichment.parquet`.
- `_volcano_svg_string(...)` — interactive SVG volcano (rasterised cloud +
  vector significant points with `<title>` tooltips).
- `_volcano_cache_fname(gene, key, xlim, size_px, ext)` — the single source of
  truth for the volcano disk-cache filename (content-independent hash of
  identity + render params), used by both `plot_3d_interface` and
  `recompute_volcanoes`.
- `recompute_volcanoes(volcano_source, pairs, volcano_dir, *, volcano_key,
  significant, xlim, size_px, n_jobs)` — re-render a specific set of
  `(gene, key)` volcanoes to the cache dir, overwriting in place. Needed after a
  **data** change (the cache is keyed by identity, not values, so a plain re-run
  would reuse stale images). Used by the **"Re-(Compute) Volcanoes"** cell, which
  floors `pvalue == 0.0` to the smallest non-zero p-value (so a true zero plots
  at its real `-log10` instead of the 1e-300 cap = 300) and re-renders only the
  affected experiments; `plot_3d_interface` then picks them up as cache hits.

**Outputs**
- `GTLOCAL/interfaces/20260610_3d_interface_PX_R2_assoc_ms.html` (whole-Px) /
  `DROPBOX_ML/interfaces/…_R2_assoc_ms.html` (FBX) — the cockpit, plus a deferred
  `…_data.js` sidecar and a local `plotly.min.js` written alongside it.
- `<interfaces>/volcanoes_px/<hash>.svg` — cached per-(gene, experiment)
  volcanoes (delete this folder to force a clean re-render after a data or
  `xlim`/`size` change; or use `recompute_volcanoes` to refresh a subset).

---

## 6. `MS_cytotox.ipynb` — cytotoxicity vs. cell-line gene expression

Relates the compound **cytotoxicity** screen (viability across a panel of cancer
cell lines) to **gene expression**, focused on FBXO31: do cell lines that express
more FBXO31 get killed more, and can the transcriptome predict sensitivity?

**Inputs**
- Cytotox viability matrix — `sig` → `mat` (compounds × cell lines, **% viability**;
  ~100 = no effect, **lower = more killing**). A pan-killer compound is dropped
  before per-cell-line aggregation.
- `expr` — `output/cytotox/expr_ourlines.parquet`, DepMap **24Q4** expression for
  our 61 lines (× ~19,193 protein-coding genes). **Unit: `log2(TPM + 1)`** (min 0,
  no negatives, ~16% exact zeros). Rough bins: TPM ≥ 1 = expressed
  (`log2 ≥ 1`); EMBL-EBI/MGI low 0.5–10 / medium 11–1000 / high >1000 TPM
  → `log2(TPM+1)` ≈ 0.6–3.5 / 3.6–10 / >10.
- DepMap cell-line annotations (`cinfo`: lineage, primary disease) for the
  clustering/heatmap tracks.

**Flow**
- Hierarchical clustering + heatmaps of viability (per-compound z-scores; blood
  vs solid; DepMap disease tracks); structural-similarity (Mantel) and GDSC
  mechanism-nomination side analyses.
- **Sensitivity split** — per-cell-line sensitivity = mean % killing across
  compounds; **tertile split** into `sensitive` / `resistant` (ambiguous middle
  dropped).
- **FBXO31 analyses** — violin (FBXO31 expression, sensitive vs resistant, one-sided
  Mann-Whitney), scatter (mean viability *or* sensitivity vs FBXO31 with trend
  line, Spearman + Pearson, most-extreme cell lines labelled), and a **per-compound
  correlation table** (`corr_by_compound`: per-compound Pearson r of viability vs
  FBXO31 across shared lines, + BH-FDR via `scipy.stats.false_discovery_control`).
- **Expression heatmap** — square-cell heatmap of N selected genes × cell lines,
  coloured by expression (slide figure).
- **ML feature matrix** — DepMap expression genes (top-variance) for the labelled
  lines, with three knobs: `COMPOUNDS` (which compounds' mean killing defines
  sensitivity), `FBXO31_MIN` (keep only lines expressing FBXO31 above a
  `log2(TPM+1)` floor), and `SPLIT_FRAC` (extreme top/bottom fraction kept).

**Module** [`python/cytotox_plots.py`](../python/cytotox_plots.py) — keeps the
notebook cells thin; all functions smoke-tested headless:
- `cell_sensitivity(mat, compounds=None)` — per-cell-line mean % killing
  (`100 - viability`) over a chosen compound set.
- `sensitivity_labels(mat, compounds=None, *, method='tertile'|'median', frac=1/3)`
  — discretise sensitivity into `sensitive`/`resistant`. `tertile` keeps the top
  and bottom `frac` (1/3 = tertiles, 0.25 = quartiles, smaller = more extreme),
  dropping the middle; `median` keeps everyone. `compounds=None, frac=1/3`
  reproduces the notebook's global `label`.
- `plot_cytotox_vs_expression(mat, expr, gene='FBXO31', *, metric='viability'|'sensitivity',
  compounds=None, group_label=None, n_label=6, expr_unit='log2(TPM + 1)', …)` —
  scatter of a per-cell-line cytotox metric (averaged over a chosen compound set)
  vs a gene's expression, with trend line, Spearman/Pearson, and the most extreme
  cell lines annotated.
- `plot_gene_expression_heatmap(expr, genes, cells=None, *, group_label=None,
  cmap='viridis', center=None, square=True, standardize=False,
  expr_unit='log2(TPM + 1)', …)` — genes × cell-lines expression heatmap. With
  `group_label` it orders/colours columns by group; otherwise columns follow the
  caller's order (e.g. ranked by FBXO31). `standardize` z-scores each gene (good
  for many genes); `center` aligns a diverging cmap's midpoint.

**Outputs** — figures only (no persisted artifacts); `expr_ourlines.parquet` is a
prebuilt input.

---

## 7. Building `data/srb_png/` — CDD Vault PNG export

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

## 8. Unit tests

The unit tests for the shared helpers (`Statistics_tools.check_ML_data`,
`Rdkit_tools.write_filtered_enum_sdf`, …) live next to the modules they
cover, in [`Scripts/tests/`](../../Scripts/tests/). See
[`Scripts/docs/documentation.md`](../../Scripts/docs/documentation.md) for full
coverage tables and the `python -m unittest discover tests/` recipe.

This `MS_ML/` repo currently doesn't ship its own test suite. If you add
tests that exercise project-specific code (notebook helpers under
[`python/`](../python/), the production screen in
[`python/compute_R2_for_all_genes.py`](../python/compute_R2_for_all_genes.py),
etc.), create `MS_ML/tests/` and document them here.

---

## How the notebooks fit together

```
        ┌─────────────────────────────┐
        │  MS_Plate_analysis.ipynb    │   decides global plate-drop list
        └──────────────┬──────────────┘   ['Plate12','Plate15','Plate23']
                       │
                       ▼  (applied at load time)
        ┌─────────────────────────────┐
        │  MS_TargetML.ipynb          │   per-gene R² screen + 3D viz
        │  (+ python/compute_R2_…)   │   over the full proteome →
        └──────────────┬──────────────┘   GENE_SAR_OUT + trained models
                       │
            ┌──────────┴───────────┐
            ▼                      ▼
 ┌─────────────────────────┐  ┌─────────────────────────────┐
 │ MS_ML_Prioritization    │  │  MS_Interface.ipynb         │  interactive FBX
 │ score libraries → SDF   │  │  R²+assoc+MS-score browser  │  target cockpit
 └─────────────────────────┘  └─────────────────────────────┘  (HTML + volcanoes)

        ┌─────────────────────────────┐   (orthogonal: same features,
        │  MS_exploratory.ipynb       │   different prediction target —
        │                             │   any-down vs target-specific)
        └─────────────────────────────┘
```

Plate analysis is upstream of target ML (provides the drop list).
Target ML produces the per-gene champion models, the prioritisation
table, and the per-gene R² (`GENE_SAR_OUT`). Two notebooks consume that
R²: `MS_ML_Prioritization` scores new compounds with the trained models,
and `MS_Interface` joins it with the FBX MS/disease scores into an
interactive 3D browser. Compound-level exploratory modelling is
orthogonal — same underlying features, different prediction target.
`MS_cytotox` is orthogonal too: it pairs a separate cell-line **cytotoxicity**
screen with **DepMap expression** (not the SAR features) to ask whether
expression — FBXO31 in particular — tracks compound sensitivity.

## Related docs

- [`autoresearch.md`](autoresearch.md) — the iterative SAR optimisation harness referenced from `MS_exploratory` / `MS_TargetML` (cell `run_one` / `update_best_if_improved`).
- [`SAR_prioritization.md`](SAR_prioritization.md) — the deployable SAR-prioritisation policy (single-RF H236 champion config).
- [`findings_FBXO31_2026-04-28.md`](findings_FBXO31_2026-04-28.md) — early case study that motivated the multi-gene generalisation.
