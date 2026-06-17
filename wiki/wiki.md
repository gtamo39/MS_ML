# MS_ML — project wiki
_Updated: 2026-06-17. Durable, aggregate memory of this repo — read at session start. Aggregate only: no SMILES / compound IDs._

## Where we are now
- **Focus:** chemistry → cell-signature / phenotype ML in `vignettes/MS_TargetML.ipynb`; just stood up this wiki.
- **Open threads:**
  - MSigDB qualification cell (`msigdb_qual`) is ready but needs the user to drop Hallmark + Reactome `.gmt` into `data/external/msigdb/`, then re-run.
  - Offered, not yet done: scaffold/temporal-split audit of the arrest model (random 5-fold is optimistic — analog leakage); guarded chemprop tail; rename the DNA-rep cluster to "Proliferation ↑" in `label_signature_clusters`.
  - single/low-vs-rest classifier built (`ee227c90`) + deployable save cell (`cb16782f`); PPV cell fixed to use `_MLd`.

## Findings
- **Signature space ≈ 2 clusters (K=2):** arrest (cell-cycle-down) vs proliferation (DNA-replication-up). _[cell d8f3a0c7; config SIG_K]_
- **Arrest cluster** (~176 compounds): cell-cycle/mitosis + DNA-replication suppressed. Predictable from chemistry at a **real but modest** level — HistGB **roc_auc ~0.80** (vs other-active) → **~0.85** (vs whole library), pr_auc ~0.39, mcc ~0.24; baseline 0.79/0.32. _[autoresearch/predict_arrest_cluster: summary.md, best.json; cell c7b6e7c4]_
- **Adding inert (Single/Silent) as negatives raises roc_auc ~0.80→~0.85** — arrest chemistry is more separable from inert than from other-active, and better matches whole-library screening. _[campaign comp_* runs]_
- **DNA-replication-up cluster = coordinated proliferation program:** DNA replication + cell-cycle/mitosis + translation/ribosome + DNA-repair/HR up; adhesion / immune / lipid / transport down. "DNA replication up" is only the single top axis (auto-label = idxmax). _[GSEA NES lollipop; cell c7b6e7c4]_
- **Activity bins:** `Single (1)` + `Low (2-10)` merged → `Single/Low (1-10)`. **single/low is the *majority*** of the screened library (~60%, 1371/2277). _[cells 9daf56c9 / 9b42a6ce merge; ee227c90 classifier; MS.csv]_
- **Best simple arrest model:** `HistGradientBoostingClassifier(lr=0.03, max_depth=6, max_leaf_nodes=15, l2=0.0)`. _[campaign tune_his_24]_

## Decisions & conventions
- **SIG_K = 2** (config.yaml).
- **Strong-signature cohort** (`func_enrich_all_sel`): MS activity in Low/Medium/High **and** peak |GSEA NES| ≥ 2.0 (`SIG_ACTIVITY_BINS`, `SIG_MAX_ABS_NES`). Defines which compounds enter the structure↔cell-state analysis.
- **Cluster picks:** arrest = min mean `Cell cycle / mitosis` NES; DNA-rep-up = max mean `DNA replication` NES.
- **Signature-clustering distance** = Euclidean on per-function-standardized NES (KMeans + t-SNE, both euclidean).
- **Autoresearch:** primary task owns `best.json`; chemistry-only features, no leakage; compare composition variants via roc_auc. _[autoresearch/README.md]_
- **FBX ingest is batch-driven:** `MS_Interface.ipynb` cell 10 concatenates every batch in `config/FBX_BATCHES` (`{}`→MEASURE/MSSCORE/REPORT) under `FBX_DIR`. Batches have disjoint `uniquecontrast` → plain concat, no dedup. Currently **3 batches = 976 experiments** (`20260601`, `20260616`, `20260616_2`). REPORT `time` col is constant 24h, not a date. **Adding a tranche** = add to `FBX_BATCHES` + add its date to the hardcoded `_FBX_DATE` dict in the plate-date cell (else KeyError — not yet auto-derived from the folder name) + one `IFACE_OVERWRITE=True` rebuild run, then flip back to `false`. _[config.yaml FBX_DIR/FBX_BATCHES; cells 10/13]_
- **Per-plate date** (`MS_Interface.ipynb`, cell after the combine): each plate gets a tranche-derived `date` on `measure`/`mscore`/`report`. Plate→tranche is unique for 114/117 plates; the 3 FBX/df_raw overlap plates (`Pw50/63/64`) resolve to FBX `2026-06-01` (FBX assigned last). Both `0616` batches → `2026-06-16`. 4 df_raw stragglers (`Pw58/60/61/62`, 8 experiments) are date-less in every export → `NaT`, fillable via `PLATE_DATE_OVERRIDES` (config). Real per-experiment `Date` col exists in the df_raw exports but is inconsistent (>1/plate) and absent for FBX, so tranche label is canonical. _[config PLATE_DATE_OVERRIDES; cell plate_date_01]_
- **`fn.plot_3d_interface(plate_dates=…)`**: when given a `{plate: 'YYYY-MM-DD'}` map, the Plates filter renders **nested-by-date** — collapsible per-date sub-blocks with tri-state parent checkboxes; falls back to the flat list when omitted. _[functions.py buildPlateGroup; window.__PLATE_DATES__]_
- **2D/3D toggle** (filter panel → "Display" section, pill switch `#disp-toggle`): 2D = orthographic camera locked down the SAR (x) axis so only association (y) × MS (z) show, x-axis hidden, rotation off. Implemented as a camera `relayout` on the SAME `Scatter3d` traces (chose this over native-2D dual traces) → every filter/slider/pin/hover stays wired; SAR range slider still filters though its axis is hidden. **Browser-verified** (headless chromium, synthetic data): 2D→orthographic+x-hidden, 3D→perspective+x-shown. 2D opens **pre-panned toward screen-centre** (CAM2D eye+center share a y/z offset `0.30/-0.35` so the look stays axis-aligned = flat; pan-not-tilt requires shifting eye AND center together — center-only tilts) and `scene.dragmode='pan'` lets you **drag to reposition + scroll to zoom** (stays flat). Fixed offset = roughly centred (a touch left on very wide windows; drag to adjust) — `scene.domain`/aspect/div-resize auto-centring all proved unreliable (div is `96vw×94vh`, overlays viewport-fixed). A perfectly-centred native 2D would need Option B (`Scatter` traces) — deferred. _[functions.py _INTERFACE_INJECT: disp-toggle CSS/HTML/JS]_ The filter panel now **always shows** (was gated on having ≥1 filter group) since the Display section is always present. _[functions.py _INTERFACE_INJECT: disp-toggle CSS/HTML/JS]_
- **Volcano p-value flooring:** a `pvalue==0` plots at y=300 under `-log10` (renderers clip at `1e-300`). Convention is to floor zeros to the smallest non-zero p, capping y at `-log10(pmin)`. Applied in BOTH the Re-Compute Volcanoes cell (floors `measure` in place + overwrites affected cached PNGs) and step 4 (floors the `meas` copy → consistent data + fresh renders). Caveat: the volcano disk cache (`volcanoes_px/`) is keyed by identity+params, NOT data — step 4 reuses existing PNGs, so regenerating old y=300 images needs the Re-Compute cell or clearing the cache dir. _[functions.py plot_volcano_significant / _volcano_svg_string; recompute_volcanoes]_
- **Interface checkpoint** (`IFACE_OVERWRITE`, `IFACE_DIR=output/interface`): the "save or load interface data" cell builds the 4 render inputs (`iface_df`/`compounds_df`/`meas`/`plate2date`) when `IFACE_OVERWRITE=True` and saves them (parquet + `plate2date.json` via stdlib `json` to keep date strings exact); when `False` it loads them and frees the heavy upstream frames (`measure`/`FBX_*`/`df_raw`/…) + `malloc_trim`. Step 4 is now **render-only**. Load path lets you skip the combine/build cells (10–15, 21) — main RAM lever (kernel was ~15 GB holding ~3 copies of the 8M-row table). _[config IFACE_OVERWRITE/IFACE_DIR; cell 1ffa162a build-or-load, ccdb281e render]_
- **Compound-panel cache** (`output/interface/panels.json`, same `IFACE_OVERWRITE` gate): `plot_3d_interface(panels=…, return_panels=True)` skips/emits the ~30s "compound panels" build (the `custom`→`window.__GENE_COMPOUNDS__` blob + plate/activity lists + thumb/volcano modes). Build → returns 3-tuple + saves; load → reads it and skips the rebuild/volcano-scan/thumbnail steps. **Caveat:** on the load path the referenced `srb_png/` + `volcanoes_px/` files must still exist on disk (side effects are skipped); stale if `compounds_df`/render params changed → rebuild with `IFACE_OVERWRITE=True`. Backward-compatible (defaults reproduce old 2-tuple behaviour; restricted-set cell 31 unaffected). _[functions.py plot_3d_interface panels/return_panels]_
- **MSigDB qualification** reuses `fn.gsea_preranked` (no gseapy dependency); `.gmt` files live in `MSIGDB_DIR`.
- **Expression unit** = log2(TPM+1); cytotox heatmap z-scores per gene across cell lines.

## Where things live
- **Notebooks:** `vignettes/MS_TargetML.ipynb` (proteomics→signature→chemistry ML), `MS_cytotox.ipynb` (cytotox / expression), `MS_Interface.ipynb` (ingest / FBX tranches).
- **Helpers:** `python/functions.py` (`gsea_preranked`, `load_gmt`, `mean_logfc_rank`, `label_signature_clusters`, `select_strong_signature_compounds`, plotting). Shared: `~/Scripts/ML_Class.py`, `Statistics_tools.py` (`plot_multi_class_to_embedding` defaults axis labels to "embedding 1/2").
- **Config:** `config/config.yaml` (`SIG_K`, `SIG_ACTIVITY_BINS`, `SIG_MAX_ABS_NES`, `MSIGDB_DIR`, paths).
- **Autoresearch:** `autoresearch/predict_arrest_cluster/` + `autoresearch/README.md` (autonomous-campaign playbook).
- **Key data / outputs:** `output/cell_signature/compound_function_enrichment.parquet` (func_enrich), `output/MS/MS.csv`, `output/MS/df_raw.csv`, `MF_features.csv`.
- **Trained models:** single/low classifier → cell `cb16782f` writes `output/ML/trained_models/20260617_SingleLow_Class/SingleLow_RF.joblib` (joblib bundle: `model` + `feature_cols`; load via `joblib.load`). Save idiom for refit-on-all-data models: cells `4e7a722d` / `cdfe49d6`.

## Reusable functions (reach for these first — don't reimplement)
Import `import ML_Class`, `import Statistics_tools as stats_tools` (both in `~/Scripts`); project helpers in `python/functions.py` (`fn`). **Check here before writing CV / metrics / plots / clustering / enrichment from scratch.**

**ML_Class — classification, CV, metrics, interpretability**
- `run_K_Fold_Xval_Classification(df, ID, model, folds, col_to_rm, …)` — random K-fold CV → (model, pred_df with `real_y`/`probas`). The workhorse.
- `K_fold_by_defined_IDs_Classification(df, ID, ID_sets, model, …)` — CV with **predefined splits** → use for **scaffold / temporal** generalization audits (don't hand-roll the split).
- `plot_roc_curve(pred_dfs, c, l, metric2show, …)` — overlay mean ROC(s) with roc_auc / pr_auc / MCC.
- `metrics_from_pred_df(pred_df)`, `compute_ROC_AUC(real_y, probas)` — metrics from predictions.
- `get_PPV_vs_proba(…)`, `plot_PR_thresholds(df)`, `plot_TPR_TNR_thresholds(df)` — operating-point / threshold curves.
- `get_important_variables(model, df, …)`, `shap_analysis_xgb(model, X, …)` — feature importance / SHAP.
- `Model_gridsearch_parpool_clf(…)`, `kfold_Model_from_params_clf(…)` — hyperparameter search.
- `chemprop_K_Fold_Xval_Classification(…)` — chemprop GNN wrapper (the "chemprop tail").
- `confusion_cf_from_pred_df(df, ctf)`, `format_confusion_matrix(cm)` — confusion matrix.

**Statistics_tools (`stats_tools`) — stats, plots, embedding, utils**
- `check_ML_data(df, …)` — sanity-check an ML df before training.
- `plot_multi_class_to_embedding(X, label, …)` (2D) / `…_3d(…)` (interactive) — labelled embedding scatter; `colors` as list or `{class: color}`; default axes "embedding 1/2".
- `assign_kmean_clusters_from_embedding(df, n_clusters, e1, e2)` — KMeans on an embedding + centroids (don't reinvent the cluster-on-tSNE block).
- `plot_nice_boxplot / violinplot / grouped_violinplot(…)` — publication box/violin with N + significance stars.
- `heatmap(…)` / `annotate_heatmap(…)`, `plot_nice_barplot(…)` — heatmap / barplot primitives.
- `rsquared`, `rmse`, `corr_closure(df)` — regression metrics / fast column correlations.
- `compute_SHAP_from_df(df, cols)`, `shap_feature_ranking(…)` — SHAP from a df.
- `color_col_from_values(…)`, `is_pareto_efficient_simple(costs)`, `get_25_med_75(arr)`, `downsample_df`, `chunker` — misc utilities.

**python/functions.py (`fn`) — project-specific (don't pull gseapy / reimplement):**
- `gsea_preranked`, `ora_enrichment`, `load_gmt`, `function_enrichment_all` — GSEA / ORA (no gseapy needed).
- `select_strong_signature_compounds`, `label_signature_clusters`, `mean_logfc_rank`, `signature_matrix_from_enrichment` — signature / cluster helpers.
- `plot_3d_interface`, `recompute_volcanoes`, `floor_zero_pvalues_and_refresh_volcanoes` — 3D interface + volcano-cache utilities. The last (one-off) floors 0.0 p-values → smallest non-zero p and overwrites affected cached volcanoes; new experiments are floored+rendered fresh by step 4 so it's only for refreshing *existing* cached images. **Its `volcano_dir`/`xlim`/`size_px` MUST match the `plot_3d_interface` call** (currently the notebook's refresh cell points at `GTLOCAL/interfaces/volcanoes_px` while step 4 renders to `DROPBOX_ML/...` — keep in sync).

## Log
- 2026-06-17 — added 2D/3D toggle (Display section in the filter panel) to `plot_3d_interface` — orthographic camera lock down the SAR axis, all interactivity preserved; filter panel now always shown; live-verified in headless chromium on a synthetic interface. (Browser checks: `ml` venv has playwright; chromium needs system libs via `sudo … playwright install-deps chromium`; render synthetic data + drive headless — never open the real interface.)
- 2026-06-17 — added compound-panel cache (`panels`/`return_panels` in `plot_3d_interface`; `output/interface/panels.json`): skips the ~30s panel build on `IFACE_OVERWRITE=False` reloads.
- 2026-06-17 — simplified `MS_Interface.ipynb`: pruned ~24 unused imports, added a single `uc2compound` (uniquecontrast→SRB-XXXXXXX) map in the FBX-load cell reused by all 5 combine cells (was duplicated), looped the chemlib yes/no mapping, dropped local `json` re-imports, `control_compounds = CONTROLS`. Behavior unchanged.
- 2026-06-17 — wrapped the Re-Compute Volcanoes logic into `fn.floor_zero_pvalues_and_refresh_volcanoes`; notebook cell is now a thin call (one-off cache-refresh, not part of the routine loop).
- 2026-06-17 — interface checkpoint (`IFACE_OVERWRITE`/`IFACE_DIR`): build-or-load cell saves/loads the 4 render inputs + frees upstream RAM; step 4 now render-only. Volcano p-floor also applied in the build.
- 2026-06-17 — added per-plate `date` (tranche-derived) + `plot_3d_interface(plate_dates=…)` nested-by-date Plates filter (`buildPlateGroup`); `PLATE_DATE_OVERRIDES` config knob.
- 2026-06-17 — built save cell (`cb16782f`) → single/low model `output/ML/trained_models/20260617_SingleLow_Class/SingleLow_RF.joblib`.
- 2026-06-17 — FBX ingest now config-driven over 3 batches (976 experiments, was 305); added `FBX_DIR`/`FBX_BATCHES`; fixed stale single-batch MSSCORE re-read in cell 13.
- 2026-06-17 — added "Reusable functions" catalog (ML_Class / Statistics_tools / functions.py).
- 2026-06-17 — created wiki + encoded maintenance policy in CLAUDE.md.
- 2026-06-17 — built single/low-vs-rest classifier (`ee227c90`); fixed PPV baseline (`_MLd`).
- 2026-06-17 — added MSigDB qualification cell + `load_gmt` / `mean_logfc_rank` (awaiting `.gmt`).
- 2026-06-17 — overlaid DNA-rep-up vs cell-cycle-down ROC (`c7b6e7c4`); added embedding axis labels.
- 2026-06-16/17 — ran ~2.75h autoresearch campaign for arrest-cluster prediction.
