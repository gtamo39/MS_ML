"""Parallelised full-genome SAR R² screen.

Trains the production model (configurable via YAML) on every gene in df_raw
that has at least `min_compounds` compounds, runs 5-fold compound-split CV,
and writes per-gene R² to CSV (same schema as the reference geneSAR_R2 file:
gene, R2, nullR2, n). Designed for the MS proteomics pipeline.

Key design choices:
- Multiprocessing pool (default 24 workers × 1 thread on 32-core machines —
  empirically optimal; benchmark in YAML).
- Replicates baseline.ipynb's data-cleaning pipeline exactly:
    * MS-recency filter (CLEAN_PROTEOMICS → SERAC source → latest date per
      compound → filter df_raw to those MoleculeBatchIDs).
    * compound = parts[0] + '-' + parts[1] from MoleculeBatchID split('-', n=2).
- Per-gene label override: most genes use winsorised logfc (1-99% per-gene clip);
  a curated list of genes that lose with winsorize use raw `logfc` instead.
- Resume support: if the output CSV exists, only computes genes not yet in it.
- Uses `Statistics_tools.rsquared` (from ../Scripts/, linregress-based R²).

Usage:
    python python/compute_R2_for_all_genes.py --config config/compute_R2_for_all_genes.yaml
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_config
# NOTE: `yaml` is imported lazily inside main() — it's only needed by the CLI
# config path, so notebooks can `import compute_R2_for_all_genes` to reuse
# compute_gene_R2() without pyyaml installed in their env.
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / 'Scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

import Statistics_tools as stats_tools  # noqa: E402
import Rdkit_tools as rdkit_tools       # noqa: E402  (only needed for features_source='compute')


# ---------- module-level state shared with worker processes via fork() ----------
# Workers inherit these via copy-on-write — no pickle overhead per call.
_STATE = {}


def _resolve_model_class(cls_name: str):
    if cls_name == 'RandomForestRegressor':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor
    if cls_name == 'ExtraTreesRegressor':
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor
    if cls_name == 'XGBRegressor':
        from xgboost import XGBRegressor
        return XGBRegressor
    raise ValueError(f'unsupported model class: {cls_name!r}')


def _build_feature_columns(mf_features: pd.DataFrame, preset: str,
                           prevalence_cutoff: float = 0.02):
    """Resolve feature column list from a named preset."""
    if preset == 'multi_fp_champion':
        morgan = [c for c in mf_features.columns
                  if c.startswith('F') and c[1:].isdigit()]
        morgan_filtered = [c for c in morgan
                           if mf_features[c].mean() > prevalence_cutoff]
        physchem = [c for c in ('MW', 'Hba', 'Hbd', 'LogP', 'TPSA', 'NRB')
                    if c in mf_features.columns]
        maccs = [c for c in mf_features.columns if c.startswith('MACCS_')]
        ap    = [c for c in mf_features.columns if c.startswith('AP_')]
        return morgan_filtered + physchem + maccs + ap
    if preset == 'morgan_physchem_only':
        morgan = [c for c in mf_features.columns
                  if c.startswith('F') and c[1:].isdigit()]
        morgan_filtered = [c for c in morgan
                           if mf_features[c].mean() > prevalence_cutoff]
        physchem = [c for c in ('MW', 'Hba', 'Hbd', 'LogP', 'TPSA', 'NRB')
                    if c in mf_features.columns]
        return morgan_filtered + physchem
    raise ValueError(f'unknown feature preset: {preset!r}')


def _winsorize_per_gene(values: np.ndarray, lo_p: float = 1, hi_p: float = 99) -> np.ndarray:
    mask = ~pd.isna(values)
    if mask.sum() < 20:
        return values.copy()
    lo, hi = np.nanpercentile(values[mask], [lo_p, hi_p])
    return np.clip(values, lo, hi)


def _init_worker(df, compound_index, feature_matrix, model_cls, model_params,
                 raw_label_genes, default_label, folds, seed, n_jobs):
    """Inherited via fork() — populate module-level globals."""
    _STATE['df']             = df
    _STATE['compound_index'] = compound_index    # dict: compound → row idx in feature_matrix
    _STATE['feature_matrix'] = feature_matrix    # np.ndarray (n_compounds, n_features)
    _STATE['model_cls']      = model_cls
    _STATE['model_params']   = model_params
    _STATE['raw_label_genes'] = raw_label_genes
    _STATE['default_label']  = default_label
    _STATE['folds']          = folds
    _STATE['seed']           = seed
    _STATE['n_jobs']         = n_jobs


def compute_gene_R2(gene, df, compound_index, feature_matrix, model_cls,
                    model_params, raw_label_genes, folds=5, seed=0, n_jobs=1):
    """5-fold compound-split CV R² (+ Pearson/Spearman r,p) for one gene.

    SINGLE SOURCE OF TRUTH for the per-gene metric — called both by the CLI
    worker ``_compute_gene`` (full-genome screen, via module-level ``_STATE``)
    and directly by notebooks (in-memory ``df_raw`` + ``MF_features``), so the
    two paths can never drift apart (e.g. the notebook silently dropping the
    correlation columns).

    Winsorisation is handled HERE per ``raw_label_genes`` (raw ``logfc`` for
    that set, per-gene 1-99% clip otherwise) — callers pass raw ``df`` with a
    ``logfc`` column and do NOT pre-clip.

    :param str gene: gene symbol to filter ``df['genes']`` on.
    :param df df: must have ``genes``, ``compound``, ``logfc``.
    :param dict compound_index: compound → row index into ``feature_matrix``.
    :param np.ndarray feature_matrix: (n_compounds, n_features) dense float matrix.
    :param type model_cls: e.g. ``RandomForestRegressor``; fresh instance per fold.
    :param dict model_params: kwargs for the model constructor.
    :param set raw_label_genes: genes that use raw ``logfc`` (skip winsorize).
    :param int folds: CV folds. :param int seed: KFold + model random_state.
    :param int n_jobs: threads for the model + joblib backend.
    :return dict: ``{gene, R2, nullR2, n, pearson_r, pearson_p, spearman_r,
        spearman_p, duration_s, label_col[, note]}``.
    """
    t0 = time.time()

    sub = df[df['genes'] == gene]
    if len(sub) == 0:
        return {'gene': gene, 'R2': np.nan, 'nullR2': np.nan, 'n': 0,
                'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan,
                'duration_s': time.time() - t0, 'note': 'no data after plate drop'}

    # Decide label
    if gene in raw_label_genes:
        label_col = 'logfc'
        labels = sub['logfc'].values
    else:
        label_col = 'logfc_clipped'
        labels = _winsorize_per_gene(sub['logfc'].values)

    # Aggregate per compound (mean across replicates)
    df_g = sub[['compound']].copy()
    df_g['_y'] = labels
    agg = df_g.groupby('compound')['_y'].mean().reset_index()
    # Map compounds to feature-matrix rows; drop missing
    agg['_idx'] = agg['compound'].map(compound_index)
    agg = agg.dropna(subset=['_y', '_idx'])
    if len(agg) < 20:
        return {'gene': gene, 'R2': np.nan, 'nullR2': np.nan, 'n': len(agg),
                'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan,
                'duration_s': time.time() - t0,
                'label_col': label_col, 'note': 'too few compounds (<20)'}

    X = feature_matrix[agg['_idx'].astype(int).values]
    y = agg['_y'].values

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_true_all, y_pred_all = [], []
    # threading backend nests fine inside multiprocessing workers (loky doesn't —
    # it silently degrades to n_jobs=1 with a warning).
    with parallel_config(backend='threading', n_jobs=n_jobs):
        for tr_idx, te_idx in kf.split(X):
            model = model_cls(
                **model_params,
                n_jobs=n_jobs,
                random_state=seed,
            )
            model.fit(X[tr_idx], y[tr_idx])
            y_pred_all.append(model.predict(X[te_idx]))
            y_true_all.append(y[te_idx])

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    try:
        r2 = float(stats_tools.rsquared(y_true, y_pred))
    except Exception as e:
        return {'gene': gene, 'R2': np.nan, 'nullR2': np.nan, 'n': len(agg),
                'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan,
                'duration_s': time.time() - t0,
                'label_col': label_col, 'note': f'rsquared failed: {e}'}

    # Pearson r + p: the *signed* version of the squared correlation in R2.
    # Sign flips warn that the model predicts the inverse of the truth (rare
    # but real on small / noisy gene cohorts). p is two-sided.
    try:
        pe_r, pe_p = pearsonr(y_true, y_pred)
        pe_r, pe_p = float(pe_r), float(pe_p)
    except Exception:
        pe_r, pe_p = np.nan, np.nan

    # Spearman rank correlation: complements R² with an outlier-robust signal.
    # Big gap between r² (Pearson²) and ρ² (Spearman²) → fit is anchored on a
    # few extreme actives. Two-sided p-value comes free with the call.
    try:
        sp_r, sp_p = spearmanr(y_true, y_pred)
        sp_r, sp_p = float(sp_r), float(sp_p)
    except Exception:
        sp_r, sp_p = np.nan, np.nan

    # nullR2 = 0 — matches reference output convention (n_null=0 in compute_gene_sar_r2)
    return {'gene': gene, 'R2': r2, 'nullR2': 0, 'n': len(agg),
            'pearson_r': pe_r, 'pearson_p': pe_p,
            'spearman_r': sp_r, 'spearman_p': sp_p,
            'duration_s': time.time() - t0, 'label_col': label_col}


def _compute_gene(gene: str) -> dict:
    """CLI-worker wrapper: reads shared fork()-inherited ``_STATE`` and delegates
    to :func:`compute_gene_R2` (the single source of truth for the metric)."""
    return compute_gene_R2(
        gene,
        df=_STATE['df'],
        compound_index=_STATE['compound_index'],
        feature_matrix=_STATE['feature_matrix'],
        model_cls=_STATE['model_cls'],
        model_params=_STATE['model_params'],
        raw_label_genes=_STATE['raw_label_genes'],
        folds=_STATE['folds'],
        seed=_STATE['seed'],
        n_jobs=_STATE['n_jobs'],
    )


def _parse_compound_batch(molecule_batch_id: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse 'SRB-0001703-001' → ('SRB-0001703', '001'). Matches baseline.ipynb
    exactly: split('-', n=2) then concat parts[0]+'-'+parts[1] (more robust to
    unusual IDs than rsplit)."""
    parts = molecule_batch_id.astype(str).str.split('-', n=2, expand=True)
    return parts[0] + '-' + parts[1], parts[2]


def _to_output_frame(new_results: list[dict], existing: pd.DataFrame) -> pd.DataFrame:
    """Schema: gene, R2, nullR2, n, pearson_r, pearson_p, spearman_r, spearman_p.
    Sort R² descending; NaN R² last. Older CSVs without the new corr columns
    are gracefully upgraded (existing rows get NaN for the new cols)."""
    df_new = pd.DataFrame(new_results)
    if existing.empty:
        df = df_new
    else:
        df = pd.concat([existing, df_new], ignore_index=True)
    # Coerce columns to the reference schema; missing cols filled with default.
    for col, default in [
        ('gene',       None),
        ('R2',         np.nan),
        ('nullR2',     0),
        ('n',          0),
        ('pearson_r',  np.nan),
        ('pearson_p',  np.nan),
        ('spearman_r', np.nan),
        ('spearman_p', np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    df = df[['gene', 'R2', 'nullR2', 'n',
             'pearson_r', 'pearson_p',
             'spearman_r', 'spearman_p']]
    df = df.sort_values('R2', ascending=False, na_position='last').reset_index(drop=True)
    return df


def _print_summary(df_results: pd.DataFrame):
    valid = df_results[df_results['R2'].notna()]
    print(f'\n=== SUMMARY ===')
    print(f'genes evaluated:   {len(valid)} / {len(df_results)}')
    if len(valid):
        print(f'mean R²:           {valid["R2"].mean():+.4f}')
        print(f'median R²:         {valid["R2"].median():+.4f}')
        for thr in (0.30, 0.20, 0.10, 0.05, 0.0):
            n = (valid['R2'] > thr).sum()
            print(f'#genes R² > {thr:>4.2f}:  {n}')
        print(f'#genes R² < 0:     {(valid["R2"] < 0).sum()}')
        if 'pearson_r' in valid.columns and valid['pearson_r'].notna().any():
            pe = valid['pearson_r'].dropna()
            print(f'mean Pearson r:    {pe.mean():+.4f}')
            print(f'median Pearson r:  {pe.median():+.4f}')
            print(f'#genes Pearson r<0: {(valid["pearson_r"] < 0).sum()}')
            for thr in (0.001, 0.01, 0.05):
                n = (valid['pearson_p'] < thr).sum()
                print(f'#genes p_P < {thr:<5g}: {n}')
        if 'spearman_r' in valid.columns and valid['spearman_r'].notna().any():
            sp = valid['spearman_r'].dropna()
            print(f'mean Spearman r:   {sp.mean():+.4f}')
            print(f'median Spearman r: {sp.median():+.4f}')
            for thr in (0.001, 0.01, 0.05):
                n = (valid['spearman_p'] < thr).sum()
                print(f'#genes p_S < {thr:<5g}: {n}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='YAML config path')
    args = ap.parse_args()

    import yaml   # lazy: CLI-only dependency (see import note at top of module)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- Load full df_raw ---
    csv_path = Path(cfg['data']['df_raw_csv'])
    print(f'Loading df_raw from {csv_path} (this may take ~30s for 3.6 GB)...')
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=['logfc'])
    df['compound'], df['batch'] = _parse_compound_batch(df['MoleculeBatchID'])
    print(f'  df_raw: {df.shape[0]:,} rows × {df.shape[1]} cols, '
          f'{df["genes"].nunique():,} unique genes')

    # --- baseline.ipynb's MS-recency filter (mandatory for fidelity) ---
    # Keep only MoleculeBatchID values present in CLEAN_PROTEOMICS (latest-date
    # snapshot of SERAC source compounds). Without this, older retests of
    # the same compound are double-counted in per-compound aggregates.
    ms_path = cfg['data'].get('clean_proteomics_csv')
    if ms_path:
        ms_path = Path(ms_path)
        print(f'Loading clean MS metadata from {ms_path}...')
        MS = pd.read_csv(ms_path)
        if 'CDD Number' in MS.columns:
            MS = MS.drop(['CDD Number'], axis=1)
        # Filter to SERAC source (matches baseline.ipynb)
        MS = MS[MS['MSData - Proteomics activities: Source'] == 'SERAC']
        MS['MSData - Proteomics activities: Date'] = pd.to_datetime(
            MS['MSData - Proteomics activities: Date']
        )
        # Latest test per compound (sort desc, groupby first)
        MS = MS.sort_values('MSData - Proteomics activities: Date',
                             ascending=False).reset_index()
        MS = MS.groupby('Molecule Name').first().reset_index()
        latest_batch_ids = set(MS['MSData - Proteomics activities: Molecule-Batch ID'])
        before = len(df)
        df = df[df['MoleculeBatchID'].isin(latest_batch_ids)].reset_index(drop=True)
        print(f'  MS-recency filter: kept {len(df):,} of {before:,} rows '
              f'({len(latest_batch_ids):,} latest-date SERAC batches)')
    else:
        print('  WARNING: no clean_proteomics_csv in config — skipping MS-recency '
              'filter (results may differ from baseline.ipynb)')

    # Apply plate drop ONCE in the parent (then fork shares it copy-on-write)
    drop_plates = list(cfg['drop_plates'] or [])
    if drop_plates:
        before = len(df)
        df = df[~df['MSPlate'].isin(drop_plates)].reset_index(drop=True)
        print(f'  dropped plates {drop_plates}: {before - len(df):,} rows removed')

    # --- Build / load MF_features ---
    # Two modes — see config docstring for the trade-off.
    features_source = cfg['data'].get('features_source', 'pickle')

    if features_source == 'compute':
        # Match the FEATURES_TYPE='prevalence' branch in baseline.ipynb /
        # MS_TargetML.ipynb's featurization cell exactly:
        #   compute_H236_features(serac_df) → prevalence cut applied below in
        #   _build_feature_columns via the 'multi_fp_champion' preset.
        chemlib_path = Path(cfg['data']['chemlib_csv'])
        print(f'Computing MF_features live from {chemlib_path} (compute_H236_features)...')
        serac_df = pd.read_csv(chemlib_path)
        # Normalise to the |compound|smiles| schema compute_H236_features expects.
        rename_map = {'Molecule Name': 'compound', 'SMILES': 'smiles'}
        serac_df = serac_df.rename(columns=rename_map)
        for c in ('CDD Number', 'Synonyms', 'Projects', 'project'):
            if c in serac_df.columns:
                serac_df = serac_df.drop(columns=[c])
        serac_df = (serac_df[['compound', 'smiles']]
                    .dropna()
                    .drop_duplicates('compound')
                    .reset_index(drop=True))
        print(f'  chemlib: {len(serac_df):,} unique compounds')
        mf_features = rdkit_tools.compute_H236_features(serac_df, v=True)
        print(f'  MF_features: {mf_features.shape}')
    else:
        mf_path = Path(cfg['data']['mf_features_pickle'])
        print(f'Loading MF_features from {mf_path}...')
        with open(mf_path, 'rb') as f:
            mf_pickle = pickle.load(f)
        mf_features = mf_pickle['MF_features']
        print(f'  MF_features: {mf_features.shape}')

    # --- Resolve features ---
    feat_cfg = cfg['features']
    if 'feature_list_path' in feat_cfg:
        with open(feat_cfg['feature_list_path']) as f:
            feature_cols = yaml.safe_load(f) if feat_cfg['feature_list_path'].endswith(('.yml','.yaml')) \
                          else __import__('json').load(f)
        print(f'  loaded {len(feature_cols)} explicit feature names from {feat_cfg["feature_list_path"]}')
    else:
        preset = feat_cfg.get('preset', 'multi_fp_champion')
        prev_cutoff = feat_cfg.get('prevalence_cutoff', 0.02)
        feature_cols = _build_feature_columns(mf_features, preset, prev_cutoff)
        print(f'  preset {preset!r}: {len(feature_cols)} features '
              f'(prevalence >{prev_cutoff:.0%})')

    missing = [c for c in feature_cols if c not in mf_features.columns]
    if missing:
        raise ValueError(f'features missing from MF_features cache: {missing[:5]}... '
                          f'({len(missing)} total)')

    # --- Build compound → feature-matrix-row index ---
    print('Building dense compound feature matrix...')
    feature_matrix = mf_features[feature_cols].astype(np.float32).values
    compound_index = {c: i for i, c in enumerate(mf_features['compound'].values)}
    print(f'  feature matrix: {feature_matrix.shape}, dtype={feature_matrix.dtype}, '
          f'~{feature_matrix.nbytes/1e6:.0f} MB')

    # Sanity check: how much of df_raw's compound set is covered by MF_features?
    df_compounds = set(df['compound'].unique())
    mf_compounds = set(compound_index.keys())
    in_both = df_compounds & mf_compounds
    only_df = df_compounds - mf_compounds
    coverage = 100 * len(in_both) / max(len(df_compounds), 1)
    print(f'  compound coverage: {len(in_both):,} / {len(df_compounds):,} '
          f'df_raw compounds in MF_features ({coverage:.1f}%)')
    if only_df:
        print(f'  ⚠ {len(only_df)} compounds in df_raw but NOT in MF_features '
              f'(silently dropped per-gene). e.g. {sorted(only_df)[:3]}')

    # --- Resolve model ---
    model_cls = _resolve_model_class(cfg['model']['cls'])
    model_params = dict(cfg['model']['params'])
    n_jobs_per_proc = cfg['model'].get('n_jobs_per_process', 4)

    # --- Eligible genes ---
    min_compounds = cfg['filtering']['min_compounds']
    gene_counts = df.groupby('genes')['compound'].nunique()
    eligible = gene_counts[gene_counts >= min_compounds].index.tolist()
    skipped = (gene_counts < min_compounds).sum()
    print(f'\nEligible genes: {len(eligible):,} (skipping {skipped:,} with n_compounds < {min_compounds})')

    # --- Resume support (CSV columns: gene, R2, nullR2, n) ---
    output_path = Path(cfg['output']['results_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if cfg['output'].get('resume', False) and output_path.exists():
        existing = pd.read_csv(output_path)
        already_done = set(existing['gene'].tolist())
        eligible = [g for g in eligible if g not in already_done]
        print(f'Resume: {len(already_done)} genes already done; {len(eligible):,} remain.')

    if not eligible:
        print('Nothing to compute — exiting.')
        if not existing.empty:
            _print_summary(existing)
        return

    # --- Plan ---
    n_processes = cfg['parallelism']['n_processes']
    raw_label_genes = set(cfg['labels']['raw_logfc_genes'] or [])
    print(f'\nPlan:')
    print(f'  {n_processes} workers × {n_jobs_per_proc}-thread RF = '
          f'{n_processes * n_jobs_per_proc} threads')
    print(f'  default label: {cfg["labels"]["default"]}')
    print(f'  raw-logfc override genes ({len(raw_label_genes)}): '
          f'{sorted(raw_label_genes) if raw_label_genes else "(none)"}')
    print(f'  model: {cfg["model"]["cls"]} {model_params}')
    print(f'  CV: {cfg["cv"]["folds"]} folds, seed={cfg["cv"]["seed"]}\n')

    # --- Run ---
    t0 = time.time()
    init_args = (
        df, compound_index, feature_matrix,
        model_cls, model_params,
        raw_label_genes,
        cfg['labels']['default'],
        cfg['cv']['folds'],
        cfg['cv']['seed'],
        n_jobs_per_proc,
    )

    # Periodic checkpointing — flush partial results every N genes
    checkpoint_every = cfg['output'].get('checkpoint_every', 200)
    new_results = []

    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n_processes,
                  initializer=_init_worker, initargs=init_args) as pool:
        try:
            from tqdm import tqdm
            iterator = tqdm(pool.imap_unordered(_compute_gene, eligible),
                            total=len(eligible), desc='genes', smoothing=0.05)
        except ImportError:
            iterator = pool.imap_unordered(_compute_gene, eligible)

        for i, result in enumerate(iterator, 1):
            new_results.append(result)
            if i % checkpoint_every == 0:
                df_partial = _to_output_frame(new_results, existing)
                df_partial.to_csv(output_path, index=False)

    elapsed = time.time() - t0
    print(f'\nWall-clock: {elapsed/60:.1f} min')

    # --- Save final (matches reference column schema: gene, R2, nullR2, n) ---
    df_results = _to_output_frame(new_results, existing)
    df_results.to_csv(output_path, index=False)
    print(f'Saved {len(df_results):,} gene results to {output_path}')

    _print_summary(df_results)


if __name__ == '__main__':
    main()
