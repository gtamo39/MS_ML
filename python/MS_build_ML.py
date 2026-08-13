import os, sys
# self-locate repo root (parent of this file's dir) so `import python.functions` and
# relative paths (config/, data/, output/) resolve no matter where the script is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO_ROOT), 'Scripts'))  # shared helpers (Rdkit_tools, ML_Class, ...)
sys.path.insert(0, os.path.expanduser('~/CDD_Vault_API/python'))          # CDD Vault API (get_df)

import re, gc, csv, json, time, pickle, importlib, argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from datetime import date
from rdkit import Chem
import yaml
from joblib import parallel_config
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm
tqdm.pandas()

# local modules
import python.functions as fn
import Rdkit_tools as rdkit_tools
import ML_Reg as ML_Reg
import ML_Class as ML_Class
import Statistics_tools as stats_tools
from get_library import get_df   # CDD Vault collection export
from mltrail import Registry     # model registry (add / predict / trail)


# Big-pharma R&D priority franchises — a (target, disease) row is "pharma relevant" when its
# OpenTargets therapeutic_areas overlap this set. Default profile is BMS-style; flagship products
# annotate each area to anchor the intuition. Tweak for a different pharma (Lilly leans
# metabolic/CNS, Roche onco/CNS/ophtho, Pfizer vaccines/inflammation).
PHARMA_PRIORITY_AREAS = {
    'cancer or benign tumor',                       # universal — BMS Opdivo/Yervoy, Roche Herceptin/Tecentriq, NVS Kisqali/Pluvicto, Pfizer Ibrance/Padcev, Lilly Verzenio
    'hematologic disease',                          # BMS Revlimid/Pomalyst (myeloma), Pfizer Elrexfio, NVS Tasigna, Roche Hemlibra, Lilly Jaypirca
    'cardiovascular disease',                       # BMS Eliquis, NVS Entresto/Leqvio (PCSK9 siRNA), Pfizer Vyndaqel
    'immune system disease',                        # BMS Zeposia/Sotyktu/Orencia, NVS Cosentyx/Xolair, Pfizer Xeljanz, Lilly Olumiant/Taltz, Roche Actemra
    'musculoskeletal or connective tissue disease', # RA / lupus / psoriasis (overlaps immune)
    'nervous system disease',                       # NVS Kesimpta/Gilenya, Roche Ocrevus/Evrysdi, Lilly Kisunla, BMS Cobenfy
    'psychiatric disorder',                         # BMS Cobenfy (schizophrenia), Lilly historic SSRIs
    'nutritional or metabolic disease',             # Lilly #1 (Mounjaro / Zepbound), Pfizer & NVS GLP-1 follow-ons
    'endocrine system disease',                     # hormone/metabolic overlap; hormone-driven cancers
    'disorder of visual system',                    # Roche ophthalmology (Vabysmo/Lucentis)
    'respiratory or thoracic disease',              # COPD/asthma (NVS Xolair, GSK)
    'infectious disease',                           # vaccines/antivirals (Pfizer Comirnaty/Paxlovid/Prevnar)
}
# OT placeholder labels + out-of-scope categories: a row may not qualify on these alone
PHARMA_DROP_AREAS = {'phenotype', 'measurement', 'biological_process',
                     'animal disease', 'medical procedure'}


# Per-gene regression screen: the model + output schema. RF params are the single-gene cell's
# (100 trees), NOT the 200-tree full-genome screen in compute_R2_for_all_genes.py.
GENE_SCREEN_RF = {'n_estimators': 100, 'max_depth': 20, 'max_features': 0.3,
                  'min_samples_leaf': 2, 'min_samples_split': 4}
GENE_SCREEN_COLS = ['gene', 'R2', 'n', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p']

# cm2rm blacklist parts, in the order they are concatenated. Also the fallback when neither the
# caller nor config CM2RM_PARTS names a subset.
CM2RM_PARTS = ('fbx_independent', 'control_compounds', 'contaminants')

# Deployable activity classifiers as prototyped in MS_ActivityClass. RF, not the HGBC the older
# 1-12 / not-high entries use — hence the _rf suffix on the registry names.
ACTIVITY_RF = {'n_estimators': 200, 'class_weight': 'balanced', 'n_jobs': 16, 'random_state': 0}
# fallback registry names; config's NON_SILENT_MODEL_NAME / SINGLELOW_MODEL_NAME win when present
NON_SILENT_NAME = 'Px_activity_non_silent_rf'
SINGLE_LOW_NAME = 'Px_activity_1_12_rf'

# ---------- state shared with the screen workers via fork() (copy-on-write, no pickling) ----------
_SCREEN = {}


def _init_screen_worker(state):
    """Inherited via fork() — populate the module-level worker state (no pickling)."""
    _SCREEN.update(state)


def _screen_gene(gene):
    """
    5-fold CV regression of logfc for ONE gene. Numerically the single-gene notebook cell
    (ML_Reg.run_K_Fold_Xval_Regression with to_impute=None / rm_empty_cols=False), but reading
    pre-selected rows of the shared float32 feature matrix instead of re-merging MF_features:
    the row set, its shuffle and the fold split are all prepared by _build_screen_tasks.
    param str gene: gene symbol
    return dict: one GENE_SCREEN_COLS row (metrics are NaN when the cohort is under min_compounds)
    """
    S = _SCREEN
    rows, y = S['tasks'][gene]
    row = dict.fromkeys(GENE_SCREEN_COLS, np.nan)
    row.update(gene=gene, n=len(y))
    if len(y) < S['min_compounds']:
        return row

    X = S['X'][rows]   # fancy-index = one copy, already float32/C-order -> sklearn does not re-cast
    y_true, y_pred = [], []
    # MANDATORY: sklearn's default loky backend refuses to nest inside a multiprocessing worker and
    # silently drops to n_jobs=1 ("Loky-backed parallel loops cannot be called in a multiprocessing").
    # Threading nests fine and is what actually makes n_jobs_per_process real.
    with parallel_config(backend='threading', n_jobs=S['n_jobs']):
        for tr, te in KFold(n_splits=S['folds'], shuffle=True, random_state=0).split(X):
            model = RandomForestRegressor(**S['model_params'], n_jobs=S['n_jobs'], random_state=S['seed'])
            model.fit(X[tr], y[tr])
            y_pred.append(model.predict(X[te]))
            y_true.append(y[te])

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    row['R2'] = float(stats_tools.rsquared(y_true, y_pred))   # squared Pearson, as everywhere else here
    row['pearson_r'],  row['pearson_p']  = (float(v) for v in pearsonr(y_true, y_pred))
    row['spearman_r'], row['spearman_p'] = (float(v) for v in spearmanr(y_true, y_pred))
    return row


def _build_screen_tasks(labels, mf_features, cm2rm, genes):
    """
    Turn the per-(gene, compound) labels into the exact per-gene inputs the single-gene cell would
    have produced, once, in the parent: `pd.merge(MF_features, tardf).dropna()` keeps MF_features'
    row order, `[~isin(cm2rm)]` filters it, and run_K_Fold_Xval_Regression then applies
    `sample(frac=1, random_state=0)` — a positional permutation. So per gene the row indices are
    MF_features positions that are NaN-free, not blacklisted and labelled, taken in that order and
    permuted with RandomState(0).
    param df labels: (genes, compound, label) from one groupby over df_raw
    param df mf_features: the feature table (compound + features)
    param set cm2rm: compounds to exclude
    param list genes: genes to prepare
    return tuple: (X float32 matrix, {gene: (row_idx, y)}, n_features)
    """
    fcols = [c for c in mf_features.columns if c != 'compound']
    X = np.ascontiguousarray(mf_features[fcols].to_numpy(dtype=np.float32))
    compounds = pd.Series(mf_features['compound'].to_numpy())
    # rows usable for every gene: no NaN feature (H237 descriptor failures) and not blacklisted
    base = ~np.isnan(X).any(axis=1) & ~compounds.isin(cm2rm).to_numpy()

    tasks, by_gene = {}, dict(tuple(labels.groupby('genes', observed=True)))
    for gene in genes:
        sub = by_gene.get(gene)
        if sub is None:
            tasks[gene] = (np.empty(0, dtype=int), np.empty(0))
            continue
        y = compounds.map(sub.set_index('compound')['label']).to_numpy(dtype=np.float64)
        sel = np.where(base & ~np.isnan(y))[0]
        perm = np.random.RandomState(0).permutation(len(sel))   # == df.sample(frac=1, random_state=0)
        tasks[gene] = (sel[perm], y[sel][perm])
    return X, tasks, len(fcols)


# ~~~~~~~~~~~~~~~~~~~~~~
# CLASSES
# ~~~~~~~~~~~~~~~~~~~~~~

class PARAMS():
    def __init__(self, config_path):
        self.config_path = config_path

    def load_params(self):
        """
        -Read the YAML config and expose every key as an attribute (e.g. params.MS_PATH).
        return self:
        """
        with open(self.config_path) as f:
            self.__dict__.update(yaml.safe_load(f))
        print(f'> loaded {len(self.__dict__) - 1} params from {self.config_path}')
        return self

    def setup_dropbox(self):
        """
        -Bind the Dropbox readers picked by DROPBOX_BACKEND: 'rclone' pulls straight from Dropbox
         (no laptop, no tunnel), 'ssh' streams the laptop's mirror over the reverse tunnel.
         params.dbx opens a path -> BytesIO, params.dbx_glob lists a remote dir.
        return self:
        """
        os.environ.setdefault('DROPBOX_SSH_HOST', self.DROPBOX_SSH_HOST)
        os.environ.setdefault('DROPBOX_SSH_PORT', str(self.DROPBOX_SSH_PORT))
        os.environ['DROPBOX_REMOTE']     = self.DROPBOX_REMOTE       # assign, not setdefault, so config always wins
        os.environ['DROPBOX_LOCAL_ROOT'] = self.DROPBOX_LOCAL_ROOT
        _rclone = self.DROPBOX_BACKEND == 'rclone'
        self.dbx      = fn.open_rclone if _rclone else fn.open_dropbox
        self.dbx_glob = fn.glob_rclone if _rclone else fn.glob_dropbox
        print(f'> Dropbox backend: {self.DROPBOX_BACKEND}')
        return self

    def load_registry(self):
        """
        -Open the shared MLTrail vault (path from the MLTrail config, not this YAML). Kept on params
         because both DATA (load_training_set) and OUTPUT (add / predict) need the same handle.
        return self:
        """
        self.registry = Registry.from_default()
        return self


class DATA():
    def __init__(self):
        self.serac_df = None
        self.MS = None
        self.df_raw = None
        self.MF = None
        self.MF_features = None
        self.control_compounds = None
        self.contaminants = None
        self.fbx_independent = None
        self.cm2rm = None
        self.validated_targets = None
        self.devalidated_targets = None
        self.ot_df = None
        self.ot_pharma = None

    def load_chemical_lib_df(self, params):
        """
        -Load the compound library (name + smiles + Px annotations). With CHEMLIB_OVERWRITE, pull the
         latest straight from CDD Vault (collections AJ/AK) and cache to CHEMLIB_PATH; otherwise read
         the cached csv. Coerce the yes/no annotation columns to 1/0/NaN.
        -Derive the validated / devalidated target lists from Px_Ligase_dependent + Px_Target_interest.
        param class params: PARAMS instance (CHEMLIB_OVERWRITE, CHEMLIB_PATH)
        return None:
        """
        if params.CHEMLIB_OVERWRITE:
            self.serac_df = (get_df(vault=7108, collections=['AK', 'AJ'],
                                    columns=['name', 'Batch Mol-Batch ID', 'smiles', 'Px_repetition(yes/no)',
                                             'Px_validated_WT(yes/no)', 'Px_Ligase_dependent(yes/no)',
                                             'Px_NameLigase_dependent', 'Px_Target_info', 'Px_Target_interest'])
                             .rename(columns={'name': 'compound'}))
            self.serac_df.to_csv(params.CHEMLIB_PATH, sep=',', index=False)
        else:
            self.serac_df = pd.read_csv(params.CHEMLIB_PATH)

        self.serac_df = self.serac_df.drop_duplicates()
        for _c in ['Px_validated_WT(yes/no)', 'Px_Ligase_dependent(yes/no)', 'Px_repetition(yes/no)']:   # 'yes'/'no'/'' -> 1/0/NaN
            self.serac_df[_c] = self.serac_df[_c].astype('string').str.strip().str.lower().map({'yes': 1, 'no': 0})

        ## targets: first token of each ';'-separated Px_Target_interest entry, upper-cased
        _has_tgt = self.serac_df['Px_Target_interest'].notnull()
        _dep = {v: self.serac_df[(self.serac_df['Px_Ligase_dependent(yes/no)'] == v) & _has_tgt] for v in (0, 1)}
        self.devalidated_targets = list({s.split(' ')[0].upper() for x in _dep[0]['Px_Target_interest'] for s in x.split(';')})
        self.validated_targets   = list({s.split(' ')[0].upper() for x in _dep[1]['Px_Target_interest'] for s in x.split(';')})

        print(f'> Chemical lib dim: {self.serac_df.shape} | '
              f'{len(self.validated_targets)} validated / {len(self.devalidated_targets)} devalidated targets')

    def get_contaminants_and_controls(self, params, parts=None):
        """
        -List the control compounds (config CONTROLS), the contaminants (config CONTAMINANTS) and the
         FBX-independent compounds (ligase-independent or not WT-validated), then build the blacklist
         cm2rm from the subset named in `parts`.
        param class params: PARAMS instance (CONTROLS, CONTAMINANTS, CM2RM_PARTS)
        param tuple parts: which of control_compounds / contaminants / fbx_independent enter cm2rm;
                           None -> config CM2RM_PARTS, else the module default CM2RM_PARTS (all three)
        return None:
        """
        parts = tuple(parts or getattr(params, 'CM2RM_PARTS', None) or CM2RM_PARTS)
        if set(parts) - set(CM2RM_PARTS):
            raise ValueError(f'unknown cm2rm part(s) {sorted(set(parts) - set(CM2RM_PARTS))}; pick from {CM2RM_PARTS}')
        sdf = self.serac_df
        self.control_compounds = list(params.CONTROLS)
        self.contaminants      = list(pd.read_csv(params.CONTAMINANTS)['Molecule Name'])
        self.fbx_independent   = list(sdf[(sdf['Px_Ligase_dependent(yes/no)'] == 0) |
                                          (sdf['Px_validated_WT(yes/no)'] == 0)]['compound'].unique())
        self.cm2rm = [c for p in parts for c in getattr(self, p)]
        print(f'> {len(self.control_compounds)} control + {len(self.contaminants):,} contaminant compounds (from config) '
              f'+ {len(self.fbx_independent)} fbx independent | cm2rm = {" + ".join(parts)} = {len(self.cm2rm)}')

    def load_proteomics_data(self, params):
        """
        -With DFRAW_OVERWRITE: rebuild df_raw (per-gene logfc/pvalue) and MS (per-compound activity)
         from the three CDD/Database tranches plus every FBX tranche under FBX_DIR, collapse to the
         latest batch/date per compound, drop controls+contaminants and non-library compounds, then
         cache both to DFRAW_PATH / MS_PATH. Otherwise read the cached parquets.
        -Either way, config EXCLUDE_DATES tranches are then dropped from df_raw (see
         drop_excluded_dates) — in memory, never from the cache.
        param class params: PARAMS instance (DFRAW_OVERWRITE, RAW_/CLEAN_PROTEOMICS_PATH, PX_*, FBX_DIR, MS_PATH, DFRAW_PATH, EXCLUDE_DATES)
        return None:
        """
        if not params.DFRAW_OVERWRITE:
            self.MS     = pd.read_parquet(params.MS_PATH)
            self.df_raw = pd.read_parquet(params.DFRAW_PATH)
            print(f'> loaded cached MS {self.MS.shape} | df_raw {self.df_raw.shape}')
            self.drop_excluded_dates(params)
            return

        # -------------------
        # Old data (CDD Vault / Database exports)
        # -------------------
        df_raw_20260429, MS20260429 = fn.load_proteomics_data(
            params.RAW_PROTEOMICS_PATH,
            params.CLEAN_PROTEOMICS_PATH,
            drop_plates=['Plate12', 'Plate15', 'Plate23'],
        )
        df_raw_20260520, MS20260520 = fn.load_proteomics_data(
            params.dbx(params.PX_20260520_DB),         # raw per-gene table (Database export)
            params.dbx(params.PX_20260520_CDDVAULT),   # metadata table (Vault export: SMILES, Collections)
            drop_plates=['Plate12', 'Plate15', 'Plate23'],
            mode='cddvault',                           # Collections recipe (drops PROTACs), join on 'Batch Molecule-Batch ID'
            collections=['AJ', 'AK'],
        )
        df_raw_20260529, MS20260529 = fn.load_proteomics_data(
            params.dbx(params.PX_20260529_DB),
            params.dbx(params.PX_20260529_CDDVAULT),
            drop_plates=['Plate12', 'Plate15', 'Plate23'],
            mode='cddvault',
            collections=['AJ', 'AK'],
        )
        # manual formatting for the 20260529 metadata (different export header)
        MS20260529 = MS20260529.rename(columns={
            'Molecule-Batch ID': 'Batch Molecule-Batch ID',
            'Nr. Down': 'MSData - Proteomics activities: Nr. Down',
            'Cmpd Activity': 'MSData - Proteomics activities: Cmpd Activity'})
        parts = MS20260529['Batch Molecule-Batch ID'].str.split('-', n=2, expand=True)
        MS20260529['Molecule Name'] = parts[0] + '-' + parts[1]   # 'SRB-0000385'
        MS20260529['batch'] = parts[2]                            # '001'

        ## df_raw — tag each tranche with its date, then keep the latest batch/date per compound
        self.df_raw = pd.concat([
            df_raw_20260429.assign(date=pd.to_datetime('20260429')),
            df_raw_20260520.assign(date=pd.to_datetime('20260520')),
            df_raw_20260529.assign(date=pd.to_datetime('20260529')),
        ]).reset_index(drop=True)
        self.df_raw = fn.keep_latest_batch_per_compound(self.df_raw)
        self.df_raw[['genes']].drop_duplicates().to_csv('data/MS/Px_genes.csv')

        ## MS — per-compound activity / ndown
        _cols = ['Molecule Name', 'MSData - Proteomics activities: Nr. Down', 'origin',
                 'MSData - Proteomics activities: Cmpd Activity']
        self.MS = (pd.concat([MS20260429.assign(origin='MS20260429'),
                              MS20260520.assign(origin='MS20260520'),
                              MS20260529.assign(origin='MS20260529')]).reset_index(drop=True)[_cols]
                   .rename(columns={'Molecule Name': 'compound',
                                    'MSData - Proteomics activities: Nr. Down': 'ndown',
                                    'MSData - Proteomics activities: Cmpd Activity': 'activity'}))
        self.MS['date'] = pd.to_datetime(self.MS['origin'].str.replace('MS', ''))
        del df_raw_20260429, df_raw_20260520, df_raw_20260529

        # -------------------
        # New (FBX) data
        # -------------------
        # every tranche folder under FBX_DIR (config.yaml), named by export date
        TRANCHES = sorted(params.dbx_glob(params.FBX_DIR, '[0-9]*'))
        _fbx = {os.path.basename(t): fn.load_fbx_tranche(t, control_compounds=self.control_compounds,
                                                         contaminants=self.contaminants,
                                                         opener=params.dbx, lister=params.dbx_glob)
                for t in TRANCHES}

        # stack tranches; on a recurring experiment (uniquecontrast) / compound, keep the LATEST export
        df_raw_fbx = (pd.concat([d.assign(_tranche=k) for k, (d, _) in _fbx.items()], ignore_index=True)
                      .sort_values('_tranche').drop_duplicates(['uniquecontrast', 'pg'], keep='last'))  # row key is (contrast, protein); deduping on contrast alone collapsed the proteome
        # stamp each FBX row with its tranche date (folder YYYYMMDD) — FBX_DFRAW_COLS carries no 'date',
        # so without this keep_latest_batch_per_compound sees NaT and drops every FBX row (NaT==NaT is False).
        df_raw_fbx['date'] = pd.to_datetime(df_raw_fbx['_tranche'].str[:8], format='%Y%m%d')
        df_raw_fbx = df_raw_fbx[fn.FBX_DFRAW_COLS + ['date']].reset_index(drop=True)
        MS_fbx = (pd.concat([m for _, m in _fbx.values()], ignore_index=True)
                  .sort_values('date').drop_duplicates('compound', keep='last')
                  [fn.FBX_MS_COLS].reset_index(drop=True))

        # align to the LIVE df_raw / MS schemas so a plain concat works regardless of any extra
        # columns the in-memory frames carry (e.g. smiles); columns FBX can't fill become NaN.
        _miss_dr = [c for c in self.df_raw.columns if c not in df_raw_fbx.columns]
        _miss_ms = [c for c in self.MS.columns     if c not in MS_fbx.columns]
        df_raw_fbx = df_raw_fbx.reindex(columns=self.df_raw.columns)
        MS_fbx     = MS_fbx.reindex(columns=self.MS.columns)
        print(f'\n> FBX formatted: df_raw_fbx {df_raw_fbx.shape} | MS_fbx {MS_fbx.shape}')
        if _miss_dr: print(f'  [warn] df_raw cols NOT populated from FBX (set NaN): {_miss_dr}')
        if _miss_ms: print(f'  [warn] MS cols NOT populated from FBX (set NaN): {_miss_ms}')

        # -------------------
        # Merging
        # -------------------
        self.df_raw = pd.concat([self.df_raw, df_raw_fbx], ignore_index=True).drop_duplicates()
        self.df_raw = fn.keep_latest_batch_per_compound(self.df_raw)   # collapse batches post-FBX too
        self.MS = fn.collapse_ms_latest_measurement(pd.concat([self.MS, MS_fbx], ignore_index=True))

        ## 2026-06-01 holds a single compound — fold it into the 2026-06-16 tranche
        self.MS.loc[self.MS['date'] == pd.Timestamp('2026-06-01'), 'date'] = pd.Timestamp('2026-06-16')

        ## drop controls/contaminants, and keep FBX glues only (no PROTACs) on both frames
        self.MS = self.MS[~self.MS['compound'].isin(self.control_compounds + self.contaminants)]
        self.MS     = self.MS[self.MS['compound'].isin(self.serac_df['compound'])]
        self.df_raw = self.df_raw[self.df_raw['compound'].isin(self.serac_df['compound'])]

        self.MS.to_parquet(params.MS_PATH, index=False)
        self.df_raw.to_parquet(params.DFRAW_PATH, index=False)
        print(f'> rebuilt MS {self.MS.shape} -> {params.MS_PATH} | df_raw {self.df_raw.shape} -> {params.DFRAW_PATH}')
        self.drop_excluded_dates(params)

    def drop_excluded_dates(self, params):
        """
        -Drop the screen dates listed in config EXCLUDE_DATES from df_raw, so every downstream
         model (notably the per-gene screen) ignores those tranches. Applied AFTER caching, i.e.
         in memory only: the parquet keeps every date, so emptying the key restores them.
        -MS is deliberately left alone — the activity-composition plots exclude dates themselves,
         and dropping compounds there would silently shrink the classifiers' training set.
        param class params: PARAMS instance (EXCLUDE_DATES)
        return None:
        """
        dates = pd.to_datetime(getattr(params, 'EXCLUDE_DATES', None) or [])
        if not len(dates):
            return
        # both sides must be datetime64 — isin() against a list of date STRINGS never matches
        drop = pd.to_datetime(self.df_raw['date']).isin(dates)
        print(f'> EXCLUDE_DATES {[d.strftime("%Y-%m-%d") for d in dates]}: dropped {drop.sum():,} df_raw '
              f'rows / {self.df_raw.loc[drop, "compound"].nunique():,} compounds (MS untouched)')
        self.df_raw = self.df_raw[~drop]

    def load_opentargets(self, params):
        """
        -Load the OpenTargets (target, disease) association scores for every gene seen in df_raw,
         cached to OT_CACHE. On a cache miss, resolve the gene symbols (multi-gene peptide strings
         exploded on [;,|]) against the local bulk dump under OT_ROOT and write the cache.
        param class params: PARAMS instance (OT_CACHE, OT_ROOT)
        return None:
        """
        if os.path.exists(params.OT_CACHE):
            self.ot_df = pd.read_parquet(params.OT_CACHE)
            print(f'> loaded cached {params.OT_CACHE}')
        else:
            unique_genes = pd.DataFrame({'gene': (self.df_raw['genes'].dropna().astype(str)
                                                  .str.split(r'[;,|]', regex=True).explode().str.strip()
                                                  .replace('', pd.NA).dropna().unique())})
            print(f'> {len(unique_genes):,} unique gene symbols in df_raw')
            self.ot_df = fn.get_opentarget_disease_score(unique_genes, gene_col='gene', top_n=20,
                                                         ot_root=params.OT_ROOT)
            os.makedirs(os.path.dirname(params.OT_CACHE), exist_ok=True)
            self.ot_df.to_parquet(params.OT_CACHE, index=False)
            print(f'> wrote {params.OT_CACHE}')

        print(f'> {len(self.ot_df):,} (target, disease) rows / '
              f'{self.ot_df["target_symbol"].nunique():,} targets resolved')

    def filter_pharma_targets(self, priority=PHARMA_PRIORITY_AREAS, drop=PHARMA_DROP_AREAS, v=True):
        """
        -Keep the ot_df rows whose therapeutic_areas overlap the pharma priority franchises and
         aren't made up of DROP placeholders alone -> ot_pharma.
        param set priority: therapeutic areas that qualify a row (default PHARMA_PRIORITY_AREAS)
        param set drop: OT placeholder / out-of-scope areas that can't qualify a row on their own
        param bool v: print the per-area row breakdown
        return None:
        """
        _areas = self.ot_df['therapeutic_areas'].fillna('').astype(str).str.split('|')
        _keep = _areas.apply(lambda t: bool(set(t) & priority) and bool(set(t) - drop))
        self.ot_pharma = self.ot_df[_keep].reset_index(drop=True)
        print(f'> {len(self.ot_pharma):,} / {len(self.ot_df):,} (target, disease) rows pass the pharma filter '
              f'| {self.ot_pharma["target_symbol"].nunique():,} unique targets')
        if v:   # a row can hit several areas, so these counts don't sum to len(ot_pharma)
            _pa = self.ot_pharma['therapeutic_areas'].str.split('|')
            for area in sorted(priority):
                print(f'  {_pa.apply(lambda t: area in t).sum():>6,}  {area}')

    def compute_features(self, params):
        """
        -Compute the chemistry feature matrix for the library, switching on FEATURES_TYPE:
         prevalence (H236 keeping Morgan bits >2%) | H236 | H237 (H236 + descriptastorus DS_*)
         | autoresearch (pickled) | anything else -> Morgan bits + physchem properties.
        -Also keeps the plain 2048-bit Morgan matrix as data.MF (used by the Tanimoto NN analyses).
        param class params: PARAMS instance (FEATURES_TYPE)
        return None:
        """
        print('> checking smiles')
        failed_CMs = rdkit_tools.check_smiles_RDKiT(self.serac_df)
        self.serac_df = self.serac_df[~self.serac_df['compound'].isin(failed_CMs)]

        print('> FEATURES_TYPE:', params.FEATURES_TYPE)
        self.MF = rdkit_tools.get_MF_bits_from_df(self.serac_df, nBits=2048)

        if params.FEATURES_TYPE == 'prevalence':
            _mf = rdkit_tools.compute_H236_features(self.serac_df, v=True)
            _morgan = [c for c in _mf.columns if c.startswith('F') and c[1:].isdigit()]
            self.MF_features = _mf.drop(columns=[c for c in _morgan if _mf[c].mean() <= 0.02]).copy()

        elif params.FEATURES_TYPE == 'autoresearch':
            with open('autoresearch/optimizeMS_genes_R2/logs/inputs_multifp.pkl', 'rb') as fh:
                self.MF_features = pickle.load(fh)['MF_features']

        elif params.FEATURES_TYPE == 'H236':
            self.MF_features = rdkit_tools.compute_H236_features(self.serac_df, v=True)

        elif params.FEATURES_TYPE == 'H237':
            # H236 + descriptastorus RDKit2D descriptors (DS_* columns, CDF-normalised to [0,1]).
            # the 200 descriptors dominate the runtime -> parallelise; ~32 is the sweet spot (not -1)
            self.MF_features = rdkit_tools.compute_H237_features(self.serac_df, n_jobs=32, v=True)

        else:
            self.MF_features = pd.merge(self.MF, rdkit_tools.compute_properties_from_smiles(self.serac_df))

        self.MF_features = self.MF_features.drop_duplicates()
        print(f'> {len(failed_CMs)} smiles failed | MF {self.MF.shape} | MF_features {self.MF_features.shape}')


class OUTPUT():
    def __init__(self):
        self.gene_metrics = None
        self.non_silent_cv = None
        self.non_silent_model_id = None
        self.single_low_cv = None
        self.single_low_model_id = None

    def save_non_silent_model(self, data, params, model=None, folds=5, name=None,
                              overwrite=False):
        """
        -Train and register the deployable non-silent ("active") binary classifier:
         label = ndown > 0 over the whole library (see _save_activity_model for the rest).
        param class data: DATA instance (MS, MF_features, serac_df)
        param class params: PARAMS instance (registry, FEATURES_TYPE, NON_SILENT_MODEL_NAME)
        param model: sklearn classifier; None -> RandomForestClassifier(**ACTIVITY_RF)
        param str name: MLTrail experiment_name; None -> config NON_SILENT_MODEL_NAME
        param bool overwrite: reset the latest version instead of appending one
        return None: metrics land in self.non_silent_cv, the id in self.non_silent_model_id
        """
        self.non_silent_cv, self.non_silent_model_id = self._save_activity_model(
            data, params, label=(data.MS['ndown'] > 0),
            name=name or getattr(params, 'NON_SILENT_MODEL_NAME', NON_SILENT_NAME),
            task='activity_non_silent_binary', positive='ndown > 0',
            model=model, folds=folds, overwrite=overwrite)

    def save_single_low_model(self, data, params, model=None, folds=5, name=None,
                              overwrite=False):
        """
        -Train and register the deployable single/low-activity binary classifier:
         label = 1 <= ndown <= 12, i.e. compounds that degrade something but stay selective —
         the positives silent-vs-active can't separate (see _save_activity_model for the rest).
        param class data: DATA instance (MS, MF_features, serac_df)
        param class params: PARAMS instance (registry, FEATURES_TYPE, SINGLELOW_MODEL_NAME)
        param model: sklearn classifier; None -> RandomForestClassifier(**ACTIVITY_RF)
        param str name: MLTrail experiment_name; None -> config SINGLELOW_MODEL_NAME
        param bool overwrite: reset the latest version instead of appending one
        return None: metrics land in self.single_low_cv, the id in self.single_low_model_id
        """
        self.single_low_cv, self.single_low_model_id = self._save_activity_model(
            data, params, label=data.MS['ndown'].between(1, 12),
            name=name or getattr(params, 'SINGLELOW_MODEL_NAME', SINGLE_LOW_NAME),
            task='activity_single_low_binary', positive='1 <= ndown <= 12',
            model=model, folds=folds, overwrite=overwrite)

    def _save_activity_model(self, data, params, label, name, task, positive, model=None,
                             folds=5, overwrite=False):
        """
        -Train and register one deployable binary activity classifier: the caller supplies the
         label mask over data.MS, features are MF_features (whatever FEATURES_TYPE is).
        -Score it by `folds`-fold CV first (roc_auc / pr_auc go into the registry as the entry's
         metrics), then refit on the whole library and hand the {model, feature_cols} bundle
         to MLTrail together with the compound/smiles/label training set, so predictions on new
         SMILES are re-featurized by MLTrail's own featurizer for this FEATURES_TYPE.
        -Re-running adds a NEW VERSION to the existing entry of that name (history is kept);
         `overwrite=True` resets the latest version in place instead — use that only to correct
         a bad version, never for a retrain.
        param series label: boolean mask over data.MS's rows defining the positive class
        param str name: MLTrail experiment_name (the entry the version is appended to)
        param str task/positive: bundle metadata documenting what was learnt
        return tuple: (CV metrics dict, MLTrail model id)
        """
        ML_data = data.MS.copy()
        ML_data['label'] = np.asarray(label, dtype=int)   # positional: MS may carry a duplicated index
        ML_data = pd.merge(ML_data[['compound', 'label']], data.MF_features, on='compound')
        feat_cols = [c for c in ML_data.columns if c not in ('compound', 'label')]
        print(f'> {name}: {len(ML_data):,} rows | features={len(feat_cols):,} ({params.FEATURES_TYPE})')
        print(ML_data['label'].value_counts().to_string())

        clf = model if model is not None else RandomForestClassifier(**ACTIVITY_RF)

        ## CV metrics for the registry
        _, cv_pred = ML_Class.run_K_Fold_Xval_Classification(
            ML_data, ID='compound', model=clf, folds=folds, col_to_rm=['compound', 'label'],
            v=False, ctf=0.5, impute_by_mean=False)
        cv = ML_Class.metrics_from_pred_df(cv_pred)
        print(f"> CV roc_auc={cv['roc_auc']:.3f}  pr_auc={cv['pr_auc']:.3f}")

        ## refit on the whole library; MLTrail joblib-dumps the bundle into the vault
        clf.fit(ML_data[feat_cols], ML_data['label'])
        bundle = {'model': clf, 'feature_cols': feat_cols, 'task': task,
                  'positive': positive, 'features': params.FEATURES_TYPE, 'n_train': len(ML_data),
                  'class_counts': ML_data['label'].value_counts().to_dict(),
                  'sklearn_ver': __import__('sklearn').__version__}

        ## training set archived alongside the model — compound/smiles/label only, features
        ## are reconstructible from the smiles by MLTrail's featurizer
        train_set = ML_data[['compound', 'label']].merge(
            data.serac_df[['compound', 'smiles']].drop_duplicates('compound'), on='compound')

        ## version the existing entry of this name rather than spawning a duplicate id
        hit = params.registry.list().pipe(lambda d: d.loc[d['experiment_name'] == name, 'id'])
        model_id = int(hit.iloc[0]) if len(hit) else None
        new_id = params.registry.add(
            model_id=model_id, overwrite=overwrite and model_id is not None, model=bundle,
            experiment_name=name, experiment_measure='proteomics_activity',
            unit='n_proteins_down', model_type='single_task_classification',
            framework='sklearn', features_type=params.FEATURES_TYPE,
            training_set=train_set, smiles_column='smiles',
            compound_id_column='compound', label_column='label',
            metrics={'roc_auc': round(cv['roc_auc'], 3), 'pr_auc': round(cv['pr_auc'], 3)})
        action = ('overwrote latest version of' if (overwrite and model_id) else
                  'new version of' if model_id else 'registered')
        print(f'> {action} MLTrail model id={new_id} "{name}" '
              f'({len(train_set):,} training rows archived)')
        return cv, new_id

    def run_gene_screen(self, data, params, genes, out_path=None, model_params=None, folds=5,
                        min_compounds=20, n_processes=None, n_jobs_per_process=8, seed=0, resume=True):
        """
        -Run the single-gene logfc regression over a list of genes, in parallel and resumable.
         Per gene: label = mean over compounds of logfc clipped at 0, features = every MF_features
         column, cm2rm compounds removed, 5-fold CV via ML_Reg.run_K_Fold_Xval_Regression — i.e.
         the single-gene notebook cell, unchanged. Labels are aggregated ONCE for all genes here
         (one pass over df_raw) so the workers never touch the 40M-row frame.
        -Each finished gene is appended + flushed to out_path, and genes already in that file are
         skipped, so an interrupted run resumes where it stopped. Delete the file to force a redo.
        param class data: DATA instance (df_raw, MF_features, cm2rm)
        param class params: PARAMS instance (GENE_SAR_OUT)
        param list genes: gene symbols to screen (deduplicated, order kept)
        param str out_path: results csv; defaults to params.GENE_SAR_OUT
        param dict model_params: RandomForestRegressor kwargs (default GENE_SCREEN_RF)
        param int min_compounds: cohorts below this get a NaN row instead of a crash in KFold
        param int n_processes: genes screened concurrently; None -> 8 (measured optimum, see comment)
        param int n_jobs_per_process: threads for the RF inside each worker
        return None: results land in self.gene_metrics (also written to out_path)
        """
        out_path = out_path or params.GENE_SAR_OUT
        model_params = model_params or GENE_SCREEN_RF
        genes = list(dict.fromkeys(genes))   # dedupe, keep order

        done = set()
        if resume and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            done = set(pd.read_csv(out_path)['gene'])
            print(f'> resuming: {len(done):,} genes already in {out_path}')
        todo = [g for g in genes if g not in done]
        if not todo:
            self.gene_metrics = pd.read_csv(out_path)
            print(f'> nothing to screen — {len(self.gene_metrics):,} rows already in {out_path}')
            return
        # 8 workers measured fastest here: throughput saturates on the memory subsystem, not on
        # cores, so filling all 256 threads is SLOWER (32x8 = 6.95 s/gene vs 8x8 = 5.81).
        n_processes = n_processes or min(8, max(1, (os.cpu_count() or 8) // n_jobs_per_process))
        n_processes = min(n_processes, len(todo))
        print(f'> {len(todo):,} / {len(genes):,} genes to screen | {n_processes} workers x '
              f'{n_jobs_per_process}-thread RF | features {data.MF_features.shape[1] - 1} ({params.FEATURES_TYPE})')

        ## per-(gene, compound) labels for every requested gene in ONE pass over df_raw
        t0 = time.time()
        sub = data.df_raw.loc[data.df_raw['genes'].isin(set(todo)), ['genes', 'compound', 'logfc']].copy()
        sub['label'] = sub['logfc'].clip(upper=0.0)   # keep only down-regulation; stabilised at 0
        labels = sub.groupby(['genes', 'compound'], observed=True)['label'].mean().reset_index()
        del sub; gc.collect()
        ## ...and the per-gene row selection + shuffle, so the workers only ever index a shared matrix
        X, tasks, n_feat = _build_screen_tasks(labels, data.MF_features, set(data.cm2rm), todo)
        del labels; gc.collect()
        print(f'> prepared {len(tasks):,} genes in {time.time() - t0:.1f}s | shared feature matrix '
              f'{X.shape} float32 ({X.nbytes / 1e6:.0f} MB)')

        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        write_header = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
        state = dict(X=X, tasks=tasks, model_params=model_params, folds=folds,
                     seed=seed, n_jobs=n_jobs_per_process, min_compounds=min_compounds)
        with open(out_path, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=GENE_SCREEN_COLS)
            if write_header:
                writer.writeheader()
            ctx = mp.get_context('fork')   # fork so X / tasks are inherited, not pickled
            with ctx.Pool(processes=n_processes, initializer=_init_screen_worker, initargs=(state,)) as pool:
                for row in tqdm(pool.imap_unordered(_screen_gene, todo), total=len(todo), desc='genes'):
                    writer.writerow(row)
                    fh.flush()   # a crash loses at most the genes still in flight

        self.gene_metrics = pd.read_csv(out_path)
        _thin = self.gene_metrics['R2'].isna().sum()
        print(f'> screened {len(todo):,} genes -> {out_path} ({len(self.gene_metrics):,} rows total'
              + (f', {_thin} with < {min_compounds} compounds)' if _thin else ')'))


# ~~~~~~~~~~~~~~~~~~~~~~
# MAIN
# ~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build/update the MS activity classification models.")
    ap.add_argument('--config', default='config/config.yaml', help="path to the YAML config")
    ap.add_argument('--output_dir', default='output/ML', help="base dir for the models + figures")
    ap.add_argument('--genes', default=None,
                    help="run the per-gene screen on these genes: a comma-separated list, a file with "
                         "one gene per line, or 'all' / 'top:<N>' for the genes with the most compounds")
    ap.add_argument('--cm2rm', default=None,
                    help="comma-separated blacklist parts (default: config CM2RM_PARTS): "
                         "contaminants | control_compounds | fbx_independent")
    ap.add_argument('--n_processes', type=int, default=None, help="genes screened concurrently (default: 8)")
    ap.add_argument('--n_jobs', type=int, default=8, help="RF threads inside each worker")
    ap.add_argument('--min_compounds', type=int, default=20,
                    help="skip the CV for genes with fewer compounds than this (NaN metrics row, n kept)")
    ap.add_argument('--save_non_silent', action='store_true',
                    help="train the non-silent (ndown > 0) classifier and register it in MLTrail")
    ap.add_argument('--save_single_low', action='store_true',
                    help="train the single/low (1 <= ndown <= 12) classifier and register it in MLTrail")
    ap.add_argument('--overwrite', action='store_true',
                    help="--save_*: reset the latest version in place instead of appending one")
    args = ap.parse_args()

    ## params:
    params = PARAMS(args.config)
    params.load_params()
    params.setup_dropbox()
    params.load_registry()

    ## data:
    data = DATA()
    data.load_chemical_lib_df(params)
    data.get_contaminants_and_controls(params, parts=tuple(args.cm2rm.split(',')) if args.cm2rm else None)
    data.load_proteomics_data(params)
    data.compute_features(params)

    ## output:
    output = OUTPUT()

    if args.genes:
        if os.path.exists(args.genes):
            genes = [g.strip() for g in open(args.genes) if g.strip()]
        elif args.genes == 'all' or args.genes.startswith('top:'):
            counts = data.df_raw.groupby('genes', observed=True)['compound'].nunique().sort_values(ascending=False)
            genes = list(counts.index if args.genes == 'all' else counts.index[:int(args.genes.split(':')[1])])
        else:
            genes = [g.strip() for g in args.genes.split(',') if g.strip()]
        output.run_gene_screen(data, params, genes, min_compounds=args.min_compounds,
                               n_processes=args.n_processes, n_jobs_per_process=args.n_jobs)

    if args.save_non_silent:
        output.save_non_silent_model(data, params, overwrite=args.overwrite)

    if args.save_single_low:
        output.save_single_low_model(data, params, overwrite=args.overwrite)
