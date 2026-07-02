"""
General-purpose helpers shared across the MS_ML project.

Currently contains:
  * OpenTargets target-disease association helpers (GraphQL API + local-bulk
    parquet backend). Moved here from Statistics_tools.py.
"""

import os
import numpy as np
import pandas as pd

from tqdm import tqdm


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OpenTargets — target-disease association scores
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_OT_URL = 'https://api.platform.opentargets.org/api/v4/graphql'


def _ot_session():
    """Build a requests.Session with retries — mirrors the pattern used in the
    PubChem cell. Cached on the function attribute so repeated calls re-use it."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    if not hasattr(_ot_session, '_s'):
        s = requests.Session()
        s.mount('https://', HTTPAdapter(max_retries=Retry(
            total=5, backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['POST'],
        )))
        _ot_session._s = s
    return _ot_session._s


def _ot_post(query, variables, timeout=20):
    r = _ot_session().post(_OT_URL,
                           json={'query': query, 'variables': variables},
                           timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get('errors'):
        raise RuntimeError(payload['errors'])
    return payload['data']


def _ot_resolve_target_id(gene_symbol):
    """gene symbol → Ensembl target id (None if no hit)."""
    q = '''query ($q: String!) {
      search(queryString: $q, entityNames:["target"]) { hits { id name entity } }
    }'''
    hits = _ot_post(q, {'q': gene_symbol})['search']['hits']
    return hits[0]['id'] if hits else None


def _ot_get_target_diseases(ensembl_id, size=30):
    """Top-`size` associated diseases for one Ensembl target id."""
    q = '''query ($id: String!, $size: Int!) {
      target(ensemblId: $id) {
        id approvedSymbol approvedName
        associatedDiseases(page: {index: 0, size: $size}) {
          count
          rows {
            score
            datatypeScores { id score }
            disease { id name therapeuticAreas { name } }
          }
        }
      }
    }'''
    return _ot_post(q, {'id': ensembl_id, 'size': size})['target']


_OT_DATATYPES = ['genetic_association', 'genetic_literature', 'somatic_mutation',
                 'animal_model', 'rna_expression', 'affected_pathway',
                 'literature', 'known_drug']


def get_opentarget_disease_score(df, gene_col='gene', top_n=30, verbose=True,
                                  ot_root=None):
    """
    For each gene symbol in ``df[gene_col]``, return the top-N associated
    diseases with overall + per-datatype association scores, one row per
    (gene, disease) pair.

    Two modes:
      * ``ot_root=None`` (default): query OpenTargets' GraphQL API. Suitable
        for ≤ a few hundred genes. Sends only the gene symbols; no project
        data leaves.
      * ``ot_root='/path/to/opentarget'``: read from a local bulk dump,
        scaling to thousands of genes in seconds. The folder must contain
        these subdirs (downloaded from https://platform.opentargets.org/downloads):
            target/                              (Targets core)
            disease/                             (Diseases core)
            association_overall_indirect/        (Associations - indirect)
            association_by_datatype_indirect/    (Associations - indirect, by data type)

    :param df df: dataframe with a column of gene symbols (HGNC / approved-symbol).
    :param str gene_col: name of the column holding gene symbols.
    :param int top_n: number of diseases to keep per target (sorted by overall score).
    :param bool verbose: print a [skip] line for unresolved symbols.
    :param str ot_root: if set, read from local bulk dump instead of the API.

    :return df: long-format with columns
        target_symbol | target_id | target_name | disease_name | disease_id |
        overall_score | genetic_association | genetic_literature | somatic_mutation |
        animal_model | rna_expression | affected_pathway | literature | known_drug |
        therapeutic_areas
    """
    if ot_root is not None:
        return _get_ot_score_local(df, gene_col, top_n, verbose, ot_root)

    # All networking helpers are local closures so this stays autoreload-safe
    # (a module-level ``_ot_session`` sometimes goes stale when superreload
    # patches in-place — see CLAUDE.md verify-changes note).
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    URL = 'https://api.platform.opentargets.org/api/v4/graphql'
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['POST'],
    )))

    def _post(q, v, timeout=20):
        r = session.post(URL, json={'query': q, 'variables': v}, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        if payload.get('errors'):
            raise RuntimeError(payload['errors'])
        return payload['data']

    def _resolve(g):
        q = '''query ($q: String!) {
          search(queryString: $q, entityNames:["target"]) { hits { id } }
        }'''
        hits = _post(q, {'q': g})['search']['hits']
        return hits[0]['id'] if hits else None

    def _diseases(eid, size):
        q = '''query ($id: String!, $size: Int!) {
          target(ensemblId: $id) {
            id approvedSymbol approvedName
            associatedDiseases(page: {index: 0, size: $size}) {
              count
              rows {
                score
                datatypeScores { id score }
                disease { id name therapeuticAreas { name } }
              }
            }
          }
        }'''
        return _post(q, {'id': eid, 'size': size})['target']

    genes = list(pd.Series(df[gene_col]).dropna().astype(str).unique())
    rows = []
    for gene in tqdm(genes, desc='OpenTargets targets'):
        try:
            tid = _resolve(gene)
            if not tid:
                if verbose:
                    print(f'  [skip] no Ensembl id for {gene!r}')
                continue
            t = _diseases(tid, size=top_n)
        except Exception as e:
            if verbose:
                print(f'  [skip] {gene!r}: {type(e).__name__} {e}')
            continue
        for r in t['associatedDiseases']['rows']:
            ds = {d['id']: d['score'] for d in r['datatypeScores']}
            rows.append({
                'target_symbol':       t['approvedSymbol'],
                'target_id':           t['id'],
                'target_name':         t['approvedName'],
                'disease_name':        r['disease']['name'],
                'disease_id':          r['disease']['id'],
                'overall_score':       r['score'],
                'genetic_association': ds.get('genetic_association', 0.0),
                'somatic_mutation':    ds.get('somatic_mutation', 0.0),
                'animal_model':        ds.get('animal_model', 0.0),
                'rna_expression':      ds.get('rna_expression', 0.0),
                'affected_pathway':    ds.get('affected_pathway', 0.0),
                'literature':          ds.get('literature', 0.0),
                'known_drug':          ds.get('known_drug', 0.0),
                'therapeutic_areas':   '|'.join(ta['name'] for ta in r['disease']['therapeuticAreas']),
            })
    return pd.DataFrame(rows)


def _get_ot_score_local(df, gene_col, top_n, verbose, ot_root):
    """
    Local-bulk backend for :func:`get_opentarget_disease_score`. Reads parquet
    files with predicate pushdown so we only pull rows for the user's genes —
    even on the full 4.7 GB association dump it returns in a few seconds.
    """
    genes = list(pd.Series(df[gene_col]).dropna().astype(str).unique())
    if verbose:
        print(f'> local OT lookup for {len(genes):,} unique gene symbols')

    # 1) Symbol → Ensembl id via the Targets core dataset (push the symbol filter into parquet)
    targets_meta = pd.read_parquet(
        os.path.join(ot_root, 'target'),
        columns=['id', 'approvedSymbol', 'approvedName'],
        filters=[('approvedSymbol', 'in', genes)],
    ).rename(columns={'id': 'target_id', 'approvedSymbol': 'target_symbol',
                      'approvedName': 'target_name'})
    if verbose:
        missing = sorted(set(genes) - set(targets_meta['target_symbol']))
        print(f'  matched {len(targets_meta):,} / {len(genes):,} symbols'
              + (f'   (e.g. unmatched: {missing[:5]} …)' if missing else ''))
    if targets_meta.empty:
        return pd.DataFrame()
    target_ids = list(targets_meta['target_id'])

    # 2) Overall associations — filter by targetId at parquet read time
    overall = pd.read_parquet(
        os.path.join(ot_root, 'association_overall_indirect'),
        columns=['diseaseId', 'targetId', 'associationScore', 'evidenceCount'],
        filters=[('targetId', 'in', target_ids)],
    ).rename(columns={'targetId': 'target_id', 'diseaseId': 'disease_id',
                      'associationScore': 'overall_score',
                      'evidenceCount': 'evidence_count'})

    # 3) Top-N diseases per target by overall score — done before the per-datatype
    #    join so the pivot only happens on the rows we'll keep.
    overall = (overall.sort_values('overall_score', ascending=False)
                      .groupby('target_id', sort=False).head(top_n))

    # 4) Per-datatype scores, filtered by (target, disease) we kept above
    keep_pairs = set(zip(overall['target_id'], overall['disease_id']))
    dt_long = pd.read_parquet(
        os.path.join(ot_root, 'association_by_datatype_indirect'),
        columns=['diseaseId', 'targetId', 'aggregationValue', 'associationScore'],
        filters=[('targetId', 'in', target_ids)],
    ).rename(columns={'targetId': 'target_id', 'diseaseId': 'disease_id',
                      'aggregationValue': 'datatype', 'associationScore': 'score'})
    dt_long = dt_long[
        list(map(lambda tup: tup in keep_pairs,
                 zip(dt_long['target_id'], dt_long['disease_id'])))
    ]
    dt_wide = (dt_long.pivot_table(index=['target_id', 'disease_id'],
                                    columns='datatype', values='score',
                                    fill_value=0.0)
                      .reset_index())

    # 5) Disease metadata (name + therapeutic-area EFO ids)
    disease_meta = pd.read_parquet(
        os.path.join(ot_root, 'disease'),
        columns=['id', 'name', 'therapeuticAreas'],
        filters=[('id', 'in', list(overall['disease_id'].unique()))],
    ).rename(columns={'id': 'disease_id', 'name': 'disease_name'})
    # Map therapeutic-area EFO ids → human-readable names within the same dataset
    ta_meta = pd.read_parquet(
        os.path.join(ot_root, 'disease'), columns=['id', 'name'],
    )
    id2name = dict(zip(ta_meta['id'], ta_meta['name']))
    def _ta_to_str(lst):
        # `lst` can be None, list, or numpy array (truthy-check is ambiguous on np.array)
        if lst is None:
            return ''
        return '|'.join(id2name.get(x, x) for x in lst)
    disease_meta['therapeutic_areas'] = disease_meta['therapeuticAreas'].apply(_ta_to_str)
    disease_meta = disease_meta.drop('therapeuticAreas', axis=1)

    # 6) Stitch
    out = (overall
           .merge(dt_wide, on=['target_id', 'disease_id'], how='left')
           .merge(targets_meta, on='target_id', how='left')
           .merge(disease_meta, on='disease_id', how='left'))

    # 7) Make sure every datatype column exists, in the canonical order, then reorder
    for c in _OT_DATATYPES:
        if c not in out.columns:
            out[c] = 0.0
    cols = (['target_symbol', 'target_id', 'target_name',
             'disease_name', 'disease_id', 'overall_score']
            + _OT_DATATYPES + ['therapeutic_areas'])
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Proteomics data loading + MS-recency filtering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_proteomics_data(
    raw_proteomics_path,
    clean_proteomics_path,
    drop_plates=('Plate12', 'Plate15', 'Plate23'),
    mode='serac',
    collections=('AJ', 'AK'),
    verbose=True,
):
    """
    Load a raw proteomics table + its metadata table, then filter the raw rows
    to the latest/selected screen per compound with the noisy plates removed.
    Returns ``(df_raw, MS)``.

    The raw side is identical across data tranches; only the metadata recipe
    differs, switched by ``mode``:

      * ``mode='serac'`` (default) — clean CDD MS export. Drop ``CDD Number``,
        keep ``Source == 'SERAC'``, parse the screen date, and for compounds
        screened more than once keep only the latest-dated row (one per
        ``Molecule Name``). Join key:
        ``MSData - Proteomics activities: Molecule-Batch ID``.
      * ``mode='cddvault'`` — CDD Vault export. Rename ``SMILES`` → ``smiles``
        and keep only the ``collections`` of interest (drops PROTACs).
        Join key: ``Batch Molecule-Batch ID``.

    Common to both: read the raw table, split ``MoleculeBatchID``
    (``SRB-0000385-001``) into ``compound`` + ``batch``, drop ``drop_plates``
    by ``MSPlate`` (a GLOBAL rule — never per-gene), then keep only the
    molecule-batches present in the metadata table.

    :param str raw_proteomics_path: raw per-(compound, gene) CSV; needs
        ``MoleculeBatchID`` and ``MSPlate`` columns.
    :param str clean_proteomics_path: metadata CSV (schema depends on ``mode``).
    :param list drop_plates: plate IDs removed from the raw table before filtering.
    :param str mode: ``'serac'`` or ``'cddvault'`` — selects the metadata recipe.
    :param list collections: ``cddvault`` mode only — Collections to keep.
    :param bool verbose: print aggregate diagnostics (counts, shapes) — no
        per-compound rows are printed.

    :return: ``(df_raw, MS)`` — filtered raw table and the metadata table.
    """
    # --- Raw table (common to all tranches) ---
    df_raw = pd.read_csv(raw_proteomics_path)
    if mode == 'cddvault':
        # Database export names this column 'unique'; match the earlier tranche.
        df_raw = df_raw.rename(columns={'unique': 'uniquecontrast'})
    parts = df_raw['MoleculeBatchID'].str.split('-', n=2, expand=True)
    df_raw['compound'] = parts[0] + '-' + parts[1]   # 'SRB-0000385'
    df_raw['batch']    = parts[2]                    # '001'

    # --- Metadata table (recipe depends on `mode`) ---
    if mode == 'serac':
        MS = pd.read_csv(clean_proteomics_path).drop(['CDD Number'], axis=1)
        MS = MS[MS['MSData - Proteomics activities: Source'] == 'SERAC']
        MS['MSData - Proteomics activities: Date'] = pd.to_datetime(
            MS['MSData - Proteomics activities: Date'])
        # If a compound is tested multiple times, keep only the latest date.
        MS = MS.sort_values('MSData - Proteomics activities: Date',
                            ascending=False).reset_index()
        MS = MS.groupby('Molecule Name').first().reset_index()
        batch_col = 'MSData - Proteomics activities: Molecule-Batch ID'
    elif mode == 'cddvault':
        MS = pd.read_csv(clean_proteomics_path).rename(columns={'SMILES': 'smiles'})
        # Collections filtering deliberately dropped — PROTAC removal is now done upstream
        # via the AJ/AK-filtered serac_df; the inner merge with serac_df downstream will
        # drop anything not in those collections. Lets this mode work on stripped Vault
        # exports that lack the Collections column (e.g. the 20260529 tranche).
        # The `collections` arg above is kept for backwards compat but is now unused.
        # Join-key column varies across Vault exports — auto-detect.
        _batch_candidates = ['Batch Molecule-Batch ID', 'Molecule-Batch ID',
                             'MSData - Proteomics activities: Molecule-Batch ID']
        batch_col = next((c for c in _batch_candidates if c in MS.columns), None)
        assert batch_col is not None, (
            f"no Molecule-Batch ID column found in {clean_proteomics_path}; "
            f"expected one of {_batch_candidates}")
    else:
        raise ValueError(f"mode must be 'serac' or 'cddvault', got {mode!r}")

    if verbose:
        print(f'> mode={mode} | MS rows: {len(MS):,} | join key: {batch_col}')
        if mode == 'serac':
            print('>', MS['Molecule Name'].nunique(), 'unique compounds')
            print('> Ligase(s)', list(MS['MSData - Proteomics activities: Ligase'].unique()))
            print('> Cellline', list(MS['MSData - Proteomics activities: Cell line'].unique()))
        else:
            print('> Collections kept:', list(collections))

    # --- Drop noisy plates (GLOBAL rule) + keep only molecule-batches in MS ---
    df_raw = df_raw[~df_raw['MSPlate'].isin(list(drop_plates))]
    df_raw = df_raw[df_raw['MoleculeBatchID'].isin(MS[batch_col])]
    if verbose:
        print('> df_raw dim:', df_raw.shape)

    return df_raw, MS


def keep_latest_batch_per_compound(df_raw, compound_col='compound',
                                   batch_col='batch', date_col='date',
                                   verbose=True):
    """
    Collapse a per-(compound, gene) raw table to a single screen per compound.

    Two rules, applied in priority order **within each compound**:

      1. **Latest batch wins** — if a compound was screened under more than one
         batch number (e.g. ``001`` and ``002``), keep only the rows of the
         highest batch number.
      2. **Latest date breaks ties** — if the surviving (highest) batch was
         screened on more than one date (the "same batch measured twice" case),
         keep only the rows from the most recent ``date_col``.

    All rows of the single winning ``(batch, date)`` screen are kept — every
    gene row and every plate replicate of that screen survives; replicate
    aggregation happens downstream, not here. Batch is coerced to a number for
    ranking (non-numeric batches rank below numeric ones).

    :param df df_raw: per-(compound, gene) table; needs ``compound_col``,
        ``batch_col`` and ``date_col``.
    :param str compound_col: compound id column.
    :param str batch_col: batch column (``'001'``, ``'002'``, …); coerced to
        numeric for ranking.
    :param str date_col: screen-date column (datetime-like / parseable).
    :param bool verbose: print aggregate before/after counts — no per-compound
        rows are printed.
    :return: row subset of ``df_raw`` (original columns, index reset).
    """
    d = df_raw.copy()
    # Rank keys (non-numeric batch -> -1 so a real batch always wins).
    d['_batch_n'] = pd.to_numeric(d[batch_col], errors='coerce').fillna(-1)
    d['_date'] = pd.to_datetime(d[date_col], errors='coerce')

    # Rule 1: highest batch number per compound.
    d['_win_batch'] = d.groupby(compound_col)['_batch_n'].transform('max')
    d = d[d['_batch_n'] == d['_win_batch']]

    # Rule 2: among the surviving batch, the latest date per compound.
    d['_win_date'] = d.groupby(compound_col)['_date'].transform('max')
    d = d[d['_date'] == d['_win_date']]

    out = (d.drop(columns=['_batch_n', '_date', '_win_batch', '_win_date'])
             .reset_index(drop=True))
    if verbose:
        print(f'> latest-batch/date filter: {len(df_raw):,} -> {len(out):,} rows '
              f'| {df_raw[compound_col].nunique():,} -> {out[compound_col].nunique():,} '
              f'compounds')
    return out


def collapse_ms_latest_measurement(MS, compound_col='compound', date_col='date',
                                   verbose=True):
    """
    Collapse the MS metadata table to one row per compound, keeping the **latest
    measurement** but stamping it with the compound's **earliest** screen date.

    For a compound screened across several tranches, the most recent screen is
    the one we trust (``ndown`` / ``activity`` from the latest ``date_col`` row),
    but it is attributed to the date the compound was *first* screened — so
    tranche/cohort grouping counts the compound when it first appeared.

    Example — ``SRB1`` with ``(ndown=3, 2026-06-01)`` and ``(ndown=5, 2026-06-16)``
    collapses to a single row ``(ndown=5, date=2026-06-01)``.

    All other columns (``origin``, ``activity``, …) come from the latest-dated
    row; only ``date_col`` is overwritten with the earliest date.

    :param df MS: MS metadata, one row per (compound, tranche).
    :param str compound_col: compound id column.
    :param str date_col: screen-date column (datetime-like / parseable).
    :param bool verbose: print aggregate before/after counts.
    :return: one row per compound (original columns, index reset).
    """
    d = MS.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    earliest = d.groupby(compound_col)[date_col].min()                 # first-seen date
    out = (d.sort_values(date_col)                                     # latest measurement
             .drop_duplicates(compound_col, keep='last')
             .reset_index(drop=True))
    out[date_col] = out[compound_col].map(earliest)                    # stamp earliest date
    if verbose:
        print(f'> MS collapse (latest measurement, earliest date): '
              f'{len(MS):,} -> {len(out):,} rows '
              f'| {MS[compound_col].nunique():,} compounds')
    return out


# Target schemas the FBX frames are coerced to (so a plain concat with the
# existing df_raw / MS frames lines up).
FBX_DFRAW_COLS = ['MoleculeBatchID', 'MSPlate', 'genes', 'pg', 'logfc', 'pvalue',
                  'adjpval', 'significant', 'uniquecontrast', 'compound', 'batch']
FBX_MS_COLS = ['compound', 'ndown', 'origin', 'activity', 'date']


def load_fbx_tranche(tranche_dir, *, control_compounds=(), contaminants=(),
                     drop_plate_substr=('MLN', 'KO', 'Eval'),
                     dfraw_cols=FBX_DFRAW_COLS, ms_cols=FBX_MS_COLS, verbose=True):
    """
    Format one AdvantEdge / FBX export folder into the ``df_raw`` / ``MS`` schemas
    so it can be ``pd.concat``-ed with the existing datasets. Returns
    ``(df_raw_fbx, MS_fbx)``.

    A tranche folder is named by its export date and holds three CSVs
    (``*_FBX_MEASURE``, ``*_FBX_MSSCORE``, ``*_FBX_REPORT``); only MEASURE and
    REPORT are used. The crosswalk mirrors ``MS_Interface``'s "combine df_raw &
    FBX_MEASURE": MEASURE carries the per-(gene × experiment) signal keyed by
    ``uniquecontrast``; REPORT maps ``uniquecontrast`` → ``srbnumber`` (the full
    ``MoleculeBatchID``), which splits into ``compound`` (``SRB-XXXXXXX``) +
    ``batch`` (``NNN``).

    The glob tolerates a re-export suffix (``*_FBX_MEASURE_02.csv``); the date is
    taken from the folder name's leading ``YYYYMMDD`` (so ``20260616_2`` parses to
    2026-06-16) while ``origin`` keeps the full folder name to stay distinct.

    :param str tranche_dir: one FBX export folder (named by export date).
    :param control_compounds: control compound ids to drop entirely.
    :param contaminants: contaminant compound ids to drop entirely.
    :param drop_plate_substr: drop any plate whose name contains one of these
        substrings (case-insensitive — e.g. MLN / KO / Eval conditions).
    :param list dfraw_cols: target df_raw column order.
    :param list ms_cols: target MS column order.
    :param bool verbose: print aggregate diagnostics (counts) — no per-compound rows.
    :return: ``(df_raw_fbx, MS_fbx)`` in the df_raw / MS schemas.
    """
    import glob as _glob
    date = os.path.basename(tranche_dir.rstrip('/'))                 # folder, e.g. '20260616' or '20260616_2'
    pick = lambda kind: _glob.glob(os.path.join(tranche_dir, f'*FBX_{kind}*.csv'))[0]
    measure = pd.read_csv(pick('MEASURE'),                           # drop the unused 'id' col
                          usecols=['pg', 'genes', 'uniquecontrast', 'logfc',
                                   'pvalue', 'adjpval', 'significant', 'plate'])
    report  = pd.read_csv(pick('REPORT'))

    # drop unwanted plates (substring match on the plate name, case-insensitive)
    _pat = '|'.join(drop_plate_substr)
    _mpl = measure['plate'].astype(str)
    _dropped = sorted(set(_mpl[_mpl.str.contains(_pat, case=False, na=False)]))
    measure = measure[~_mpl.str.contains(_pat, case=False, na=False)]
    report  = report[~report['plate'].astype(str).str.contains(_pat, case=False, na=False)]
    if verbose and _dropped:
        print(f'> {date}: dropped {len(_dropped)} plates matching {drop_plate_substr}: {_dropped}')

    # control + contaminant compounds to drop entirely
    _remove = set(map(str, control_compounds)) | set(map(str, contaminants))

    # uniquecontrast -> srbnumber (MoleculeBatchID) -> compound + batch
    rep = (report[['uniquecontrast', 'srbnumber']]
           .dropna(subset=['srbnumber']).drop_duplicates('uniquecontrast'))
    sp  = rep['srbnumber'].astype(str).str.split('-', n=2, expand=True)
    rep = rep.assign(MoleculeBatchID=rep['srbnumber'], compound=sp[0] + '-' + sp[1], batch=sp[2])

    # --- df_raw schema (per gene x experiment) ---
    df_raw_fbx = (measure.merge(rep[['uniquecontrast', 'MoleculeBatchID', 'compound', 'batch']],
                                on='uniquecontrast', how='left')
                  .rename(columns={'plate': 'MSPlate'}))
    n_qc = int(df_raw_fbx['MoleculeBatchID'].isna().sum())           # control/QC contrasts: no compound
    df_raw_fbx = df_raw_fbx.dropna(subset=['MoleculeBatchID'])[dfraw_cols]
    n_ctrl = int(df_raw_fbx['compound'].isin(_remove).sum())         # control + contaminant rows
    df_raw_fbx = df_raw_fbx[~df_raw_fbx['compound'].isin(_remove)]

    # --- MS schema (per-compound activity summary; representative = max nr_down) ---
    rms = report.dropna(subset=['srbnumber']).copy()
    s2  = rms['srbnumber'].astype(str).str.split('-', n=2, expand=True)
    rms['compound'] = s2[0] + '-' + s2[1]
    rms = rms[~rms['compound'].isin(_remove)]                        # drop controls + contaminants
    MS_fbx = (rms.sort_values('nr_down', ascending=False).drop_duplicates('compound', keep='first')
              .rename(columns={'nr_down': 'ndown'})
              .assign(origin='MS' + date, date=pd.to_datetime(date[:8]))[ms_cols])

    if verbose:
        print(f'> {date}: df_raw_fbx {len(df_raw_fbx):,} rows '
              f'({df_raw_fbx["compound"].nunique():,} compounds, '
              f'{df_raw_fbx["uniquecontrast"].nunique():,} experiments; '
              f'{n_qc:,} QC rows w/o compound + {n_ctrl:,} control/contaminant rows dropped) '
              f'| MS_fbx {len(MS_fbx):,} compounds')
    return df_raw_fbx, MS_fbx


def plot_activity_rate_by_tranche(MS, date_col='date', activity_col='activity',
                                  silent_label='Silent', colors=None,
                                  annotate=True, ax=None, dpi=150):
    """
    Bar plot of per-tranche MS activity rate = fraction of compounds that are
    ACTIVE (``activity != silent_label``), one bar per screening tranche, ordered
    chronologically by date.

    Expects the unified MS table, one row per compound-tranche, e.g.::

        compound      ndown  origin       activity      date
        SRB-0000385   3.0    MS20260429   Low (2-10)    2026-04-29

    Tranches are grouped by ``date`` (parsed to datetime, sorted ascending). Each
    bar shows the activity rate (%) inside and the compound count ``n`` on top.

    :param df MS: unified MS metadata (compound | ndown | origin | activity | date).
    :param str date_col: tranche-date column (parsed with ``pd.to_datetime``).
    :param str activity_col: categorical activity column; auto-falls back to the
        first column containing 'activity' if ``activity_col`` is absent.
    :param str silent_label: the inactive label (everything else counts as active).
    :param list colors: per-bar colours (extended/truncated to the #tranches).
    :param bool annotate: draw rate-inside + n-on-top labels.
    :param ax: optional matplotlib axes; created if None.
    :param int dpi: figure resolution (only used when ``ax`` is None; default 150).
    :return: ``(ax, summary)`` — axes and a per-tranche DataFrame
        (``date, n, n_active, activity_rate``).
    """
    import matplotlib.pyplot as plt

    df = MS.copy()
    if activity_col not in df.columns:
        activity_col = next((c for c in df.columns if 'activity' in c.lower()),
                            activity_col)
    df[date_col] = pd.to_datetime(df[date_col])

    summary = (df.groupby(date_col)
                 .agg(n=(activity_col, 'size'),
                      n_active=(activity_col, lambda s: int((s != silent_label).sum())))
                 .reset_index()
                 .sort_values(date_col))
    summary['activity_rate'] = summary['n_active'] / summary['n']

    labels = [d.strftime('%Y-%m-%d') for d in summary[date_col]]
    rates  = summary['activity_rate'].values
    ns     = summary['n'].values

    if colors is None:
        colors = ['#ff0051', 'pink', 'lightblue', '#0003fb', 'purple']
    colors = (list(colors) * (len(rates) // len(colors) + 1))[:len(rates)]

    if ax is None:
        _, ax = plt.subplots(dpi=dpi, figsize=(1.6 * len(rates) + 1, 5))
    bars = ax.bar(labels, rates, width=0.8, color=colors, edgecolor='black')

    # clean "nice barplot" aesthetics (despine + light horizontal grid)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('grey')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('activity rate (fraction active)')
    ax.set_title('MS activity rate by tranche')

    if annotate:
        for bar, r, n in zip(bars, rates, ns):
            x = bar.get_x() + bar.get_width() / 2
            ax.text(x, r / 2, f'{r:.1%}', ha='center', va='center', fontsize=9,
                    weight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax.text(x, r + 0.015, f'n={n:,}', ha='center', va='bottom',
                    fontsize=8, color='#333')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return ax, summary


def plot_activity_composition_over_time(
        MS, date_col='date', activity_col='activity',
        cats=('Silent', 'Single (1)', 'Low (2-10)', 'Medium (11-25)', 'High (>25)'),
        colors=None,
        silent_label='Silent', show_rate_line=True, dpi=150, ax=None):
    """
    100%-stacked area of MS activity-category composition across screening tranches
    (x = tranche date) — the "shift toward signal" view. Right-axis labels = the
    categories; each x-tick shows the date with the compound count ``n=`` beneath;
    an optional bold line tracks the activity rate (non-silent share).

    Expects the unified MS table, one row per compound-tranche::

        compound      ndown  origin       activity      date
        SRB-0000385   3.0    MS20260429   Low (2-10)    2026-04-29

    :param df MS: unified MS metadata.
    :param str date_col: tranche-date column (parsed with ``pd.to_datetime``).
    :param str activity_col: categorical activity column; auto-falls back to the
        first column containing 'activity'.
    :param cats: category order, BOTTOM → TOP of the stack.
    :param colors: per-category fill colours — a list (positional, same order as
        ``cats``, recycled if shorter) OR a dict ``{category: colour}``. ``None``
        (default) keeps the built-in earth-tone palette.
    :param str silent_label: inactive label (used for the activity-rate line).
    :param bool show_rate_line: overlay the non-silent activity-rate line.
    :param int dpi: figure resolution (only used when ``ax`` is None; default 150).
    :param ax: optional matplotlib axes.
    :return: ``(ax, summary)`` — axes and a per-tranche DataFrame indexed by date
        with an ``n`` column and one share column per category.
    """
    import matplotlib.pyplot as plt
    _DEFAULT_COLORS = ('#d8cdbf', '#c9b79a', '#88a06a', '#d99a3a', '#b8412f')
    cats = list(cats)
    if colors is None:
        colors = list(_DEFAULT_COLORS)
    elif isinstance(colors, dict):
        colors = [colors.get(c, '#cccccc') for c in cats]      # map by category; grey fallback
    else:
        colors = list(colors)
    colors = [colors[i % len(colors)] for i in range(len(cats))]   # match length / recycle

    df = MS.copy()
    if activity_col not in df.columns:
        activity_col = next((c for c in df.columns if 'activity' in c.lower()),
                            activity_col)
    df[date_col] = pd.to_datetime(df[date_col])

    grp   = df.groupby(date_col)
    dates = sorted(grp.groups)
    ns    = [len(grp.get_group(d)) for d in dates]
    shares = np.zeros((len(cats), len(dates)))
    for j, d in enumerate(dates):
        vc  = grp.get_group(d)[activity_col].value_counts(normalize=True)
        col = np.array([vc.get(c, 0.0) for c in cats])
        shares[:, j] = col / col.sum() if col.sum() else col

    if ax is None:
        _, ax = plt.subplots(figsize=(1.8 * len(dates) + 3, 5.5), dpi=dpi)
    ax.stackplot(dates, shares, colors=colors, labels=cats,
                 edgecolor='white', linewidth=0.6)

    if show_rate_line:
        rate = 1 - shares[cats.index(silent_label)]
        ax.plot(dates, rate, color='#5b2a86', lw=3.5, marker='o', ms=5,
                label='activity rate (non-silent)')

    ax.set_ylim(0, 1); ax.set_xlim(min(dates), max(dates))
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_ylabel('share'); ax.set_xlabel('MS tranche')
    ax.set_xticks(dates)
    ax.set_xticklabels([f'{pd.Timestamp(d):%Y-%m-%d}\nn={n:,}'
                        for d, n in zip(dates, ns)])
    ax.set_title('A shift toward signal — MS activity composition over tranches',
                 fontsize=13)

    # right-axis category labels at each band's mid-height in the LAST tranche
    last = shares[:, -1]; mids = np.cumsum(last) - last / 2
    axr = ax.twinx(); axr.set_ylim(0, 1); axr.set_yticks(mids)
    axr.set_yticklabels([c.upper() for c in cats]); axr.tick_params(length=0)

    ax.legend(loc='upper left', bbox_to_anchor=(1.28, 1.0), frameon=False, fontsize=8)

    summary = pd.DataFrame(shares.T, index=[pd.Timestamp(d) for d in dates], columns=cats)
    summary.insert(0, 'n', ns); summary.index.name = date_col
    return ax, summary


def plot_activity_composition_bars(
        MS, date_col='date', activity_col='activity',
        cats=('Silent', 'Single (1)', 'Low (2-10)', 'Medium (11-25)', 'High (>25)'),
        colors=None, silent_label='Silent',
        annotate=True, min_pct_label=3.0, min_label_h_frac=0.03,
        show_n=True, dpi=150, ax=None):
    """
    Stacked **vertical bars** of MS activity composition, one bar per screening
    tranche. Unlike :func:`plot_activity_composition_over_time` (which normalises
    every tranche to 100 % and hides how many compounds each holds), here the
    **bar height is the compound count** ``n`` — so the absolute scale is visible
    — while each activity segment is annotated with its **within-tranche
    proportion** (per-segment counts live in the returned ``summary``). Same
    categories / colour palette as the area view.

    Expects the unified MS table, one row per compound-tranche::

        compound      ndown  origin       activity      date
        SRB-0000385   3.0    MS20260429   Low (2-10)    2026-04-29

    :param df MS: unified MS metadata.
    :param str date_col: tranche-date column (parsed with ``pd.to_datetime``).
    :param str activity_col: categorical activity column; auto-falls back to the
        first column containing 'activity'.
    :param cats: category order, BOTTOM → TOP of each bar.
    :param colors: per-category colours — a positional list (same order as
        ``cats``, recycled if shorter) OR a dict ``{category: colour}``. ``None``
        keeps the built-in palette (shared with the area view).
    :param str silent_label: inactive label (only used to label the activity rate).
    :param bool annotate: write each segment's proportion (%) inside it.
    :param float min_pct_label: skip the in-segment label below this share (%) to
        avoid clutter on thin slices.
    :param float min_label_h_frac: also skip a label when its segment is shorter
        than this fraction of the y-axis — i.e. too thin to fit text without
        overlapping (small tranches); those counts stay in ``summary``.
    :param bool show_n: print the non-silent activity rate + ``n=`` atop each bar.
    :param int dpi: figure resolution (only used when ``ax`` is None; default 150).
    :param ax: optional matplotlib axes.
    :return: ``(ax, summary)`` — axes and a per-tranche DataFrame indexed by date
        with an ``n`` column and one **count** column per category.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    _DEFAULT_COLORS = ('#d8cdbf', '#c9b79a', '#88a06a', '#d99a3a', '#b8412f')
    cats = list(cats)
    if colors is None:
        colors = list(_DEFAULT_COLORS)
    elif isinstance(colors, dict):
        colors = [colors.get(c, '#cccccc') for c in cats]      # map by category; grey fallback
    else:
        colors = list(colors)
    colors = [colors[i % len(colors)] for i in range(len(cats))]   # match length / recycle

    df = MS.copy()
    if activity_col not in df.columns:
        activity_col = next((c for c in df.columns if 'activity' in c.lower()),
                            activity_col)
    df[date_col] = pd.to_datetime(df[date_col])

    grp   = df.groupby(date_col)
    dates = sorted(grp.groups)
    ns    = np.array([len(grp.get_group(d)) for d in dates])
    counts = np.zeros((len(cats), len(dates)))
    for j, d in enumerate(dates):
        vc = grp.get_group(d)[activity_col].value_counts()
        counts[:, j] = [vc.get(c, 0) for c in cats]
    # within-tranche shares (column-normalised counts) — for the in-bar labels
    col_tot = counts.sum(axis=0)
    shares  = np.divide(counts, col_tot, out=np.zeros_like(counts),
                        where=col_tot > 0)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 * len(dates) + 3, 5.5), dpi=dpi)
    x = np.arange(len(dates))
    y_top = ns.max() * 1.12                                    # axis top; sets the fit threshold
    bottom = np.zeros(len(dates))
    for i, c in enumerate(cats):
        ax.bar(x, counts[i], bottom=bottom, color=colors[i], label=c,
               edgecolor='white', linewidth=0.6, width=0.8)
        if annotate:
            for j in range(len(dates)):
                # label only slices that clear the share floor AND are tall enough to fit text
                if (shares[i, j] * 100 >= min_pct_label
                        and counts[i, j] >= min_label_h_frac * y_top):
                    ax.text(x[j], bottom[j] + counts[i, j] / 2,
                            f'{shares[i, j]:.0%}',
                            ha='center', va='center', fontsize=8, color='black',
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        bottom += counts[i]

    if show_n:
        rate = 1 - shares[cats.index(silent_label)]            # non-silent share per tranche
        for j in range(len(dates)):
            ax.text(x[j], ns[j], f'{rate[j]:.0%} active\nn={ns[j]:,}',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_ylim(0, y_top)
    ax.set_ylabel('compounds (n)'); ax.set_xlabel('MS tranche')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{pd.Timestamp(d):%Y-%m-%d}' for d in dates])
    ax.set_title('MS activity composition by tranche — bar height = compound count',
                 fontsize=13)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    # legend top→bottom = stack top→bottom (High first), so it reads like the bars
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc='upper left', bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8)

    summary = pd.DataFrame(counts.T, index=[pd.Timestamp(d) for d in dates], columns=cats)
    summary.insert(0, 'n', ns); summary.index.name = date_col
    return ax, summary


def plot_activity_area_absolute(
        MS, date_col='date', activity_col='activity',
        cats=('Silent', 'Single (1)', 'Low (2-10)', 'Medium (11-25)', 'High (>25)'),
        colors=None, cumulative=True, annotate_total=True,
        show_rate_line=True, silent_label='Silent', rate_color='#1d3557',
        dpi=150, ax=None):
    """
    **Absolute** (count, not 100%-normalised) stacked area of MS activity
    composition across tranches — the "growing library" view. With
    ``cumulative=True`` (default) each band is the running count of compounds
    screened up to a tranche, so the stack grows monotonically; a dotted
    **TOTAL** boundary with a marker per tranche traces the height and the total
    is annotated at the first and last tranche. Styled after the editorial
    template (cream ground, earth palette, horizontal gridlines, bottom legend
    with a dotted TOTAL swatch).

    Expects the unified MS table, one row per compound-tranche (see
    :func:`plot_activity_composition_over_time`). With the upstream
    ``collapse_ms_latest_measurement`` each compound sits in its first-seen
    tranche, so the cumulative curve is the library size over time.

    :param df MS: unified MS metadata.
    :param str date_col: tranche-date column (parsed with ``pd.to_datetime``).
    :param str activity_col: categorical activity column; auto-falls back to the
        first column containing 'activity'.
    :param cats: category order, BOTTOM → TOP of the stack.
    :param colors: per-category colours — positional list OR ``{category: colour}``
        dict OR ``None`` for the built-in earth-tone template palette.
    :param bool cumulative: stack the running total across tranches (template
        look); ``False`` plots each tranche's own counts.
    :param bool annotate_total: label the total at the first/last tranche.
    :param bool show_rate_line: overlay the per-tranche non-silent activity rate
        on a right-hand 0–100% axis (same definition as the area view; computed
        per tranche, not cumulatively, so the trend isn't flattened by the first
        large tranche).
    :param str silent_label: inactive label (the rate is the non-silent share).
    :param str rate_color: colour of the activity-rate line / right axis.
    :param int dpi: figure resolution (only used when ``ax`` is None; default 150).
    :param ax: optional matplotlib axes.
    :return: ``(ax, summary)`` — axes and a per-tranche DataFrame indexed by date
        with an ``n`` (= column total, cumulative if set) column and one count
        column per category.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _TEMPLATE_COLORS = ('#b0492f', '#7f9b6b', '#5c5147', '#c9bca0', '#d99a3a')
    _BG, _TOTAL_C, _GRID = 'white', '#2b2b2b', '#e6e6e6'
    cats = list(cats)
    if colors is None:
        colors = list(_TEMPLATE_COLORS)
    elif isinstance(colors, dict):
        colors = [colors.get(c, '#cccccc') for c in cats]
    else:
        colors = list(colors)
    colors = [colors[i % len(colors)] for i in range(len(cats))]

    df = MS.copy()
    if activity_col not in df.columns:
        activity_col = next((c for c in df.columns if 'activity' in c.lower()),
                            activity_col)
    df[date_col] = pd.to_datetime(df[date_col])

    grp   = df.groupby(date_col)
    dates = sorted(grp.groups)
    counts = np.zeros((len(cats), len(dates)))
    for j, d in enumerate(dates):
        vc = grp.get_group(d)[activity_col].value_counts()
        counts[:, j] = [vc.get(c, 0) for c in cats]
    per_tranche = counts.copy()                                # raw per-tranche, for the rate line
    if cumulative:
        counts = counts.cumsum(axis=1)                         # running library size per band
    totals = counts.sum(axis=0)
    x = np.arange(len(dates))

    if ax is None:
        _, ax = plt.subplots(figsize=(1.3 * len(dates) + 4, 5.5), dpi=dpi)
    fig = ax.get_figure()
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=_GRID, linewidth=1.0)

    ax.stackplot(x, counts, colors=colors, labels=cats,
                 edgecolor='white', linewidth=0.7)
    # dotted TOTAL boundary + a marker per tranche
    ax.plot(x, totals, color=_TOTAL_C, lw=1.4, linestyle=(0, (1, 1)),
            marker='o', ms=4, mfc=_TOTAL_C, mec=_TOTAL_C, zorder=5)
    if annotate_total:
        # tuck totals just below their markers so they clear the rate-line labels
        for j in (0, len(dates) - 1):
            ax.annotate(f'{int(totals[j]):,}', (x[j], totals[j]),
                        textcoords='offset points', xytext=(0, -13), va='top',
                        ha='center', fontsize=11, fontweight='bold', color=_TOTAL_C,
                        path_effects=[pe.withStroke(linewidth=3, foreground=_BG)])

    ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, totals.max() * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{pd.Timestamp(d):%Y-%m-%d}' for d in dates])
    ax.set_ylabel('compounds (cumulative n)' if cumulative else 'compounds (n)')
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot([0, 1], [1, 1], transform=ax.transAxes, color='black', lw=0.8,
            clip_on=False, zorder=6)                            # editorial top rule

    # per-tranche non-silent activity rate on a right-hand 0–100% axis
    if show_rate_line:
        pt_tot = per_tranche.sum(axis=0)
        rate = 1 - np.divide(per_tranche[cats.index(silent_label)], pt_tot,
                             out=np.zeros_like(pt_tot), where=pt_tot > 0)
        axr = ax.twinx()
        axr.set_ylim(0, 1); axr.set_xlim(ax.get_xlim())
        axr.plot(x, rate, color=rate_color, lw=3, marker='o', ms=5, zorder=7)
        for j in range(len(dates)):                            # label each dot with its rate
            axr.annotate(f'{rate[j]:.0%}', (x[j], rate[j]),
                         textcoords='offset points', xytext=(0, 9),
                         ha='center', fontsize=9, fontweight='bold',
                         color=rate_color, zorder=8,
                         path_effects=[pe.withStroke(linewidth=3, foreground=_BG)])
        axr.set_yticks([0, .25, .5, .75, 1])
        axr.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        axr.set_ylabel('per tranche activity rate (non-silent)', color=rate_color)
        axr.tick_params(axis='y', colors=rate_color, length=0)
        for s in axr.spines.values():
            s.set_visible(False)

    # bottom legend: category swatches + dotted TOTAL box + the rate line
    handles = [Patch(facecolor=colors[i], label=c.upper()) for i, c in enumerate(cats)]
    handles.append(Line2D([0], [0], marker='s', markersize=10, linestyle='none',
                          markerfacecolor='none', markeredgecolor=_TOTAL_C,
                          label='TOTAL'))
    if show_rate_line:
        handles.append(Line2D([0], [0], color=rate_color, lw=3, marker='o', ms=5,
                              label='activity rate (non-silent)'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=len(cats) + 2, frameon=False, fontsize=8, handlelength=1.4)

    summary = pd.DataFrame(counts.T, index=[pd.Timestamp(d) for d in dates], columns=cats)
    summary.insert(0, 'n', totals.astype(int)); summary.index.name = date_col
    return ax, summary


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3D target-prioritisation scatter (R² × overall_score × MCS fold-enrichment)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_HOVER_INJECT = '''
<style>
  /* fill the viewport on the standalone HTML so the plot isn't a small
     top-left box. Plotly writes inline width/height on the graph div, so
     we override with !important. */
  html, body { height: 100%; margin: 0; padding: 0; background: white; }
  body { display: flex; align-items: center; justify-content: center; }
  .plotly-graph-div, .js-plotly-plot {
    width: 96vw !important; height: 94vh !important; margin: 0 auto !important;
  }
  #hover-img { position: fixed; top: 12px; right: 12px; z-index: 9999;
               background: white; border: 1px solid #bbb; padding: 6px;
               border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
               display: none; font: 11px sans-serif; color: #333;
               max-height: 92vh; overflow-y: auto; max-width: 96vw;
               user-select: text; }
  /* Pinned state — slightly bolder border so you can tell it's "stuck" */
  #hover-img.pinned { border-color: #1D3557; border-width: 2px; padding: 5px;
                      box-shadow: 0 4px 14px rgba(0,0,0,0.25); }
  #hover-img .row { display: flex; flex-direction: row; gap: 6px;
                    align-items: flex-start; flex-wrap: wrap; }
  #hover-img .cell { display: flex; flex-direction: column; align-items: center;
                     border: 1px solid #eee; border-radius: 4px; padding: 3px; }
  #hover-img .cell img { display: block; width: 170px; height: 110px;
                         object-fit: contain;
                         user-select: none; -webkit-user-drag: none; pointer-events: none; }
  #hover-img .cell .cap { padding-top: 2px; max-width: 170px; word-wrap: break-word;
                          text-align: center; line-height: 1.25;
                          user-select: text; cursor: text; }
  /* Triple-click selects just the compound id, easy copy/paste */
  #hover-img .cell .cap b { user-select: all; }
  #hover-img .header { display: flex; align-items: center; gap: 8px;
                       padding-bottom: 4px; }
  #hover-img .gene { font-weight: 600; text-align: left; user-select: text; }
  #hover-img .meta { color: #555; font-size: 10px; font-family: ui-monospace, monospace;
                     user-select: text; flex: 1; }
  #hover-img .hint { color: #999; font-size: 10px; font-style: italic; }
  #hover-img.pinned .hint { display: none; }
  #hover-img .close { display: none; cursor: pointer; font-size: 16px;
                      color: #888; padding: 0 6px; border-radius: 3px;
                      user-select: none; line-height: 1; }
  #hover-img.pinned .close { display: inline-block; }
  #hover-img .close:hover { background: #eee; color: #333; }
  /* Volcano panel — only when pinned, shown on cell-hover via JS. */
  #hover-img .volcano { display: none; margin-top: 6px; text-align: center; }
  #hover-img .volcano .vlabel { font-size: 10px; color: #555; margin-bottom: 2px; }
  #hover-img .volcano img { max-width: 100%; height: auto;
                            border: 1px solid #eee; border-radius: 4px; }

  /* Per-gene patents panel — pinned immediately to the LEFT of the compound
     panel (#hover-img) with an 8px gap. The exact horizontal position is set
     by JS after each render so it tracks the compound panel's actual width.
     Top/right here are fallbacks before JS runs. Populated from the global
     `window.__GENE_PATENTS__` lookup built by plot_target_3d. */
  #hover-patents {
    position: fixed; top: 12px; right: 660px; z-index: 9999;
    background: white; border: 1px solid #bbb; border-radius: 6px;
    padding: 6px 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font: 11px sans-serif; color: #333; max-width: 320px;
    max-height: 92vh; overflow-y: auto; user-select: text;
    display: none;
  }
  #hover-patents.pinned { border-color: #1D3557; border-width: 2px; padding: 5px 7px;
                          box-shadow: 0 4px 14px rgba(0,0,0,0.25); }
  #hover-patents .pat-header { display: flex; align-items: baseline; gap: 6px;
                                font-weight: 700; padding-bottom: 4px;
                                border-bottom: 1px solid #eee; margin-bottom: 4px; }
  #hover-patents .pat-gene   { font-size: 12px; }
  #hover-patents .pat-depmap { font-size: 10px; color: #1D3557;
                                text-decoration: none; }
  #hover-patents .pat-depmap:hover { text-decoration: underline; }
  #hover-patents .pat-table  { border-collapse: collapse; width: 100%;
                                font-size: 11px; }
  #hover-patents .pat-table td { padding: 2px 4px; vertical-align: top; }
  #hover-patents .pat-table tr:nth-child(even) td { background: #f8f8f8; }
  #hover-patents .pat-empty  { color: #999; font-style: italic; padding: 4px 0; }

  /* Axis-legend panel — fixed bottom-left, short labels with `title=` tooltips
     for the full per-axis explanation. (Plotly 3D axis titles live inside the
     WebGL canvas and don't support native HTML tooltips.) */
  #axis-legend {
    position: fixed; bottom: 12px; left: 12px; z-index: 9998;
    background: white; border: 1px solid #bbb; padding: 6px 8px;
    border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    font: 11px sans-serif; color: #333; max-width: 360px;
    user-select: text;
  }
  #axis-legend .title { font-weight: 700; padding-bottom: 3px; }
  #axis-legend .ax { display: block; padding: 1px 0; cursor: help; }
  #axis-legend .ax b  { display: inline-block; min-width: 1.2em; color: #555; }
  #axis-legend .ax .lab { font-weight: 600; }
  #axis-legend .ax:hover { background: #f3f3f3; border-radius: 3px; }
</style>
<div id="hover-img">
  <div class="header">
    <span class="gene" id="hover-img-gene"></span>
    <span class="meta" id="hover-img-meta"></span>
    <span class="hint">hover → click dot to pin → hover a compound for its volcano</span>
    <span class="close" id="hover-img-close" title="Close (Esc)">×</span>
  </div>
  <div class="row" id="hover-img-row"></div>
  <div class="volcano" id="hover-img-volcano">
    <div class="vlabel" id="hover-img-volcano-label"></div>
    <img id="hover-img-volcano-img" alt="volcano"/>
  </div>
</div>
<div id="hover-patents"></div>
<div id="axis-legend">
  <div class="title">ⓘ Axis legend</div>
  <span class="ax" title="5-fold cross-validated squared Pearson correlation between predicted and observed per-compound logfc. Higher = chemistry features (Morgan FP + physchem + MACCS + AtomPair) explain more of the activity variance for this gene. Computed by python/compute_R2_for_all_genes.py with the H236 production RF (n=200, depth=20).">
    <b>X</b> <span class="lab">SAR predictability (R²)</span>
  </span>
  <span class="ax" title="OpenTargets target–disease association score (max across the priority disease franchises listed in cell d3fe884f). Higher = more clinical/literature support for the gene as a therapeutic target.">
    <b>Y</b> <span class="lab">OpenTargets overall_score</span>
  </span>
  <span class="ax" title="MCS scaffold enrichment: Fisher odds ratio for the consensus scaffold of the top-K most-active compounds vs the rest. Z is log-scaled. High fold = a clear chemotype dominates the actives — actionable for chemistry expansion.">
    <b>Z</b> <span class="lab">MCS fold-enrichment</span>
  </span>
</div>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    var box  = document.getElementById("hover-img");
    var row  = document.getElementById("hover-img-row");
    var gn   = document.getElementById("hover-img-gene");
    var meta = document.getElementById("hover-img-meta");
    var clo  = document.getElementById("hover-img-close");
    var volBox = document.getElementById("hover-img-volcano");
    var volImg = document.getElementById("hover-img-volcano-img");
    var volLab = document.getElementById("hover-img-volcano-label");
    var patBox = document.getElementById("hover-patents");
    var patents = window.__GENE_PATENTS__ || {};
    var depmapTpl = window.__DEPMAP_URL__ || "https://depmap.org/portal/gene/{gene}";
    var gd   = document.querySelector(".plotly-graph-div") || document.querySelector(".js-plotly-plot");
    if (!gd) return;
    var pinned = false;
    var currentGene = "";
    function positionPatBox() {
      // Anchor the patents panel immediately to the LEFT of the compound panel
      // (#hover-img), with an 8px gap. Recomputed after every render because
      // the compound panel resizes with the number of compounds + volcano.
      if (!patBox || !box) return;
      var gap = 8;
      // Read compound-panel geometry. Force display to measure offsetWidth/Left
      // accurately (a hidden box has 0 width).
      var prevDisp = box.style.display;
      if (prevDisp === "none" || !prevDisp) box.style.display = "block";
      var boxRect = box.getBoundingClientRect();
      box.style.display = prevDisp;
      var rightPx = Math.max(8, window.innerWidth - boxRect.left + gap);
      patBox.style.left = "auto";
      patBox.style.right = rightPx + "px";
    }
    function renderPatents(gene) {
      if (!patBox) return;
      var html = patents[gene];
      if (!html) {
        // No patents for this gene — still show a slim card with the DepMap link.
        var depmap = depmapTpl.replace("{gene}", encodeURIComponent(gene));
        html = '<div class="pat-header">'
             +   '<span class="pat-gene">' + gene + '</span>'
             +   ' <a class="pat-depmap" href="' + depmap + '" target="_blank" '
             +     'rel="noopener" title="open in DepMap">DepMap ↗</a>'
             + '</div>'
             + '<div class="pat-empty">no patent entries for this gene</div>';
      }
      patBox.innerHTML = html;
      patBox.style.display = "block";
      positionPatBox();
    }
    window.addEventListener("resize", positionPatBox);
    function render(p) {
      if (!p || !p.customdata) return false;
      var arr = p.customdata;
      if (!arr || !arr.length) return false;
      var metaTxt = "";
      var html = "";
      var cellIdx = 0;            // running compound-slot index for volcano lookup
      for (var i = 0; i < arr.length; i++) {
        var t = arr[i];
        if (!t) continue;
        // Gene-level meta row: ['__META__', '', '<key>=<val>']
        if (t[0] === "__META__") { metaTxt = t[2] || ""; continue; }
        if (!t[1]) continue;
        html += '<div class="cell" data-idx="' + cellIdx + '" data-cmp="' + (t[0] || '') + '">'
              + '<img src="data:image/png;base64,' + t[1] + '" draggable="false"/>'
              + '<div class="cap"><b>' + (t[0] || '') + '</b>'
              + (t[4] ? ' ' + t[4] : '')                      // compound meta icons (Daniela CSV, etc.)
              + (t[2] ? '<br>logfc ' + t[2] : '') + '</div>'
              + '</div>';
        cellIdx++;
      }
      if (!html) return false;
      var gene = (p.data && p.data.text && p.data.text[p.pointNumber]) || '';
      currentGene = gene;
      gn.textContent = gene;
      meta.textContent = metaTxt;
      row.innerHTML = html;
      // Stash the customdata array on the row so per-cell hover handlers can read it.
      row._arr = arr;
      // Reset volcano panel on each fresh render.
      volBox.style.display = "none";
      volImg.src = "";
      // Render the sibling patents panel.
      renderPatents(gene);
      return true;
    }
    function unpin() {
      pinned = false;
      box.classList.remove("pinned");
      box.style.display = "none";
      volBox.style.display = "none";
      if (patBox) { patBox.classList.remove("pinned"); patBox.style.display = "none"; }
    }
    // Event delegation: any compound cell, when the panel is pinned, shows
    // its associated volcano (customdata column index 3) on hover.
    row.addEventListener("mouseover", function(e) {
      if (!pinned) return;
      var cell = e.target.closest(".cell");
      if (!cell) return;
      var arr = row._arr;
      if (!arr) return;
      // Skip __META__ row when locating the cell's source entry.
      var skip = (arr[0] && arr[0][0] === "__META__") ? 1 : 0;
      var idx = parseInt(cell.getAttribute("data-idx"), 10) + skip;
      var t = arr[idx];
      if (!t || !t[3]) return;
      volImg.src = "data:image/png;base64," + t[3];
      volLab.textContent = currentGene + " · " + (cell.getAttribute("data-cmp") || "");
      volBox.style.display = "block";
    });
    row.addEventListener("mouseout", function(e) {
      if (!pinned) return;
      // Only hide when the cursor truly leaves the row (not when moving between cells).
      if (e.relatedTarget && row.contains(e.relatedTarget)) return;
      volBox.style.display = "none";
    });
    gd.on("plotly_hover", function(e) {
      if (pinned) return;
      if (render(e.points && e.points[0])) box.style.display = "block";
      else box.style.display = "none";
    });
    gd.on("plotly_unhover", function() {
      if (pinned) return;
      box.style.display = "none";
      if (patBox) patBox.style.display = "none";
    });
    gd.on("plotly_click", function(e) {
      if (render(e.points && e.points[0])) {
        pinned = true;
        box.classList.add("pinned");
        box.style.display = "block";
        if (patBox) patBox.classList.add("pinned");
      }
    });
    clo.addEventListener("click", unpin);
    document.addEventListener("keydown", function(e) {
      if (e.key === "Escape" && pinned) unpin();
    });
  });
</script>
'''


# JS/HTML injected by plot_3d_interface. Diverges from _HOVER_INJECT by adding
# a paginated compound panel (◀ / ▶ to walk all compounds for a target, K per
# page) and an axis legend driven by window.__AXIS_LABELS__ instead of the
# hard-coded R²/overall_score/MCS text. Kept separate so plot_target_3d is
# unaffected.
_INTERFACE_INJECT = '''
<style>
  html, body { height: 100%; margin: 0; padding: 0; background: white; }
  body { display: flex; align-items: center; justify-content: center; }
  .plotly-graph-div, .js-plotly-plot {
    width: 96vw !important; height: 94vh !important; margin: 0 auto !important;
  }
  #hover-img { position: fixed; top: 12px; right: 12px; z-index: 9999;
               background: white; border: 1px solid #bbb; padding: 6px;
               border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
               display: none; font: 11px sans-serif; color: #333;
               max-height: 92vh; overflow-y: auto; max-width: 96vw;
               user-select: text; }
  #hover-img.pinned { border-color: #1D3557; border-width: 2px; padding: 5px;
                      box-shadow: 0 4px 14px rgba(0,0,0,0.25); }
  #hover-img .row { display: flex; flex-direction: row; gap: 6px;
                    align-items: flex-start; flex-wrap: wrap; }
  #hover-img .cell { display: flex; flex-direction: column; align-items: center;
                     border: 1px solid #eee; border-radius: 4px; padding: 3px; }
  #hover-img .cell img { display: block; width: 170px; height: 110px;
                         object-fit: contain;
                         user-select: none; -webkit-user-drag: none; pointer-events: none; }
  #hover-img .cell .noimg { width: 170px; height: 110px; display: flex;
                            align-items: center; justify-content: center;
                            color: #bbb; font-style: italic; }
  #hover-img .cell .cap { padding-top: 2px; max-width: 170px; word-wrap: break-word;
                          text-align: center; line-height: 1.25;
                          user-select: text; cursor: text; }
  #hover-img .cell .cap b { user-select: all; }
  #hover-img .cell .sub { color: #777; font-size: 9px; }
  #hover-img .cell .pl  { color: #1D3557; font-size: 9px; }
  #hover-img .cell { cursor: pointer; }
  /* Click-pinned compound — its volcano(s) stay shown while you scroll. */
  #hover-img .cell.vpin { border-color: #1D3557;
                          box-shadow: 0 0 0 1px #1D3557 inset; background: #f3f6fb; }
  #hover-img .header { display: flex; align-items: center; gap: 8px;
                       padding-bottom: 4px; }
  #hover-img .gene { font-weight: 600; text-align: left; user-select: text; }
  #hover-img .meta { color: #555; font-size: 10px; font-family: ui-monospace, monospace;
                     user-select: text; flex: 1; }
  #hover-img .hint { color: #999; font-size: 10px; font-style: italic; }
  #hover-img.pinned .hint { display: none; }
  #hover-img .close { display: none; cursor: pointer; font-size: 16px;
                      color: #888; padding: 0 6px; border-radius: 3px;
                      user-select: none; line-height: 1; }
  #hover-img.pinned .close { display: inline-block; }
  #hover-img .close:hover { background: #eee; color: #333; }
  #hover-img .pager { display: none; align-items: center; justify-content: center;
                      gap: 10px; padding: 2px 0 6px 0; }
  #hover-img .pager .pg-btn { cursor: pointer; user-select: none; font-size: 15px;
                              color: #1D3557; padding: 0 8px; border-radius: 4px;
                              border: 1px solid #cdd6e0; line-height: 1.6; }
  #hover-img .pager .pg-btn:hover { background: #eef2f7; }
  #hover-img .pager .pg-btn.disabled { color: #ccc; border-color: #eee;
                                       cursor: default; background: none; }
  #hover-img .pager .pg-ind { font-size: 11px; color: #555; min-width: 150px;
                              text-align: center; }
  #hover-img .empty { color: #999; font-style: italic; padding: 6px 2px; }
  #hover-img .volcano { display: none; margin-top: 6px; text-align: center; }
  #hover-img .volcano .vlabel { font-size: 10px; color: #555; margin: 4px 0 2px 0; }
  #hover-img .volcano .vmiss { color: #bbb; font-style: italic; font-size: 10px; }
  #hover-img .volcano img { max-width: 100%; height: auto;
                            border: 1px solid #eee; border-radius: 4px; }
  /* interactive SVG volcano (hover a significant point -> gene-name tooltip) */
  #hover-img .volcano .vobj { width: 360px; height: 360px; max-width: 100%;
                              border: 1px solid #eee; border-radius: 4px; display: block;
                              margin: 0 auto; }
  /* Plate filter — tick boxes choosing which plates' compounds + volcanoes show. */
  #filter-panel { position: fixed; top: 12px; left: 12px; z-index: 9998;
                  background: white; border: 1px solid #bbb; border-radius: 6px;
                  padding: 6px 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
                  font: 11px sans-serif; color: #333; max-height: 80vh; width: 415px;
                  overflow-y: auto; display: none; user-select: none; }
  #filter-panel .fp-group + .fp-group { margin-top: 8px; }
  /* section labels grouping the filters (Compound filters / Target filters) */
  #filter-panel .fp-section { font-weight: 700; font-size: 10px; letter-spacing: .4px;
                              text-transform: uppercase; color: #457B9D; margin: 0 0 5px; }
  #filter-panel .fp-section.fp-sec2 { margin-top: 12px; padding-top: 8px;
                                      border-top: 1px solid #ddd; }
  /* collapsible whole-section header (e.g. PIN/HIDE) */
  #filter-panel .fp-section.fp-collap { cursor: pointer; }
  #filter-panel .fp-section.collapsed .fp-caret { transform: rotate(-90deg); }
  #sec-search.collapsed + #search-group { display: none; }
  #filter-panel .pf-head { font-weight: 700; padding-bottom: 4px; cursor: pointer;
                           border-bottom: 1px solid #eee; margin-bottom: 4px; }
  #filter-panel .pf-head span { color: #1D3557; cursor: pointer; font-weight: 400;
                                font-size: 10px; }
  #filter-panel .pf-head span:hover { text-decoration: underline; }
  /* collapsible caret + collapsed state */
  #filter-panel .fp-caret { display: inline-block; width: 10px; color: #1D3557;
                            transition: transform .12s; }
  #filter-panel .fp-group.collapsed .fp-caret { transform: rotate(-90deg); }
  #filter-panel .fp-group.collapsed .fp-boxes { display: none; }
  /* plate list as two columns to keep the panel short */
  #filter-panel .fp-boxes.pf-2col { display: grid;
                                    grid-template-columns: repeat(2, max-content);
                                    column-gap: 14px; }
  #filter-panel label { display: block; padding: 1px 0; cursor: pointer; white-space: nowrap; }
  #filter-panel input { margin-right: 5px; vertical-align: middle; }
  /* Display section: 2D / 3D pill toggle */
  #filter-panel .disp-row { display: flex; align-items: center; gap: 10px; padding: 2px 0 4px; }
  #filter-panel .disp-label { font-weight: 600; color: #1D3557; }
  #filter-panel .disp-toggle { display: inline-flex; cursor: pointer; user-select: none;
                               border: 1px solid #bbb; border-radius: 999px; overflow: hidden;
                               font-weight: 700; font-size: 10px; }
  #filter-panel .disp-toggle .seg { padding: 2px 11px; color: #888; background: #f0f0f0;
                                    transition: background .12s, color .12s; }
  #filter-panel .disp-toggle .seg.active { color: #fff; background: #1D3557; }
  /* Plates nested by date: collapsible sub-block per date inside the Plates group. */
  #filter-panel .pf-date + .pf-date { margin-top: 4px; }
  #filter-panel .pf-date-head { cursor: pointer; font-weight: 600; color: #1D3557;
                                padding: 2px 0; white-space: nowrap; }
  #filter-panel .pf-date-head .pf-date-n { color: #888; font-weight: 400; font-size: 10px; }
  #filter-panel .pf-date.collapsed .pf-date-boxes { display: none; }
  #filter-panel .pf-date.collapsed .fp-caret { transform: rotate(-90deg); }
  #filter-panel .pf-date-boxes { padding-left: 12px; }
  /* Gene search + pin overlay */
  #gene-search-wrap { position: relative; display: flex; gap: 5px; }
  #gene-search { flex: 1 1 auto; min-width: 0; padding: 3px 6px; font: 12px sans-serif;
                 border: 1px solid #bbb; border-radius: 4px; }
  #filter-panel .mode-btn { flex: 0 0 auto; padding: 3px 10px; font: 600 11px sans-serif; cursor: pointer;
                color: #fff; background: #1D3557; border: none; border-radius: 4px; white-space: nowrap; }
  #filter-panel .mode-btn:hover { background: #16324f; }
  #select-btn.active { background: #ff8c00; }            /* pin mode = orange */
  #select-btn.active:hover { background: #e67e00; }
  #hide-btn.active { background: #d62828; }              /* hide mode = red */
  #hide-btn.active:hover { background: #b81f1f; }
  #gene-ac { position: absolute; top: 100%; left: 0; right: 0; z-index: 10001;
             background: #fff; border: 1px solid #bbb; border-top: none; border-radius: 0 0 4px 4px;
             max-height: 220px; overflow-y: auto; display: none; box-shadow: 0 3px 8px rgba(0,0,0,0.15); }
  #gene-ac .ac-item { padding: 3px 8px; font: 12px sans-serif; cursor: pointer; }
  #gene-ac .ac-item.active, #gene-ac .ac-item:hover { background: #e8eef6; }
  #gene-ac .ac-empty { padding: 3px 8px; font: 12px sans-serif; color: #999; }
  #gene-ac .ac-tag { font-size: 9px; color: #fff; background: #b8860b; border-radius: 3px; padding: 0 4px; }
  /* click-to-select mode: target (crosshair) cursor over the plot + compound rows */
  body.select-mode .plotly-graph-div, body.select-mode .plotly-graph-div canvas { cursor: crosshair !important; }
  body.select-mode #ifx-row .cell { cursor: crosshair !important; }
  #pinned-box { margin-top: 6px; display: none; }
  #pinned-box .pin-hd { font-weight: 700; font-size: 11px; color: #1D3557; padding-bottom: 3px; }
  #pinned-box #pin-clear { color: #888; font-weight: 400; cursor: pointer; }
  #pinned-box #pin-clear:hover { text-decoration: underline; }
  #pinned-box .pin-chip { display: inline-flex; align-items: center; gap: 4px; margin: 2px 3px 0 0;
                          padding: 1px 4px 1px 7px; font: 11px sans-serif; background: #fff7d6;
                          border: 1px solid #FFC400; border-radius: 10px; }
  #pinned-box .pin-x { cursor: pointer; color: #b8860b; font-weight: 700; padding: 0 2px; }
  #pinned-box .pin-x:hover { color: #1D3557; }
  #pinned-box .pin-cmp { background: #eef2f8; border-color: #1D3557; }
  #pinned-box .pin-n { color: #888; font-size: 10px; }
  /* HIDE sub-block (mirror of pin, in red) */
  #hide-search-wrap { position: relative; display: flex; gap: 5px; margin-top: 8px; }
  #hide-search { flex: 1 1 auto; min-width: 0; padding: 3px 6px; font: 12px sans-serif;
                 border: 1px solid #bbb; border-radius: 4px; }
  #hide-ac { position: absolute; top: 100%; left: 0; right: 0; z-index: 10001;
             background: #fff; border: 1px solid #bbb; border-top: none; border-radius: 0 0 4px 4px;
             max-height: 220px; overflow-y: auto; display: none; box-shadow: 0 3px 8px rgba(0,0,0,0.15); }
  #hide-ac .ac-item { padding: 3px 8px; font: 12px sans-serif; cursor: pointer; }
  #hide-ac .ac-item.active, #hide-ac .ac-item:hover { background: #fdecec; }
  #hide-ac .ac-empty { padding: 3px 8px; font: 12px sans-serif; color: #999; }
  #hide-ac .ac-tag { font-size: 9px; color: #fff; background: #b8860b; border-radius: 3px; padding: 0 4px; }
  /* click-to-hide mode: a 'no' (not-allowed) cursor over the plot + compound rows */
  body.hide-mode .plotly-graph-div, body.hide-mode .plotly-graph-div canvas { cursor: not-allowed !important; }
  body.hide-mode #ifx-row .cell { cursor: not-allowed !important; }
  #hidden-box { margin-top: 6px; display: none; }
  #hidden-box .pin-hd { font-weight: 700; font-size: 11px; color: #d62828; padding-bottom: 3px; }
  #hidden-box #hide-clear { color: #888; font-weight: 400; cursor: pointer; }
  #hidden-box #hide-clear:hover { text-decoration: underline; }
  #hidden-box .hide-chip { display: inline-flex; align-items: center; gap: 4px; margin: 2px 3px 0 0;
                           padding: 1px 4px 1px 7px; font: 11px sans-serif; background: #fdecec;
                           border: 1px solid #d62828; border-radius: 10px; }
  #hidden-box .pin-x { cursor: pointer; color: #d62828; font-weight: 700; padding: 0 2px; }
  #hidden-box .pin-x:hover { color: #1D3557; }
  #hidden-box .pin-n { color: #888; font-size: 10px; }
  /* Download selection */
  #sec-download { display: flex; align-items: center; justify-content: space-between; }
  #dl-btn { padding: 2px 10px; font: 600 11px sans-serif; cursor: pointer; text-transform: none;
            letter-spacing: 0; color: #fff; background: #1D3557; border: none; border-radius: 4px; }
  #dl-btn:hover { background: #16324f; }
  #dl-note { margin-top: 5px; font: 11px sans-serif; color: #2A9D8F; word-break: break-all; }
  /* Session save/load: two buttons in the section header (mirror #dl-btn) */
  #sec-session { display: flex; align-items: center; justify-content: space-between; }
  #sec-session .sess-btns { display: inline-flex; gap: 6px; }
  #sess-save-btn, #sess-load-btn { padding: 2px 10px; font: 600 11px sans-serif; cursor: pointer;
            text-transform: none; letter-spacing: 0; color: #fff; background: #1D3557; border: none; border-radius: 4px; }
  #sess-save-btn:hover, #sess-load-btn:hover { background: #16324f; }
  #sess-note { margin-top: 5px; font: 11px sans-serif; color: #2A9D8F; word-break: break-all; }
  #hover-patents {
    position: fixed; top: 12px; right: 660px; z-index: 9999;
    background: white; border: 1px solid #bbb; border-radius: 6px;
    padding: 6px 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font: 11px sans-serif; color: #333; max-width: 320px;
    max-height: 92vh; overflow-y: auto; user-select: text; display: none;
  }
  #hover-patents.pinned { border-color: #1D3557; border-width: 2px; padding: 5px 7px;
                          box-shadow: 0 4px 14px rgba(0,0,0,0.25); }
  #hover-patents .pat-header { display: flex; align-items: baseline; gap: 6px;
                                font-weight: 700; padding-bottom: 4px;
                                border-bottom: 1px solid #eee; margin-bottom: 4px; }
  #hover-patents .pat-gene   { font-size: 12px; }
  #hover-patents .pat-depmap { font-size: 10px; color: #1D3557; text-decoration: none; }
  #hover-patents .pat-depmap:hover { text-decoration: underline; }
  #hover-patents .pat-table  { border-collapse: collapse; width: 100%; font-size: 11px; }
  #hover-patents .pat-table td { padding: 2px 4px; vertical-align: top; }
  #hover-patents .pat-table tr:nth-child(even) td { background: #f8f8f8; }
  #hover-patents .pat-empty  { color: #999; font-style: italic; padding: 4px 0; }
  #axis-legend {            /* bottom-left, immediately above the slider box */
    position: fixed; bottom: 104px; left: 12px; z-index: 9998;
    background: white; border: 1px solid #bbb; padding: 6px 8px;
    border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    font: 11px sans-serif; color: #333; max-width: 360px; user-select: text;
  }
  #axis-legend .title { font-weight: 700; padding-bottom: 3px; }
  #axis-legend .ax { display: block; padding: 1px 0; cursor: help; }
  #axis-legend .ax:hover { background: #f3f3f3; border-radius: 3px; }
  #axis-legend .ax b  { display: inline-block; min-width: 1.2em; color: #555; }
  #axis-legend .ax .lab { font-weight: 600; }
  /* Per-gene degradation-research box (bottom-right). */
  #research-box { position: fixed; right: 12px; bottom: 12px; z-index: 9998;
                  background: white; border: 1px solid #bbb; border-radius: 6px;
                  padding: 8px 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                  font: 11px sans-serif; color: #333; width: 360px; max-height: 52vh;
                  overflow-y: auto; display: none; user-select: text; }
  #research-box.pinned { border-color: #1D3557; border-width: 2px; }
  #research-box .rb-head { display: flex; align-items: baseline; gap: 8px;
                           padding-bottom: 4px; border-bottom: 1px solid #eee;
                           margin-bottom: 6px; flex-wrap: wrap; }
  #research-box .rb-gene { font-weight: 700; font-size: 13px; }
  #research-box .rb-class { color: #666; font-size: 10px; }
  #research-box .rb-conf { font-size: 9px; font-weight: 700; padding: 1px 6px;
                           border-radius: 8px; color: #fff; }
  #research-box .rb-sec { margin-bottom: 5px; line-height: 1.3; }
  #research-box .rb-lab { font-weight: 600; color: #1D3557; }
  #research-box .rb-src a { color: #1D3557; text-decoration: none; margin-right: 6px;
                            font-size: 10px; }
  #research-box .rb-src a:hover { text-decoration: underline; }
  /* Range sliders — flattened: 3 axes side-by-side, bottom-left. */
  #range-panel { position: fixed; bottom: 12px; left: 12px; z-index: 9998;
                 background: white; border: 1px solid #bbb; border-radius: 6px;
                 padding: 6px 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
                 font: 11px sans-serif; color: #333;
                 display: none; user-select: none; }
  #range-panel .rp-title { font-weight: 700; padding-bottom: 4px;
                           border-bottom: 1px solid #eee; margin-bottom: 6px; }
  #range-panel .rp-count { color: #1D3557; font-weight: 400; font-size: 10px; }
  #range-panel .rp-cols { display: flex; gap: 16px; align-items: flex-start; }
  #range-panel .rp-row { margin: 0; width: 165px; }
  /* fixed-height header (name + value each on its own line) so the three dual
     sliders line up regardless of label length */
  #range-panel .rp-name { display: block; font-weight: 600; line-height: 13px;
                          min-height: 26px; }
  #range-panel .rp-val  { display: block; float: none; color: #555;
                          font-family: ui-monospace, monospace; font-size: 10px;
                          margin: 1px 0 3px 0; }
  /* one dual-handle slider per axis: two range inputs overlaid on one track */
  #range-panel .rp-dual { position: relative; height: 20px; margin-top: 7px; }
  #range-panel .rp-dual .rp-track { position: absolute; top: 8px; left: 0; right: 0;
                                    height: 4px; background: #d8dee6; border-radius: 2px; }
  #range-panel .rp-dual input[type=range] { position: absolute; top: 0; left: 0;
      width: 100%; height: 20px; margin: 0; background: none; pointer-events: none;
      -webkit-appearance: none; appearance: none; }
  #range-panel .rp-dual input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; appearance: none; pointer-events: all;
      height: 16px; width: 16px; border-radius: 50%; background: #1D3557;
      border: 2px solid #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.3); cursor: pointer; }
  #range-panel .rp-dual input[type=range]::-moz-range-thumb { pointer-events: all;
      height: 16px; width: 16px; border-radius: 50%; background: #1D3557;
      border: 2px solid #fff; cursor: pointer; }
  #range-panel .rp-dual input[type=range]::-webkit-slider-runnable-track { background: none; }
  #range-panel .rp-dual input[type=range]::-moz-range-track { background: none; }
  #range-panel .rp-reset { color: #1D3557; cursor: pointer; font-size: 10px;
                           font-weight: 400; }
  #range-panel .rp-reset:hover { text-decoration: underline; }
</style>
<div id="filter-panel">
  <div class="fp-section fp-collap" id="sec-search"><span class="fp-caret">&#9662;</span>PIN/HIDE</div>
  <div class="fp-group" id="search-group">
    <div id="gene-search-wrap">
      <input type="text" id="gene-search" placeholder="search to pin…" autocomplete="off" spellcheck="false">
      <button id="select-btn" class="mode-btn" title="Pin mode — when on (orange), click a target dot or a compound row to pin it">Pin</button>
      <div id="gene-ac"></div>
    </div>
    <div id="pinned-box"></div>
    <div id="hide-search-wrap">
      <input type="text" id="hide-search" placeholder="search to hide…" autocomplete="off" spellcheck="false">
      <button id="hide-btn" class="mode-btn" title="Hide mode — when on (red), click a target dot or a compound row to hide it">Hide</button>
      <div id="hide-ac"></div>
    </div>
    <div id="hidden-box"></div>
  </div>
  <div class="fp-section fp-sec2" id="sec-display">Display</div>
  <div class="fp-group" id="display-group">
    <div class="disp-row">
      <span class="disp-label">Axes</span>
      <span class="disp-toggle" id="disp-toggle" role="switch" title="3D = SAR predictability × association × MS score; 2D = a flat association × MS view (SAR axis hidden; SAR range slider still filters).">
        <span class="seg seg2d" data-mode="2D">2D</span><span class="seg seg3d active" data-mode="3D">3D</span>
      </span>
    </div>
  </div>
  <div class="fp-section fp-sec2" id="sec-compound">Compound filters</div>
  <div class="fp-group collapsed" id="plate-group">
    <div class="pf-head" title="Show only experiments run on the ticked assay plates; a gene greys out when none of its compounds were measured on a ticked plate."><span class="fp-caret">&#9662;</span>Plates <span id="pf-all">all</span> / <span id="pf-none">none</span></div>
    <div id="pf-boxes" class="fp-boxes pf-2col"></div>
  </div>
  <div class="fp-group collapsed" id="activity-group">
    <div class="pf-head" title="Keep only compound experiments at the ticked activity levels — the bucketed number of genes that compound significantly down-modulated."><span class="fp-caret">&#9662;</span>Activity <span id="af-all">all</span> / <span id="af-none">none</span></div>
    <div id="af-boxes" class="fp-boxes"></div>
  </div>
  <div class="fp-group collapsed" id="compound-group" style="display:none">
    <div class="pf-head" title="Toggle whole compound classes on or off — control compounds and known contaminants."><span class="fp-caret">&#9662;</span>Other</div>
    <div id="cf-boxes" class="fp-boxes"><label><input type="checkbox" id="control-toggle" checked> Controls</label><label><input type="checkbox" id="contaminant-toggle" checked> Contaminants</label></div>
  </div>
  <div class="fp-group collapsed" id="validation-compound-group" style="display:none">
    <div class="pf-head" title="Show or hide compounds by FBXO31 validation status — validated (dependent) vs devalidated (independent) compounds."><span class="fp-caret">&#9662;</span>Validation <span id="vcf-all">all</span> / <span id="vcf-none">none</span></div>
    <div id="vcf-boxes" class="fp-boxes"></div>
  </div>
  <div class="fp-section fp-sec2" id="sec-target">Target filters</div>
  <div class="fp-group collapsed" id="depmap-group" style="display:none">
    <div class="pf-head" title="Filter genes by their DepMap cell-fitness class (Pan-essential, Selective, Non-essential)."><span class="fp-caret">&#9662;</span>DepMap dependency <span id="df-all">all</span> / <span id="df-none">none</span></div>
    <div id="df-boxes" class="fp-boxes"></div>
  </div>
  <div class="fp-group collapsed" id="confidence-group" style="display:none">
    <div class="pf-head" title="Filter genes by the confidence of the degradation-target research assessment (High, Med, Low)."><span class="fp-caret">&#9662;</span>Confidence <span id="cn-all">all</span> / <span id="cn-none">none</span></div>
    <div id="cn-boxes" class="fp-boxes"></div>
  </div>
  <div class="fp-group collapsed" id="lof-group" style="display:none">
    <div class="pf-head" title="Filter genes by whether loss of function is expected to be therapeutically beneficial (Yes, No, Maybe)."><span class="fp-caret">&#9662;</span>LoF benefit <span id="lf-all">all</span> / <span id="lf-none">none</span></div>
    <div id="lf-boxes" class="fp-boxes"></div>
  </div>
  <div class="fp-group collapsed" id="validation-group" style="display:none">
    <div class="pf-head" title="Filter genes by FBXO31 validation status — experimentally dependent vs independent targets."><span class="fp-caret">&#9662;</span>Validation <span id="vf-all">all</span> / <span id="vf-none">none</span></div>
    <div id="vf-boxes" class="fp-boxes"></div>
  </div>
  <div class="fp-section fp-sec2" id="sec-download">Download selection<button id="dl-btn" title="Download the current selection (proteins + their compounds) as CSV">Save</button></div>
  <div class="fp-group" id="download-group">
    <div id="dl-note"></div>
  </div>
  <div class="fp-section fp-sec2" id="sec-session">Session<span class="sess-btns"><button id="sess-save-btn" title="Save the whole interface state (pins, hides, filters, ranges, view) to a .iface file">Save</button><button id="sess-load-btn" title="Load a previously saved .iface session and restore its state">Load</button></span></div>
  <div class="fp-group" id="session-group">
    <input type="file" id="sess-file" accept=".iface,.json,application/json" style="display:none">
    <div id="sess-note"></div>
  </div>
</div>
<div id="hover-img">
  <div class="header">
    <span class="gene" id="ifx-gene"></span>
    <span class="meta" id="ifx-meta"></span>
    <span class="hint">hover → click dot to pin → ◀ ▶ to page → hover a compound to peek, click it to pin its volcano(s) · tick plates at left</span>
    <span class="close" id="ifx-close" title="Close (Esc)">×</span>
  </div>
  <div class="pager" id="ifx-pager">
    <span class="pg-btn" id="ifx-prev" title="previous (←)">◀</span>
    <span class="pg-ind" id="ifx-ind"></span>
    <span class="pg-btn" id="ifx-next" title="next (→)">▶</span>
  </div>
  <div class="row" id="ifx-row"></div>
  <div class="volcano" id="ifx-volcano"></div>
</div>
<div id="hover-patents"></div>
<div id="axis-legend"></div>
<div id="research-box"></div>
<div id="range-panel">
  <div class="rp-title">Ranges <span class="rp-reset" id="rp-reset">reset</span>
    <span class="rp-count" id="rp-count"></span></div>
  <div class="rp-cols">
    <div class="rp-row" data-axis="x">
      <span class="rp-name" id="x-name"></span><span class="rp-val" id="x-val"></span>
      <div class="rp-dual"><div class="rp-track"></div>
        <input type="range" id="x-lo"><input type="range" id="x-hi"></div>
    </div>
    <div class="rp-row" data-axis="y">
      <span class="rp-name" id="y-name"></span><span class="rp-val" id="y-val"></span>
      <div class="rp-dual"><div class="rp-track"></div>
        <input type="range" id="y-lo"><input type="range" id="y-hi"></div>
    </div>
    <div class="rp-row" data-axis="z">
      <span class="rp-name" id="z-name"></span><span class="rp-val" id="z-val"></span>
      <div class="rp-dual"><div class="rp-track"></div>
        <input type="range" id="z-lo"><input type="range" id="z-hi"></div>
    </div>
  </div>
</div>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    var box   = document.getElementById("hover-img");
    var row   = document.getElementById("ifx-row");
    var gn    = document.getElementById("ifx-gene");
    var meta  = document.getElementById("ifx-meta");
    var clo   = document.getElementById("ifx-close");
    var pager = document.getElementById("ifx-pager");
    var prevB = document.getElementById("ifx-prev");
    var nextB = document.getElementById("ifx-next");
    var indEl = document.getElementById("ifx-ind");
    var volBox = document.getElementById("ifx-volcano");
    var patBox = document.getElementById("hover-patents");
    var researchBox = document.getElementById("research-box");
    var research = window.__GENE_RESEARCH__ || {};
    var legEl  = document.getElementById("axis-legend");
    var pf     = document.getElementById("filter-panel");
    var pfBoxes = document.getElementById("pf-boxes");
    var afBoxes = document.getElementById("af-boxes");
    var dfBoxes = document.getElementById("df-boxes");
    var cnBoxes = document.getElementById("cn-boxes");
    var lfBoxes = document.getElementById("lf-boxes");
    var vfBoxes = document.getElementById("vf-boxes");
    var vcfBoxes = document.getElementById("vcf-boxes");
    var patents = window.__GENE_PATENTS__ || {};
    var depmapTpl = window.__DEPMAP_URL__ || "https://depmap.org/portal/gene/{gene}";
    var pageSize = window.__PAGE_SIZE__ || 5;
    var axis = window.__AXIS_LABELS__ || {x: "X", y: "Y", z: "Z"};
    var axisHelp = window.__AXIS_HELP__ || {};
    var plates = window.__PLATES__ || [];
    var ticked = {};
    var plateDefaults = window.__PLATE_DEFAULTS__ || null;   // plates to start ticked; null/absent = all
    plates.forEach(function(p) { ticked[p] = plateDefaults ? (plateDefaults.indexOf(p) !== -1) : true; });
    var activities = window.__ACTIVITIES__ || [];
    // Optional focus set: if __ACTIVITY_DEFAULTS__ is given, only those levels
    // start ticked (others off) so the view opens focused on e.g. Low + Single.
    var actDefaults = window.__ACTIVITY_DEFAULTS__ || null;
    var tickedAct = {};
    activities.forEach(function(a) {
      tickedAct[a] = actDefaults ? (actDefaults.indexOf(a) !== -1) : true;
    });
    // Control-compound filter: a single tickbox in the Activity panel. When OFF,
    // every entry whose compound id is a control is hidden everywhere — panel list,
    // volcanoes, and gene greying — exactly like an activity level being unticked.
    var controlCompounds = {};
    (window.__CONTROL_COMPOUNDS__ || []).forEach(function(c) { controlCompounds[c] = true; });
    var controlOn = (window.__CONTROL_DEFAULT_ON__ !== false);  // default state of the tickbox
    var contaminantCompounds = {};
    (window.__CONTAMINANT_COMPOUNDS__ || []).forEach(function(c) { contaminantCompounds[c] = true; });
    var contaminantOn = (window.__CONTAMINANT_DEFAULT_ON__ !== false);
    // A compound is shown unless its class is toggled off (controls and/or contaminants).
    function cmpAllowed(t) {
      if (isHiddenCompound(t[0])) return false;   // hidden compounds drop from count/export/panel
      if (!cmpValAllowed(t[0])) return false;     // compound-validation (FBXO31 dependent/independent)
      if (!controlOn && controlCompounds[t[0]]) return false;
      if (!contaminantOn && contaminantCompounds[t[0]]) return false;
      return true;
    }
    // DepMap dependency filter (gene-level): each gene's depmap_dependency category
    // (Pan-essential / Selective / Non-essential / Other / "(no data)"). Unticking a
    // category greys out the genes in it. Reuses the generic checkbox group below.
    var geneDepmap = window.__GENE_DEPMAP__ || {};
    var depmapCats = window.__DEPMAP_CATS__ || [];
    // Optional focus default: if __*_DEFAULTS__ is a list, only those categories start
    // ticked (an empty list ticks none); null/absent => all ticked. [] is truthy in JS,
    // so "|| null" preserves an emitted empty list while mapping a missing global to null.
    var depDefaults = window.__DEPMAP_DEFAULTS__ || null;
    var tickedDep = {};
    depmapCats.forEach(function(c) { tickedDep[c] = depDefaults ? (depDefaults.indexOf(c) !== -1) : true; });
    function depAllowed(gene) {
      if (!depmapCats.length) return true;
      return tickedDep[geneDepmap[gene] || "(no data)"] !== false;
    }
    // Research confidence filter (gene-level): High / Med / Low / "(no data)".
    var geneConf = window.__GENE_CONF__ || {};
    var confCats = window.__CONF_CATS__ || [];
    var confDefaults = window.__CONF_DEFAULTS__ || null;
    var tickedConf = {};
    confCats.forEach(function(c) { tickedConf[c] = confDefaults ? (confDefaults.indexOf(c) !== -1) : true; });
    function confAllowed(gene) {
      if (!confCats.length) return true;
      return tickedConf[geneConf[gene] || "(no data)"] !== false;
    }
    // LoF therapeutic benefit filter (gene-level): Yes / No / Maybe / "(no data)".
    var geneLof = window.__GENE_LOF__ || {};
    var lofCats = window.__LOF_CATS__ || [];
    var lofDefaults = window.__LOF_DEFAULTS__ || null;
    var tickedLof = {};
    lofCats.forEach(function(c) { tickedLof[c] = lofDefaults ? (lofDefaults.indexOf(c) !== -1) : true; });
    function lofAllowed(gene) {
      if (!lofCats.length) return true;
      return tickedLof[geneLof[gene] || "(no data)"] !== false;
    }
    // Target-validation filter (gene-level): "Yes" tickbox controls genes in
    // validated_targets, "No" controls devalidated_targets. A gene in neither
    // set is unaffected; a gene in both shows while either box is ticked.
    var validatedSet = {};
    (window.__VALIDATED_TARGETS__ || []).forEach(function(g) { validatedSet[g] = true; });
    var devalidatedSet = {};
    (window.__DEVALIDATED_TARGETS__ || []).forEach(function(g) { devalidatedSet[g] = true; });
    var valLabelYes = window.__VAL_LABEL_YES__ || "Yes";   // label for the validated box
    var valLabelNo  = window.__VAL_LABEL_NO__  || "No";    // label for the devalidated box
    var valCats = window.__VALIDATION_CATS__ || [];        // subset of the two labels present
    var valDefaults = window.__VALIDATION_DEFAULTS__ || null;
    var tickedVal = {};
    valCats.forEach(function(c) { tickedVal[c] = valDefaults ? (valDefaults.indexOf(c) !== -1) : true; });
    function valAllowed(gene) {
      if (!valCats.length) return true;
      var isV = validatedSet[gene], isD = devalidatedSet[gene];
      if (!isV && !isD) return true;                       // not annotated -> unaffected
      if (isV && tickedVal[valLabelYes] !== false) return true;
      if (isD && tickedVal[valLabelNo] !== false) return true;
      return false;
    }
    // Compound-validation filter (compound-level, mirror of the gene one): the
    // "dependent" tickbox controls compounds in validated_compounds, "independent"
    // controls devalidated_compounds. A compound in neither set is unaffected.
    // Gates in cmpAllowed(), so an unticked box hides those compounds from the
    // panel/count/export and greys genes whose only compounds were hidden.
    var cmpValidatedSet = {};
    (window.__VALIDATED_COMPOUNDS__ || []).forEach(function(c) { cmpValidatedSet[c] = true; });
    var cmpDevalidatedSet = {};
    (window.__DEVALIDATED_COMPOUNDS__ || []).forEach(function(c) { cmpDevalidatedSet[c] = true; });
    var cmpValLabelYes = window.__CMP_VAL_LABEL_YES__ || "Yes";
    var cmpValLabelNo  = window.__CMP_VAL_LABEL_NO__  || "No";
    var cmpValCats = window.__CMP_VALIDATION_CATS__ || [];
    var cmpValDefaults = window.__CMP_VALIDATION_DEFAULTS__ || null;
    var tickedCmpVal = {};
    cmpValCats.forEach(function(c) { tickedCmpVal[c] = cmpValDefaults ? (cmpValDefaults.indexOf(c) !== -1) : true; });
    function cmpValAllowed(c) {
      if (!cmpValCats.length) return true;
      var isV = cmpValidatedSet[c], isD = cmpDevalidatedSet[c];
      if (!isV && !isD) return true;
      if (isV && tickedCmpVal[cmpValLabelYes] !== false) return true;
      if (isD && tickedCmpVal[cmpValLabelNo] !== false) return true;
      return false;
    }
    var gd = document.querySelector(".plotly-graph-div") || document.querySelector(".js-plotly-plot");
    if (!gd) return;

    // Axis legend with per-axis explanations shown as a hover tooltip (like
    // plot_target_3d). Build via DOM + the title property so help text needs no
    // HTML escaping.
    legEl.innerHTML = '<div class="title">ⓘ Axis legend (hover for details)</div>';
    ['x', 'y', 'z'].forEach(function(k, i) {
      var sp = document.createElement('span');
      sp.className = 'ax';
      if (axisHelp[k]) sp.title = axisHelp[k];
      sp.innerHTML = '<b>' + ['X', 'Y', 'Z'][i] + '</b> <span class="lab"></span>';
      sp.querySelector('.lab').textContent = axis[k] || '';
      legEl.appendChild(sp);
    });

    // declared before the display IIFE (which assigns it) so a later var-initializer can't clobber it
    var setMode2DHook = function(two) {};         // set in the display block; switches 2D/3D view
    // --- Display: 2D / 3D toggle. 2D = orthographic camera down the SAR (x) axis so only
    // association (y) × MS (z) show; x-axis hidden, rotation locked. Same Scatter3d traces,
    // so every filter / slider / pin / hover interaction is untouched. SAR slider still filters.
    (function () {
      var tg = document.getElementById("disp-toggle");
      if (!tg || !gd || typeof Plotly === "undefined") return;
      var CAM3D = {eye: {x: 1.25, y: 1.25, z: 1.25}, up: {x: 0, y: 0, z: 1},
                   center: {x: 0, y: 0, z: 0}, projection: {type: "perspective"}};
      // pre-panned toward screen-centre: eye & center share the y/z offset so the look
      // stays axis-aligned (flat); without the shift the ortho view sits bottom-right.
      // User can drag further from here (dragmode 'pan').
      var CAM2D = {eye: {x: 2.5, y: 0.30, z: -0.35}, up: {x: 0, y: 0, z: 1},
                   center: {x: 0, y: 0.30, z: -0.35}, projection: {type: "orthographic"}};
      function setMode(two) {
        tg.querySelector(".seg2d").classList.toggle("active", two);
        tg.querySelector(".seg3d").classList.toggle("active", !two);
        Plotly.relayout(gd, {
          "scene.camera": two ? CAM2D : CAM3D,
          "scene.xaxis.visible": !two,
          // 2D: 'pan' lets you drag the plot to reposition it (and scroll to zoom)
          // without rotating out of the flat view; 3D restores rotate (turntable).
          "scene.dragmode": two ? "pan" : "turntable"
        });
      }
      tg.addEventListener("click", function (e) {
        var seg = e.target.closest(".seg");
        setMode(seg ? seg.getAttribute("data-mode") === "2D"
                    : !tg.querySelector(".seg2d").classList.contains("active"));
      });
      setMode2DHook = setMode;   // let session-load switch the view
    })();

    var pinned = false;
    var currentGene = "";
    var fullArr = [];
    var entries = [];      // [{t: row, idx: absolute index in fullArr}] minus __META__
    var page = 0;
    var volPinIdx = null;  // data-eidx of the compound whose volcano(s) are click-pinned
    var recolor3d = function() {};  // set by the slider block; re-applies gene colouring
    // --- pin / hide state (search boxes + click-to-select) ---
    var pinnedGenes = [];        // gene names pinned directly
    var pinnedCompounds = [];    // compound ids pinned (each pins its target genes)
    var hiddenGenes = [];        // gene names hidden directly (drop from the plot)
    var hiddenCompounds = [];    // compound ids hidden (drop the compound only; its genes stay)
    var _hiddenSet = {};         // cache: directly-hidden genes
    var _hiddenCmpSet = {};      // cache: directly-hidden compound ids
    function rebuildHidden() {
      _hiddenSet = {};
      hiddenGenes.forEach(function(g) { _hiddenSet[g] = 1; });
      _hiddenCmpSet = {};
      hiddenCompounds.forEach(function(c) { _hiddenCmpSet[c] = 1; });
    }
    function isHiddenGene(g) { return _hiddenSet[g] === 1; }
    function isHiddenCompound(c) { return _hiddenCmpSet[c] === 1; }
    // effective pinned-gene set = (directly-pinned ∪ pinned-compounds' targets) minus hidden
    function effectivePinSet() {
      var s = {}, cg = window.__COMPOUND_GENES__ || {}, xyz = window.__GENE_XYZ__ || {};
      pinnedCompounds.forEach(function(c) {
        (cg[c] || []).forEach(function(g) { if (xyz[g] && !isHiddenGene(g)) s[g] = 1; });
      });
      pinnedGenes.forEach(function(g) { if (!isHiddenGene(g)) s[g] = 1; });
      return s;
    }
    var clickMode = "";                           // "", "pin", or "hide" (set in the pin/hide block)
    var togglePinGene = function(g) {};           // assigned in the pin/hide block
    var togglePinCompound = function(c) {};
    var toggleHideGene = function(g) {};
    var toggleHideCompound = function(c) {};
    var refreshLabelsHook = function() {};        // set in the range block; rebuilds in-range labels
    var updateCountHook = function() {};          // set in the range block; refreshes the protein/compound tally
    var exportCSVHook = function() { return null; };  // set in the range block; builds the selection CSV
    var applyPinHideHook = function() {};         // set in the pin/hide block; re-renders pins/hides from the arrays

    // A gene is "active" under the current Plate + Activity ticks if it has at
    // least one compound whose plate AND activity are both ticked. Used to grey
    // out genes that have no compound on the selected plates/activities.
    function geneHasVisibleCompound(gene) {
      var arr = (window.__GENE_COMPOUNDS__ || {})[gene];
      if (!arr) return false;
      for (var i = 0; i < arr.length; i++) {
        var t = arr[i];
        if (!t || t[0] === "__META__") continue;
        if (!cmpAllowed(t)) continue;
        if (Array.isArray(t[3])) {
          for (var j = 0; j < t[3].length; j++) {
            var pl = t[3][j];
            var plateOk = (!plates.length) || ticked[pl[0]];
            var actOk = (!activities.length) || pl[3] === undefined || tickedAct[pl[3]];
            if (plateOk && actOk) return true;
          }
        } else {
          return true;   // single-volcano (non-plate) entry — always counts
        }
      }
      return false;
    }

    // Collect this gene's DISTINCT visible compound ids (passing class+plate+activity)
    // into `out` (used as a set), so the range panel can total compounds across the
    // in-range proteins without double-counting a compound that hits several genes.
    function collectVisibleCompounds(gene, out) {
      var arr = (window.__GENE_COMPOUNDS__ || {})[gene];
      if (!arr) return;
      for (var i = 0; i < arr.length; i++) {
        var t = arr[i];
        if (!t || t[0] === "__META__") continue;
        if (entryVisible(t)) out[t[0]] = 1;
      }
    }

    // Gather (Batch Molecule-Batch ID -> set of genes) for a gene's visible compounds,
    // honouring class/plate/activity filters (mbid lives at plate-row index 5, falling
    // back to the compound id). Feeds the Download-selection CSV.
    function collectExport(gene, map) {
      var arr = (window.__GENE_COMPOUNDS__ || {})[gene];
      if (!arr) return;
      for (var i = 0; i < arr.length; i++) {
        var t = arr[i];
        if (!t || t[0] === "__META__" || !cmpAllowed(t)) continue;
        if (isPaged(t)) {
          var vis = visPlates(t);   // plate-rows: [plate, logfc, volcano, activity, n_genes, mbid]
          for (var j = 0; j < vis.length; j++) {
            var pl = vis[j], bid = pl[5] || t[0], plate = pl[0] || "", act = pl[3] || "";
            var key = JSON.stringify([bid, plate, act]);
            (map[key] || (map[key] = {bid: bid, plate: plate, act: act, genes: {}})).genes[gene] = 1;
          }
        } else {
          var k2 = JSON.stringify([t[0], "", ""]);
          (map[k2] || (map[k2] = {bid: t[0], plate: "", act: "", genes: {}})).genes[gene] = 1;
        }
      }
    }

    // --- plate-aware helpers ---
    // A compound entry's volcano slot (t[3]) is either:
    //   * an Array of [plate, logfc, volcano_b64, activity]  (plate-aware FBX mode), or
    //   * a base64 string                                    (single-volcano legacy mode).
    // A plate-row is visible only if BOTH its plate and its activity are ticked.
    function isPaged(t) { return Array.isArray(t[3]); }
    function visPlates(t) {
      return t[3].filter(function(pl) {
        return ticked[pl[0]] && (!activities.length || pl[3] === undefined || tickedAct[pl[3]]);
      });
    }
    function entryVisible(t) {
      if (!cmpAllowed(t)) return false;
      if (isPaged(t)) return visPlates(t).length > 0;
      return true;
    }
    function bestLogfc(pls) {
      var b = null;
      pls.forEach(function(pl) {
        var v = parseFloat(pl[1]);
        if (!isNaN(v) && (b === null || v < b)) b = v;
      });
      return b === null ? '' : b.toFixed(2);
    }

    function rbEsc(s) {
      return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    function renderResearch(gene) {
      if (!researchBox) return;
      var r = research[gene];
      if (!r) { researchBox.style.display = "none"; return; }
      var conf = (r.confidence || '').toLowerCase();
      var cc = conf.indexOf('high') >= 0 ? '#2A9D8F'
             : conf.indexOf('low') >= 0 ? '#E63946' : '#E9A23B';
      function sec(lab, val) {
        if (val == null || val === '' || val.length === 0) return '';
        var txt = Array.isArray(val) ? val.map(rbEsc).join(', ') : rbEsc(val);
        return '<div class="rb-sec"><span class="rb-lab">' + lab + ':</span> ' + txt + '</div>';
      }
      var html = '<div class="rb-head"><span class="rb-gene">'
               + rbEsc(r.gene_name || gene) + '</span>'
               + (r.confidence ? '<span class="rb-conf" style="background:' + cc + ';">'
                                 + rbEsc(r.confidence) + ' conf</span>' : '')
               + (r.target_class ? '<span class="rb-class">' + rbEsc(r.target_class) + '</span>' : '')
               + '</div>';
      html += sec('LoF benefit', r.lof_therapeutic_benefit);
      html += sec('Degrader vs inhibitor', r.degrader_vs_inhibitor_rationale);
      html += sec('Degrader feasibility', r.degrader_feasibility);
      html += sec('DepMap dependency', r.depmap_dependency);
      html += sec('Top indications', r.opentargets_top_indications);
      html += sec('Existing degraders', r.existing_degraders);
      html += sec('Safety flags', r.safety_flags);
      html += sec('Biology', r.biology_rationale);
      if (Array.isArray(r.sources) && r.sources.length) {
        var links = r.sources.map(function(u, i) {
          return '<a href="' + rbEsc(u) + '" target="_blank" rel="noopener">[' + (i + 1) + ']</a>';
        }).join('');
        html += '<div class="rb-sec rb-src"><span class="rb-lab">Sources:</span> ' + links + '</div>';
      }
      researchBox.innerHTML = html;
      researchBox.style.display = "block";
    }

    function positionPatBox() {
      if (!patBox || !box) return;
      var gap = 8;
      var prevDisp = box.style.display;
      if (prevDisp === "none" || !prevDisp) box.style.display = "block";
      var boxRect = box.getBoundingClientRect();
      box.style.display = prevDisp;
      var rightPx = Math.max(8, window.innerWidth - boxRect.left + gap);
      patBox.style.left = "auto";
      patBox.style.right = rightPx + "px";
    }
    function renderPatents(gene) {
      if (!patBox) return;
      var html = patents[gene];
      if (!html) {
        var depmap = depmapTpl.replace("{gene}", encodeURIComponent(gene));
        html = '<div class="pat-header"><span class="pat-gene">' + gene + '</span>'
             + ' <a class="pat-depmap" href="' + depmap + '" target="_blank" '
             + 'rel="noopener" title="open in DepMap">DepMap ↗</a></div>'
             + '<div class="pat-empty">no patent entries for this gene</div>';
      }
      patBox.innerHTML = html;
      patBox.style.display = "block";
      positionPatBox();
    }
    window.addEventListener("resize", positionPatBox);

    function renderPage() {
      // Filter to compounds visible under the currently-ticked plates (Option B:
      // the list itself shrinks/grows with the plate selection).
      var vis = entries.filter(function(e) { return entryVisible(e.t); });
      var total = vis.length;
      var pages = Math.max(1, Math.ceil(total / pageSize));
      if (page < 0) page = 0;
      if (page > pages - 1) page = pages - 1;
      var slice = vis.slice(page * pageSize, page * pageSize + pageSize);
      var html = "";
      for (var i = 0; i < slice.length; i++) {
        var t = slice[i].t, eidx = slice[i].idx;
        var _tm = window.__THUMB_MODE__ || "b64", _td = window.__THUMB_DIR__ || "srb_png";
        var img = t[1]
          ? (_tm === "path"
               ? '<img loading="lazy" src="' + _td + '/' + t[1] + '.png" draggable="false"/>'
               : '<img src="data:image/png;base64,' + t[1] + '" draggable="false"/>')
          : '<div class="noimg">(no structure)</div>';
        var lf, note;
        if (isPaged(t)) {
          var vps = visPlates(t);
          lf = bestLogfc(vps);
          note = vps.length > 1
            ? (vps.length + ' plates')
            : (vps.length === 1 ? vps[0][0] : '');
        } else {
          lf = t[2];
          note = t[5] || '';
        }
        html += '<div class="cell" data-eidx="' + eidx + '" data-cmp="' + (t[0] || '') + '">'
              + img
              + '<div class="cap"><b>' + (t[0] || '') + '</b>'
              + (t[4] ? ' ' + t[4] : '')
              + (lf ? '<br>logfc ' + lf : '')
              + (note ? '<br><span class="pl">' + note + '</span>' : '')
              + '</div></div>';
      }
      row.innerHTML = total ? html
        : '<div class="empty">no compounds on the selected plate(s)</div>';
      if (pages > 1) {
        pager.style.display = "flex";
        indEl.textContent = "page " + (page + 1) + "/" + pages + " · " + total + " compounds";
        prevB.classList.toggle("disabled", page === 0);
        nextB.classList.toggle("disabled", page === pages - 1);
      } else {
        pager.style.display = "none";
      }
      volPinIdx = null;          // re-render (page/plate change) clears the volcano pin
      volBox.style.display = "none";
      volBox.innerHTML = "";
      positionPatBox();
    }

    var GENE_COMPOUNDS = window.__GENE_COMPOUNDS__ || {};
    function render(p) {
      if (!p) return false;
      // customdata is now the gene name (string); the entries live in the map.
      var gene = (typeof p.customdata === "string" && p.customdata)
                 ? p.customdata
                 : ((p.data && p.data.text && p.data.text[p.pointNumber]) || "");
      if (!gene) return false;
      var arr = GENE_COMPOUNDS[gene];
      if (!arr || !arr.length) return false;
      var metaTxt = "";
      var ents = [];
      for (var i = 0; i < arr.length; i++) {
        var t = arr[i];
        if (!t) continue;
        if (t[0] === "__META__") { metaTxt = t[2] || ""; continue; }
        ents.push({t: t, idx: i});
      }
      if (!ents.length) return false;
      currentGene = gene;
      fullArr = arr;
      entries = ents;
      page = 0;
      gn.textContent = gene;
      meta.textContent = metaTxt;
      renderPage();
      renderPatents(gene);
      renderResearch(gene);
      return true;
    }
    function unpin() {
      pinned = false;
      volPinIdx = null;
      box.classList.remove("pinned");
      box.style.display = "none";
      volBox.style.display = "none";
      if (patBox) { patBox.classList.remove("pinned"); patBox.style.display = "none"; }
      if (researchBox) { researchBox.classList.remove("pinned"); researchBox.style.display = "none"; }
    }
    function goPage(delta) {
      var vis = entries.filter(function(e) { return entryVisible(e.t); });
      var pages = Math.max(1, Math.ceil(vis.length / pageSize));
      var np = page + delta;
      if (np < 0 || np > pages - 1) return;
      page = np;
      renderPage();
    }
    prevB.addEventListener("click", function(e) { e.stopPropagation(); goPage(-1); });
    nextB.addEventListener("click", function(e) { e.stopPropagation(); goPage(1); });

    // Build the volcano HTML for a compound cell: one labelled volcano per ticked
    // plate where it passed (plate-aware), or a single volcano (legacy). "" if none.
    // Volcano <img> source: a relative PNG path (cached-folder mode) or an inline
    // base64 blob (embedded mode), per window.__VOLCANO_MODE__. loading="lazy" so
    // the browser only fetches each PNG when its panel is actually shown.
    var VMODE = window.__VOLCANO_MODE__ || "b64";
    var VBASE = window.__VOLCANO_BASE__ || "";   // shared dir prefix, factored out of each row
    function vimg(v) {
      // 'svg' -> interactive <object> (native <title> tooltips on significant
      // points); 'path' -> external PNG <img>; 'b64' -> inline PNG <img>.
      // External modes store only the filename; prepend the shared base here.
      if (VMODE === "svg") {
        var d = VBASE ? VBASE + "/" + v : v;
        return '<object class="vobj" type="image/svg+xml" data="' + d + '"></object>';
      }
      if (VMODE === "path") {
        var p = VBASE ? VBASE + "/" + v : v;
        return '<img loading="lazy" src="' + p + '"/>';
      }
      return '<img loading="lazy" src="' + ("data:image/png;base64," + v) + '"/>';
    }
    function buildVolcanoHtml(cell) {
      var idx = parseInt(cell.getAttribute("data-eidx"), 10);
      var t = fullArr[idx];
      if (!t) return "";
      var cmp = cell.getAttribute("data-cmp") || "";
      var html = "";
      if (isPaged(t)) {
        var vps = visPlates(t);
        if (!vps.length) return "";
        vps.forEach(function(pl) {
          var act = pl[3] ? ' · ' + pl[3] : '';
          var ng = pl[4] ? ' (' + pl[4] + ' genes)' : '';
          var cid = pl[5] || cmp;   // MoleculeBatchID (per plate) when available, else compound
          html += '<div class="vlabel">' + currentGene + ' · ' + cid + ' · '
                + pl[0] + act + ' (logfc ' + pl[1] + ')' + ng + '</div>';
          html += pl[2] ? vimg(pl[2]) : '<div class="vmiss">(no volcano)</div>';
        });
      } else {
        if (!t[3]) return "";
        html = '<div class="vlabel">' + currentGene + ' · ' + cmp + '</div>' + vimg(t[3]);
      }
      return html;
    }
    function showVolcano(cell) {
      var html = buildVolcanoHtml(cell);
      if (!html) { volBox.style.display = "none"; return; }
      volBox.innerHTML = html;
      volBox.style.display = "block";
    }
    function markVolPin(cell) {
      var prev = row.querySelector(".cell.vpin");
      if (prev) prev.classList.remove("vpin");
      if (cell) cell.classList.add("vpin");
    }
    // Hover = peek (only while no volcano is click-pinned). Click = pin/toggle so
    // you can scroll the stacked volcanoes without them vanishing.
    row.addEventListener("mouseover", function(e) {
      if (!pinned || volPinIdx !== null) return;
      var cell = e.target.closest(".cell");
      if (cell) showVolcano(cell);
    });
    row.addEventListener("mouseout", function(e) {
      if (!pinned || volPinIdx !== null) return;
      if (e.relatedTarget && row.contains(e.relatedTarget)) return;
      volBox.style.display = "none";
    });
    row.addEventListener("click", function(e) {
      var cell = e.target.closest(".cell");
      if (!cell) return;
      if (clickMode) {                // click-to-select: a compound row toggles that compound's pin/hide
        e.stopPropagation();
        var _c = cell.getAttribute("data-cmp"), _m = clickMode;
        setTimeout(function() { _m === "hide" ? toggleHideCompound(_c) : togglePinCompound(_c); }, 0);
        return;
      }
      if (!pinned) return;
      e.stopPropagation();
      var idx = parseInt(cell.getAttribute("data-eidx"), 10);
      if (volPinIdx === idx) {        // click the same compound -> unpin
        volPinIdx = null;
        markVolPin(null);
        volBox.style.display = "none";
      } else {                        // pin this compound's volcano(s)
        volPinIdx = idx;
        markVolPin(cell);
        showVolcano(cell);
      }
    });
    gd.on("plotly_hover", function(e) {
      if (pinned) return;
      if (render(e.points && e.points[0])) box.style.display = "block";
      else box.style.display = "none";
    });
    gd.on("plotly_unhover", function() {
      if (pinned) return;
      box.style.display = "none";
      if (patBox) patBox.style.display = "none";
      if (researchBox) researchBox.style.display = "none";
    });
    gd.on("plotly_click", function(e) {
      var _p = e.points && e.points[0];
      if (clickMode && _p) {      // click-to-select: a dot toggles its gene pin/hide
        var _g = (_p.data && _p.data.text && _p.data.text[_p.pointNumber]) || _p.customdata || "";
        var _m = clickMode;       // defer out of Plotly's click dispatch — redraw() re-entrant here hangs
        if (_g) { setTimeout(function() { _m === "hide" ? toggleHideGene(_g) : togglePinGene(_g); }, 0); return; }
      }
      if (render(_p)) {
        pinned = true;
        box.classList.add("pinned");
        box.style.display = "block";
        if (patBox) patBox.classList.add("pinned");
        if (researchBox) researchBox.classList.add("pinned");
      }
    });
    clo.addEventListener("click", unpin);
    document.addEventListener("keydown", function(e) {
      if (!pinned) return;
      if (e.key === "Escape") unpin();
      else if (e.key === "ArrowLeft")  goPage(-1);
      else if (e.key === "ArrowRight") goPage(1);
    });

    // --- checkbox panels (plates + activity) ---
    // One generic group; both filter the same way (Option B): a plate-row is
    // shown only if its plate AND its activity are ticked, and a compound is
    // listed only if it has a visible plate-row.
    function buildGroup(items, tickedMap, boxesEl, allId, noneId) {
      if (!items.length) {
        if (boxesEl.parentNode) boxesEl.parentNode.style.display = "none";
        return false;
      }
      var html = "";
      items.forEach(function(v) {
        html += '<label><input type="checkbox" value="' + v + '"'
              + (tickedMap[v] ? ' checked' : '') + '>' + v + '</label>';
      });
      boxesEl.innerHTML = html;
      boxesEl.addEventListener("change", function(e) {
        if (!e.target || e.target.type !== "checkbox") return;
        tickedMap[e.target.value] = e.target.checked;
        page = 0;
        recolor3d();              // re-colour genes (a gene greys out if it has no
        if (pinned) renderPage(); // compound on the ticked plates/activities)
      });
      function setAll(v) {
        items.forEach(function(it) { tickedMap[it] = v; });
        var cbs = boxesEl.querySelectorAll("input");
        for (var i = 0; i < cbs.length; i++) cbs[i].checked = v;
        page = 0;
        recolor3d();
        if (pinned) renderPage();
      }
      document.getElementById(allId).addEventListener("click", function(e) { e.stopPropagation(); setAll(true); });
      document.getElementById(noneId).addEventListener("click", function(e) { e.stopPropagation(); setAll(false); });
      return true;
    }
    // Plates grouped into collapsible per-date sub-blocks (a "nested dropdown" by
    // date) when __PLATE_DATES__ is supplied; otherwise the plain flat list. Each
    // date carries a tri-state parent checkbox toggling all its plates at once.
    function buildPlateGroup(items, dates, tickedMap, boxesEl, allId, noneId) {
      if (!items.length) { if (boxesEl.parentNode) boxesEl.parentNode.style.display = "none"; return false; }
      if (!dates || !Object.keys(dates).length)
        return buildGroup(items, tickedMap, boxesEl, allId, noneId);  // flat fallback
      var byDate = {}, order = [];
      items.forEach(function(p) {
        var d = dates[p] || "(no date)";
        if (!byDate[d]) { byDate[d] = []; order.push(d); }
        byDate[d].push(p);
      });
      order.sort(function(a, b) {                       // real dates ascending, "(no date)" last
        if (a === "(no date)") return 1;
        if (b === "(no date)") return -1;
        return a < b ? -1 : (a > b ? 1 : 0);
      });
      boxesEl.classList.remove("pf-2col");              // each date sub-block owns its 2-col grid
      var html = "";
      order.forEach(function(d) {
        html += '<div class="pf-date collapsed"><div class="pf-date-head"><span class="fp-caret">&#9662;</span>'
              + '<input type="checkbox" class="pf-date-all" checked>' + d
              + ' <span class="pf-date-n">(' + byDate[d].length + ')</span></div>'
              + '<div class="pf-date-boxes pf-2col">';
        byDate[d].forEach(function(p) {
          html += '<label><input type="checkbox" value="' + p + '"'
                + (tickedMap[p] ? ' checked' : '') + '>' + p + '</label>';
        });
        html += '</div></div>';
      });
      boxesEl.innerHTML = html;
      function syncParent(dateEl) {
        var cbs = dateEl.querySelectorAll(".pf-date-boxes input"), on = 0;
        for (var i = 0; i < cbs.length; i++) if (cbs[i].checked) on++;
        var par = dateEl.querySelector(".pf-date-all");
        par.checked = on === cbs.length;
        par.indeterminate = on > 0 && on < cbs.length;
      }
      var blocks = boxesEl.querySelectorAll(".pf-date");
      for (var i = 0; i < blocks.length; i++) syncParent(blocks[i]);
      boxesEl.addEventListener("change", function(e) {
        var t = e.target; if (!t || t.type !== "checkbox") return;
        if (t.classList.contains("pf-date-all")) {      // parent toggles all plates in its date
          var box = t.closest(".pf-date").querySelectorAll(".pf-date-boxes input");
          for (var j = 0; j < box.length; j++) { box[j].checked = t.checked; tickedMap[box[j].value] = t.checked; }
          t.indeterminate = false;
        } else {
          tickedMap[t.value] = t.checked;
          syncParent(t.closest(".pf-date"));
        }
        page = 0; recolor3d(); if (pinned) renderPage();
      });
      boxesEl.addEventListener("click", function(e) {   // caret/header collapses the date's plate list
        if (e.target.type === "checkbox") return;       // let the parent checkbox toggle instead
        var head = e.target.closest(".pf-date-head");
        if (head) head.parentNode.classList.toggle("collapsed");
      });
      function setAll(v) {
        items.forEach(function(it) { tickedMap[it] = v; });
        var cbs = boxesEl.querySelectorAll("input[type=checkbox]");
        for (var i = 0; i < cbs.length; i++) { cbs[i].checked = v; cbs[i].indeterminate = false; }
        page = 0; recolor3d(); if (pinned) renderPage();
      }
      document.getElementById(allId).addEventListener("click", function(e) { e.stopPropagation(); setAll(true); });
      document.getElementById(noneId).addEventListener("click", function(e) { e.stopPropagation(); setAll(false); });
      return true;
    }
    var hasPlateG = buildPlateGroup(plates, window.__PLATE_DATES__ || null, ticked, pfBoxes, "pf-all", "pf-none");
    var hasActG   = buildGroup(activities, tickedAct, afBoxes, "af-all", "af-none");
    // "Compound" group: one tickbox per compound class (Controls, Contaminants).
    // Unticking a class removes its compounds from every gene (panel, volcanoes, gene
    // greying). Each tickbox shows only if that class has compounds; the group shows if
    // any class does. Its own collapsible group, like Plates / Activity.
    function wireCompoundToggle(id, hasSet, getOn, setOn) {
      var b = document.getElementById(id);
      if (!b) return false;
      if (!hasSet) { if (b.parentNode) b.parentNode.style.display = "none"; return false; }
      b.checked = getOn();   // reflect the configured default (may start unticked)
      b.addEventListener("change", function() {
        setOn(b.checked);
        page = 0;
        recolor3d();          // re-grey genes whose only compounds are in hidden classes
        if (pinned) renderPage();
      });
      return true;
    }
    var _hasCtrl = wireCompoundToggle("control-toggle", Object.keys(controlCompounds).length > 0,
                     function() { return controlOn; }, function(v) { controlOn = v; });
    var _hasCont = wireCompoundToggle("contaminant-toggle", Object.keys(contaminantCompounds).length > 0,
                     function() { return contaminantOn; }, function(v) { contaminantOn = v; });
    var _cg = document.getElementById("compound-group");
    if (_cg && (_hasCtrl || _hasCont)) _cg.style.display = "";
    // Compound-validation group (FBXO31 dependent/independent) — compound-centric mirror
    // of the target Validation filter; reuses buildGroup (its change handler re-runs the mask).
    var hasCmpValG = buildGroup(cmpValCats, tickedCmpVal, vcfBoxes, "vcf-all", "vcf-none");
    var _vcg = document.getElementById("validation-compound-group");
    if (_vcg && hasCmpValG) _vcg.style.display = "";
    // DepMap dependency group — generic checkbox group keyed by category.
    var hasDepG = buildGroup(depmapCats, tickedDep, dfBoxes, "df-all", "df-none");
    var _dg = document.getElementById("depmap-group");
    if (_dg && hasDepG) _dg.style.display = "";
    // Confidence + LoF benefit groups — generic checkbox groups keyed by category.
    var hasConfG = buildGroup(confCats, tickedConf, cnBoxes, "cn-all", "cn-none");
    var _cnG = document.getElementById("confidence-group");
    if (_cnG && hasConfG) _cnG.style.display = "";
    var hasLofG = buildGroup(lofCats, tickedLof, lfBoxes, "lf-all", "lf-none");
    var _lfG = document.getElementById("lof-group");
    if (_lfG && hasLofG) _lfG.style.display = "";
    // Target validation (Y/N) group — generic checkbox group keyed by "Yes"/"No".
    var hasValG = buildGroup(valCats, tickedVal, vfBoxes, "vf-all", "vf-none");
    var _vg = document.getElementById("validation-group");
    if (_vg && hasValG) _vg.style.display = "";
    // Section labels: hide a section header if none of its groups are present.
    var _secC = document.getElementById("sec-compound");
    if (_secC) _secC.style.display = (hasPlateG || hasActG || _hasCtrl || _hasCont || hasCmpValG) ? "" : "none";
    var _secT = document.getElementById("sec-target");
    if (_secT) _secT.style.display = (hasDepG || hasConfG || hasLofG || hasValG) ? "" : "none";
    pf.style.display = "block";   // always shown — the Display (2D/3D) section is always present
    // clicking a group header (not the all/none spans) collapses/expands its boxes
    var _heads = document.querySelectorAll("#filter-panel .pf-head");
    for (var _h = 0; _h < _heads.length; _h++) {
      _heads[_h].addEventListener("click", function() { this.parentNode.classList.toggle("collapsed"); });
    }
    // collapsible whole-section header (PIN/HIDE) — toggles the section + hides #search-group via CSS
    var _secSearch = document.getElementById("sec-search");
    if (_secSearch) _secSearch.addEventListener("click", function() { this.classList.toggle("collapsed"); });

    // --- range sliders (R² / association / MS score) ---
    // Each axis has a dual handle (lo/hi). On change we slice every colour trace
    // (indices R.areaTraces) to the in-range subset, leaving the grey backdrop
    // (trace 0) full. Out-of-range genes therefore appear only as grey dots with
    // no customdata → no compound panel on hover. Labels auto-hide past labelMax.
    var R = window.__RANGES__;
    if (R && typeof Plotly !== "undefined") {
      var rp = document.getElementById("range-panel");
      rp.style.display = "block";          // always visible once sliders are configured
      // park the axis legend immediately above the (variable-height) slider box.
      // Use offsetHeight (stable once laid out) + a ResizeObserver so it tracks the
      // panel's final size instead of a stale pre-layout measurement.
      function positionAxisLegend() {
        if (legEl) legEl.style.bottom = (12 + rp.offsetHeight + 8) + "px";
      }
      positionAxisLegend();
      window.addEventListener("resize", positionAxisLegend);
      if (window.ResizeObserver) new ResizeObserver(positionAxisLegend).observe(rp);
      else setTimeout(positionAxisLegend, 200);
      var AX = ["x", "y", "z"];
      var els = {}, orig = {};
      function fmt(v) { return (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)); }
      // Full per-area arrays come from the injected __AREA_DATA__ (plain numbers).
      // We must NOT read gd.data[ti].x — Plotly stores trace coords as base64
      // {dtype, bdata} objects, not JS arrays (.slice/.length would fail).
      var AREA = window.__AREA_DATA__ || [];
      function captureOrig() {
        orig = {};
        R.areaTraces.forEach(function(ti, i) {
          var d = AREA[i] || {x: [], y: [], z: [], gene: [], hover: []};
          orig[ti] = {x: d.x, y: d.y, z: d.z, text: d.gene, cd: d.gene,
                      hover: d.hover || []};
        });
      }
      var lastMasks = {}, lastTotal = 0;
      // Range-panel tally: proteins + DISTINCT compounds currently SHOWN = in-range
      // ∪ pinned (pins override range/filters), with each protein's compounds still
      // gated by the class/plate/activity filters. Recomputed on filter change AND
      // on pin/unpin (via updateCountHook), reusing the stashed lastMasks.
      function updateCount() {
        var cmpSet = {}, protSet = {};
        R.areaTraces.forEach(function(ti) {
          var o = orig[ti], m = lastMasks[ti]; if (!o || !m) return;
          for (var k = 0; k < m.length; k++) if (m[k]) {
            protSet[o.text[k]] = 1; collectVisibleCompounds(o.text[k], cmpSet);
          }
        });
        Object.keys(effectivePinSet()).forEach(function(g) { protSet[g] = 1; collectVisibleCompounds(g, cmpSet); });
        document.getElementById("rp-count").textContent =
          Object.keys(protSet).length + " proteins — " + Object.keys(cmpSet).length + " compounds";
      }
      // Current selection (in-range ∪ pinned) as CSV: one row per Batch Molecule-Batch ID
      // with the genes it hits joined by "; ". Returns {csv, nRows}.
      function buildExportCSV() {
        var map = {};
        R.areaTraces.forEach(function(ti) {
          var o = orig[ti], m = lastMasks[ti]; if (!o || !m) return;
          for (var k = 0; k < m.length; k++) if (m[k]) collectExport(o.text[k], map);
        });
        Object.keys(effectivePinSet()).forEach(function(g) { collectExport(g, map); });
        function cell(s) {
          s = String(s == null ? "" : s);
          return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
        }
        var keys = Object.keys(map).sort();
        var lines = ["Batch Molecule-Batch ID,genes,Plate,Activity"];
        keys.forEach(function(k) {
          var r = map[k];
          lines.push([cell(r.bid), cell(Object.keys(r.genes).sort().join("; ")),
                      cell(r.plate), cell(r.act)].join(","));
        });
        return {csv: lines.join("\\n"), nRows: keys.length};
      }
      // Build gene-name labels (scene.annotations) for in-range genes, but only for
      // VISIBLE (legend-enabled) area traces, and capped to labelMax labels sampled
      // EVENLY across the MS-score range. Constant-size SVG => no GPU depth-scaling.
      // Capping instead of hide-all means widening the view keeps the top labels rather than clearing them
      // (bug 1); skipping legendonly traces means toggling a disease area hides its
      // labels along with its dots (bug 2).
      function refreshLabels() {
        var cand = [], pinSet = effectivePinSet();
        R.areaTraces.forEach(function(ti) {
          var o = orig[ti], m = lastMasks[ti]; if (!o || !m) return;
          if (gd.data[ti] && gd.data[ti].visible === "legendonly") return;
          for (var k = 0; k < m.length; k++) if (m[k] && !pinSet[o.text[k]]) {
            cand.push({x: o.x[k], y: o.y[k], z: o.z[k], text: o.text[k]});   // pinned genes label via the overlay trace
          }
        });
        if (R.labelMax !== undefined && cand.length > R.labelMax) {
          // Order by MS, then EVENLY sample across the range (stride) so labels spread
          // over the whole cloud instead of all clustering in the high-MS corner.
          cand.sort(function(a, b) { return b.z - a.z; });
          var stride = cand.length / R.labelMax, picked = [];
          for (var s = 0; s < cand.length && picked.length < R.labelMax; s += stride) {
            picked.push(cand[Math.floor(s)]);
          }
          cand = picked;
        }
        var _gx = window.__GENE_XYZ__ || {};   // pinned genes: always labelled (exempt from cap), same 11px font
        Object.keys(pinSet).forEach(function(g) {
          var c = _gx[g]; if (c) cand.push({x: c[0], y: c[1], z: c[2], text: g});
        });
        Plotly.relayout(gd, {"scene.annotations": cand.map(function(c) {
          return {x: c.x, y: c.y, z: c.z, text: c.text, showarrow: false,
                  yshift: 9, font: {size: 11, color: "#000"}};
        })});
      }
      function applyRanges() {
        var b = {};
        AX.forEach(function(a) {
          var loV = parseFloat(els[a].lo.value), hiV = parseFloat(els[a].hi.value);
          if (loV > hiV) { var t = loV; loV = hiV; hiV = t; }
          els[a].val.textContent = fmt(loV) + " – " + fmt(hiV);
          // A range <input> snaps to `step`, so a handle at the very end can land
          // just shy of the data extreme and drop a boundary gene. Treat a handle
          // within one step of its limit as unbounded (filter below is >= / <=).
          var st = R[a].step || 0;
          b[a] = [(loV <= R[a].min + st) ? -Infinity : loV,
                  (hiV >= R[a].max - st) ?  Infinity : hiV];
        });
        var total = 0, masks = {};
        R.areaTraces.forEach(function(ti) {
          var o = orig[ti]; if (!o) return;
          var m = [];
          for (var k = 0; k < o.x.length; k++) {
            // in slider range AND has a compound on a ticked plate+activity
            var inr = o.x[k] >= b.x[0] && o.x[k] <= b.x[1]
                   && o.y[k] >= b.y[0] && o.y[k] <= b.y[1]
                   && o.z[k] >= b.z[0] && o.z[k] <= b.z[1]
                   && geneHasVisibleCompound(o.text[k])
                   && depAllowed(o.text[k])
                   && confAllowed(o.text[k])
                   && lofAllowed(o.text[k])
                   && valAllowed(o.text[k])
                   && !isHiddenGene(o.text[k]);   // hidden genes/compound-targets drop from the view
            m.push(inr); if (inr) total++;
          }
          masks[ti] = m;
        });
        // Mutate the trace data directly, then force a full redraw (Plotly.restyle of
        // x/y/z on a gl3d scatter3d updates the data but doesn't reliably repaint the
        // 3D scene — Plotly.redraw() does).
        R.areaTraces.forEach(function(ti) {
          var o = orig[ti], m = masks[ti]; if (!o || !m || !gd.data[ti]) return;
          var fx = [], fy = [], fz = [], ft = [], fcd = [], fhov = [];
          for (var k = 0; k < m.length; k++) if (m[k]) {
            fx.push(o.x[k]); fy.push(o.y[k]); fz.push(o.z[k]);
            ft.push(o.text[k]); fcd.push(o.cd[k]); fhov.push(o.hover[k]);
          }
          var tr = gd.data[ti];
          tr.x = fx; tr.y = fy; tr.z = fz; tr.text = ft; tr.customdata = fcd;
          tr.hovertext = fhov;   // keep the tooltip aligned with the filtered points
        });
        // Labels are rebuilt by refreshLabels() (scene.annotations) — see there for the
        // visible-traces + top-N-by-MS logic. Stash the masks/total so a later legend
        // toggle can re-derive labels without recomputing the range filter.
        lastMasks = masks; lastTotal = total;
        Plotly.redraw(gd);
        refreshLabels();
        updateCount();
      }
      recolor3d = applyRanges;   // let the Plate/Activity checkboxes re-colour too
      refreshLabelsHook = refreshLabels;   // expose to the pin block (label dedup)
      updateCountHook = updateCount;       // expose to the pin block (tally includes pins)
      exportCSVHook = buildExportCSV;      // expose to the download block
      // Toggling a disease area in the legend changes trace visibility (plotly fires
      // plotly_restyle); re-derive labels so hidden areas drop their labels too.
      if (gd.on) gd.on("plotly_restyle", refreshLabels);
      AX.forEach(function(a) {
        var cfg = R[a];
        var lo = document.getElementById(a + "-lo");
        var hi = document.getElementById(a + "-hi");
        [lo, hi].forEach(function(s) { s.min = cfg.min; s.max = cfg.max; s.step = cfg.step; });
        lo.value = (cfg.lo !== undefined ? cfg.lo : cfg.min);   // default = focused corner box
        hi.value = (cfg.hi !== undefined ? cfg.hi : cfg.max);
        document.getElementById(a + "-name").textContent = cfg.label;
        els[a] = {lo: lo, hi: hi, val: document.getElementById(a + "-val")};
        lo.addEventListener("input", applyRanges);
        hi.addEventListener("input", applyRanges);
      });
      document.getElementById("rp-reset").addEventListener("click", function() {
        AX.forEach(function(a) {
          els[a].lo.value = (R[a].lo !== undefined ? R[a].lo : R[a].min);
          els[a].hi.value = (R[a].hi !== undefined ? R[a].hi : R[a].max);
        });
        applyRanges();
      });
      // gd.data may not be populated at DOMContentLoaded — poll briefly, then init.
      var _need = R.areaTraces.length ? R.areaTraces[R.areaTraces.length - 1] : 0;
      (function tryInit(n) {
        if (gd.data && gd.data.length > _need) { captureOrig(); applyRanges(); }
        else if (n > 0) { setTimeout(function() { tryInit(n - 1); }, 100); }
        else { captureOrig(); applyRanges(); }
      })(50);
    }

    // --- gene search + pin overlay --------------------------------------------------
    // A search box with substring autocomplete; pinning a gene shows it as a gold
    // diamond on the dedicated overlay trace (window.__PIN_TRACE__) that no filter
    // touches, so it stays visible regardless of sliders/ticks. Hovering it still
    // opens the compound panel (customdata=gene), whose contents respect activity/plate.
    (function() {
      var PIN_TRACE = window.__PIN_TRACE__;
      var ALL_GENES = window.__ALL_GENES__ || [];
      var ALL_COMPOUNDS = window.__ALL_COMPOUNDS__ || [];
      var GENE_XYZ  = window.__GENE_XYZ__ || {};
      var GENE_COLOR = window.__GENE_COLOR__ || {};
      var COMPOUND_GENES = window.__COMPOUND_GENES__ || {};
      var searchEl = document.getElementById("gene-search");
      var acEl     = document.getElementById("gene-ac");
      var selBtn   = document.getElementById("select-btn");
      var boxEl    = document.getElementById("pinned-box");
      var hideEl    = document.getElementById("hide-search");
      var hideAcEl  = document.getElementById("hide-ac");
      var hideBtn   = document.getElementById("hide-btn");
      var hideBoxEl = document.getElementById("hidden-box");
      if (!searchEl || !boxEl || PIN_TRACE == null) return;

      function redrawPins() {
        var genes = Object.keys(effectivePinSet());
        var xs = [], ys = [], zs = [], ts = [], cds = [], hov = [], cols = [];
        genes.forEach(function(g) {
          var c = GENE_XYZ[g]; if (!c) return;
          xs.push(c[0]); ys.push(c[1]); zs.push(c[2]); ts.push(g); cds.push(g); hov.push(g);
          cols.push(GENE_COLOR[g] || "#1D3557");   // colour by disease/pharma category
        });
        if (typeof Plotly !== "undefined" && gd.data && gd.data[PIN_TRACE]) {
          var tr = gd.data[PIN_TRACE];
          tr.x = xs; tr.y = ys; tr.z = zs; tr.text = ts; tr.customdata = cds; tr.hovertext = hov;
          tr.marker.color = cols;
          Plotly.redraw(gd);
        }
        refreshLabelsHook();   // pinned genes label via scene.annotations (dedup with range labels)
        updateCountHook();     // pinned proteins + their compounds enter the tally
      }
      function nTargets(c) {
        return (COMPOUND_GENES[c] || []).filter(function(g) { return GENE_XYZ[g]; }).length;
      }
      function renderBox() {
        // hide supersedes pin in the chip list: a pinned item that's also hidden shows
        // only in the Hidden box (it reappears here when unhidden — the pin is retained).
        var gs = pinnedGenes.filter(function(g) { return hiddenGenes.indexOf(g) === -1; });
        var cs = pinnedCompounds.filter(function(c) { return hiddenCompounds.indexOf(c) === -1; });
        var total = gs.length + cs.length;
        if (!total) { boxEl.style.display = "none"; boxEl.innerHTML = ""; return; }
        boxEl.style.display = "block";
        var h = '<div class="pin-hd">Pinned (' + total + ') <span id="pin-clear">clear</span></div>';
        cs.forEach(function(c) {
          h += '<span class="pin-chip pin-cmp">' + c + ' <span class="pin-n">(' + nTargets(c) + ')</span>'
             + '<span class="pin-x" data-c="' + c + '">×</span></span>';
        });
        gs.forEach(function(g) {
          h += '<span class="pin-chip">' + g + '<span class="pin-x" data-g="' + g + '">×</span></span>';
        });
        boxEl.innerHTML = h;
      }
      function pinGene(g) {
        if (!g || !GENE_XYZ[g] || pinnedGenes.indexOf(g) !== -1) return;
        pinnedGenes.push(g); redrawPins(); renderBox();
      }
      function unpinGene(g) {
        var i = pinnedGenes.indexOf(g);
        if (i >= 0) { pinnedGenes.splice(i, 1); redrawPins(); renderBox(); }
      }
      function pinCompound(c) {
        if (!c || !COMPOUND_GENES[c] || pinnedCompounds.indexOf(c) !== -1) return;
        pinnedCompounds.push(c); redrawPins(); renderBox();
      }
      function unpinCompound(c) {
        var i = pinnedCompounds.indexOf(c);
        if (i >= 0) { pinnedCompounds.splice(i, 1); redrawPins(); renderBox(); }
      }
      // expose toggles for click-to-pin (dot -> gene, panel row -> compound)
      togglePinGene = function(g) {
        if (!g) return;
        if (pinnedGenes.indexOf(g) !== -1) unpinGene(g); else pinGene(g);
      };
      togglePinCompound = function(c) {
        if (!c) return;
        if (pinnedCompounds.indexOf(c) !== -1) unpinCompound(c); else pinCompound(c);
      };
      // --- HIDE: directly-hidden genes/compounds; recolor3d re-runs the mask (which now
      // excludes hidden) so hidden dots vanish; redrawPins drops them from the overlay too.
      function afterHideChange() {
        rebuildHidden(); recolor3d(); redrawPins(); renderHideBox(); renderBox();
        // re-render the open compound panel so a hidden compound's thumbnail drops live
        if (box && box.style.display !== "none" && currentGene) renderPage();
      }
      function hideGene(g) {
        if (!g || !GENE_XYZ[g] || hiddenGenes.indexOf(g) !== -1) return;
        hiddenGenes.push(g); afterHideChange();
      }
      function unhideGene(g) {
        var i = hiddenGenes.indexOf(g);
        if (i >= 0) { hiddenGenes.splice(i, 1); afterHideChange(); }
      }
      function hideCompound(c) {
        if (!c || !COMPOUND_GENES[c] || hiddenCompounds.indexOf(c) !== -1) return;
        hiddenCompounds.push(c); afterHideChange();
      }
      function unhideCompound(c) {
        var i = hiddenCompounds.indexOf(c);
        if (i >= 0) { hiddenCompounds.splice(i, 1); afterHideChange(); }
      }
      toggleHideGene = function(g) {
        if (!g) return;
        if (hiddenGenes.indexOf(g) !== -1) unhideGene(g); else hideGene(g);
      };
      toggleHideCompound = function(c) {
        if (!c) return;
        if (hiddenCompounds.indexOf(c) !== -1) unhideCompound(c); else hideCompound(c);
      };
      function renderHideBox() {
        var total = hiddenGenes.length + hiddenCompounds.length;
        if (!total) { hideBoxEl.style.display = "none"; hideBoxEl.innerHTML = ""; return; }
        hideBoxEl.style.display = "block";
        var h = '<div class="pin-hd">Hidden (' + total + ') <span id="hide-clear">clear</span></div>';
        hiddenCompounds.forEach(function(c) {
          h += '<span class="pin-chip hide-chip">' + c + ' <span class="pin-n">(' + nTargets(c) + ')</span>'
             + '<span class="pin-x" data-c="' + c + '">×</span></span>';
        });
        hiddenGenes.forEach(function(g) {
          h += '<span class="pin-chip hide-chip">' + g + '<span class="pin-x" data-g="' + g + '">×</span></span>';
        });
        hideBoxEl.innerHTML = h;
      }
      // re-render pins/hides from the (externally mutated) arrays — used by session-load
      applyPinHideHook = function() { rebuildHidden(); redrawPins(); renderBox(); renderHideBox(); };

      // mutually-exclusive click modes: Pin (orange) / Hide (red)
      function setMode(mode) {       // "", "pin", "hide"
        clickMode = mode;
        if (selBtn) selBtn.classList.toggle("active", mode === "pin");
        if (hideBtn) hideBtn.classList.toggle("active", mode === "hide");
        document.body.classList.toggle("select-mode", mode === "pin");
        document.body.classList.toggle("hide-mode", mode === "hide");
      }
      if (selBtn) selBtn.addEventListener("click", function() { setMode(clickMode === "pin" ? "" : "pin"); });
      if (hideBtn) hideBtn.addEventListener("click", function() { setMode(clickMode === "hide" ? "" : "hide"); });

      function pinChoice(ch) { if (ch) { if (ch.t === "cmp") pinCompound(ch.v); else pinGene(ch.v); } }
      function hideChoice(ch) { if (ch) { if (ch.t === "cmp") hideCompound(ch.v); else hideGene(ch.v); } }
      function exactMatch(q) {
        q = (q || "").trim().toUpperCase(); if (!q) return null;
        for (var i = 0; i < ALL_GENES.length; i++)
          if (ALL_GENES[i].toUpperCase() === q) return {v: ALL_GENES[i], t: "gene"};
        for (var j = 0; j < ALL_COMPOUNDS.length; j++)
          if (ALL_COMPOUNDS[j].toUpperCase() === q) return {v: ALL_COMPOUNDS[j], t: "cmp"};
        return null;
      }
      // generic substring autocomplete over genes + compounds; onPick({v,t}) does the action
      function wireSearch(inp, ac, onPick) {
        var items = [], data = [], active = -1;
        function hilite() { for (var i = 0; i < items.length; i++) items[i].className = "ac-item" + (i === active ? " active" : ""); }
        function chosen() { if (active >= 0 && data[active]) return data[active]; return exactMatch(inp.value) || data[0] || null; }
        function render() {
          var q = (inp.value || "").trim().toUpperCase();
          ac.innerHTML = ""; items = []; data = []; active = -1;
          if (!q) { ac.style.display = "none"; return; }
          var matches = [];
          for (var i = 0; i < ALL_GENES.length && matches.length < 12; i++)
            if (ALL_GENES[i].toUpperCase().indexOf(q) !== -1) matches.push({v: ALL_GENES[i], t: "gene"});
          for (var j = 0; j < ALL_COMPOUNDS.length && matches.length < 12; j++)
            if (ALL_COMPOUNDS[j].toUpperCase().indexOf(q) !== -1) matches.push({v: ALL_COMPOUNDS[j], t: "cmp"});
          if (!matches.length) { ac.innerHTML = '<div class="ac-empty">no match</div>'; ac.style.display = "block"; return; }
          matches.forEach(function(mm) {
            var d = document.createElement("div"); d.className = "ac-item";
            d.innerHTML = mm.v + (mm.t === "cmp" ? ' <span class="ac-tag">compound</span>' : '');
            d.addEventListener("mousedown", function(e) { e.preventDefault(); onPick(mm); inp.value = ""; render(); });
            ac.appendChild(d); items.push(d); data.push(mm);
          });
          ac.style.display = "block";
        }
        inp.addEventListener("input", render);
        inp.addEventListener("keydown", function(e) {
          if (e.key === "ArrowDown") { e.preventDefault(); active = Math.min(active + 1, items.length - 1); hilite(); }
          else if (e.key === "ArrowUp") { e.preventDefault(); active = Math.max(active - 1, 0); hilite(); }
          else if (e.key === "Enter") { e.preventDefault(); onPick(chosen()); inp.value = ""; render(); }
          else if (e.key === "Escape") { inp.value = ""; render(); }
        });
        document.addEventListener("click", function(e) { if (e.target !== inp && !ac.contains(e.target)) ac.style.display = "none"; });
      }
      wireSearch(searchEl, acEl, pinChoice);
      if (hideEl && hideAcEl) wireSearch(hideEl, hideAcEl, hideChoice);

      boxEl.addEventListener("click", function(e) {
        if (e.target.id === "pin-clear") { pinnedGenes = []; pinnedCompounds = []; redrawPins(); renderBox(); }
        else if (e.target.classList.contains("pin-x")) {
          if (e.target.getAttribute("data-c")) unpinCompound(e.target.getAttribute("data-c"));
          else unpinGene(e.target.getAttribute("data-g"));
        }
      });
      if (hideBoxEl) hideBoxEl.addEventListener("click", function(e) {
        if (e.target.id === "hide-clear") { hiddenGenes = []; hiddenCompounds = []; afterHideChange(); }
        else if (e.target.classList.contains("pin-x")) {
          if (e.target.getAttribute("data-c")) unhideCompound(e.target.getAttribute("data-c"));
          else unhideGene(e.target.getAttribute("data-g"));
        }
      });
    })();

    // --- download selection (CSV) ---------------------------------------------------
    // Exports the current selection (in-range ∪ pinned proteins) as
    // "Batch Molecule-Batch ID,genes". A standalone file:// page can't write to a chosen
    // path silently, so we use showSaveFilePicker when available (lets you pick the folder,
    // e.g. next to the HTML) and fall back to a normal browser download otherwise.
    (function() {
      var btn  = document.getElementById("dl-btn");
      var note = document.getElementById("dl-note");
      if (!btn) return;
      function fname() {   // timestamped default; the browser save dialog lets you rename
        var d = new Date(), p = function(n) { return (n < 10 ? "0" : "") + n; };
        return "" + d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate())
             + "_" + p(d.getHours()) + "_" + p(d.getMinutes()) + "_" + p(d.getSeconds()) + ".csv";
      }
      btn.addEventListener("click", async function() {
        var out = exportCSVHook();
        if (!out) { note.textContent = "export unavailable (no range data)"; return; }
        if (!out.nRows) { note.textContent = "current selection is empty"; return; }
        var name = fname();
        var blob = new Blob([out.csv], {type: "text/csv;charset=utf-8"});
        try {
          if (window.showSaveFilePicker) {
            var h = await window.showSaveFilePicker({suggestedName: name,
              types: [{description: "CSV", accept: {"text/csv": [".csv"]}}]});
            var w = await h.createWritable(); await w.write(blob); await w.close();
            note.textContent = "✓ saved " + h.name + " (" + out.nRows + " rows)";
            return;
          }
        } catch (e) { if (e && e.name === "AbortError") return; }   // cancelled, or fall through
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = name;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        setTimeout(function() { URL.revokeObjectURL(a.href); }, 1000);
        note.textContent = "✓ downloaded " + name + " (" + out.nRows + " rows) → Downloads";
      });
    })();

    // --- session save / load (.iface) -----------------------------------------------
    // Captures the full interface state (pins, hides, every filter tickbox, the three
    // range sliders, 2D/3D mode + camera) to a .iface JSON file and restores it on load,
    // so a saved session reproduces exactly what was on screen. file:// can't write
    // silently → showSaveFilePicker when available, else a normal download; load reads a
    // user-picked file via FileReader.
    (function() {
      var saveBtn = document.getElementById("sess-save-btn");
      var loadBtn = document.getElementById("sess-load-btn");
      var fileIn  = document.getElementById("sess-file");
      var note    = document.getElementById("sess-note");
      if (!saveBtn || !loadBtn || !fileIn) return;
      var AXS = ["x", "y", "z"];

      function gather() {
        var s = {version: 1,
                 pinnedGenes: pinnedGenes.slice(), pinnedCompounds: pinnedCompounds.slice(),
                 hiddenGenes: hiddenGenes.slice(), hiddenCompounds: hiddenCompounds.slice(),
                 filters: {plates: Object.assign({}, ticked), activity: Object.assign({}, tickedAct),
                           control: controlOn, contaminant: contaminantOn,
                           depmap: Object.assign({}, tickedDep), confidence: Object.assign({}, tickedConf),
                           lof: Object.assign({}, tickedLof), validationTarget: Object.assign({}, tickedVal),
                           validationCompound: Object.assign({}, tickedCmpVal)}};
        if (els && els.x && els.x.lo) {
          s.ranges = {};
          AXS.forEach(function(a) { if (els[a]) s.ranges[a] = [els[a].lo.value, els[a].hi.value]; });
        }
        var seg2d = document.querySelector("#disp-toggle .seg2d");
        s.mode2d = !!(seg2d && seg2d.classList.contains("active"));
        try {
          var cam = gd && gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
          if (cam) s.camera = JSON.parse(JSON.stringify(cam));
        } catch (e) {}
        return s;
      }

      // overwrite a ticked-map + sync its checkboxes from a saved {value: bool} object
      function applyGroup(map, el, saved) {
        if (!saved) return;
        Object.keys(saved).forEach(function(k) { map[k] = saved[k]; });
        if (el) Array.prototype.forEach.call(el.querySelectorAll("input[value]"), function(cb) {
          if (Object.prototype.hasOwnProperty.call(saved, cb.value)) cb.checked = !!saved[cb.value];
        });
      }
      // after setting plate leaves, re-derive each date's tri-state parent checkbox
      function refreshPlateParents() {
        Array.prototype.forEach.call(document.querySelectorAll("#pf-boxes .pf-date"), function(d) {
          var kids = d.querySelectorAll(".pf-date-boxes input[value]"), all = kids.length > 0;
          for (var i = 0; i < kids.length; i++) if (!kids[i].checked) { all = false; break; }
          var par = d.querySelector(".pf-date-all"); if (par) par.checked = all;
        });
      }
      function setArr(arr, vals) { arr.length = 0; (vals || []).forEach(function(v) { arr.push(v); }); }

      function apply(s) {
        if (!s || typeof s !== "object") throw new Error("not a session object");
        setArr(pinnedGenes, s.pinnedGenes); setArr(pinnedCompounds, s.pinnedCompounds);
        setArr(hiddenGenes, s.hiddenGenes); setArr(hiddenCompounds, s.hiddenCompounds);
        applyPinHideHook();                       // rebuild hidden cache + re-render pins/hides
        if (s.ranges && els && els.x) AXS.forEach(function(a) {
          if (s.ranges[a] && els[a]) { els[a].lo.value = s.ranges[a][0]; els[a].hi.value = s.ranges[a][1]; }
        });
        var f = s.filters || {};
        applyGroup(ticked, document.getElementById("pf-boxes"), f.plates); refreshPlateParents();
        applyGroup(tickedAct, document.getElementById("af-boxes"), f.activity);
        applyGroup(tickedDep, document.getElementById("df-boxes"), f.depmap);
        applyGroup(tickedConf, document.getElementById("cn-boxes"), f.confidence);
        applyGroup(tickedLof, document.getElementById("lf-boxes"), f.lof);
        applyGroup(tickedVal, document.getElementById("vf-boxes"), f.validationTarget);
        applyGroup(tickedCmpVal, document.getElementById("vcf-boxes"), f.validationCompound);
        if (typeof f.control === "boolean") {
          controlOn = f.control; var cb = document.getElementById("control-toggle"); if (cb) cb.checked = controlOn;
        }
        if (typeof f.contaminant === "boolean") {
          contaminantOn = f.contaminant; var cb2 = document.getElementById("contaminant-toggle"); if (cb2) cb2.checked = contaminantOn;
        }
        recolor3d();                              // re-run the mask (filters+hidden+ranges) + labels + count
        if (typeof s.mode2d === "boolean") setMode2DHook(s.mode2d);
        if (s.camera) { try { Plotly.relayout(gd, {"scene.camera": s.camera}); } catch (e) {} }
      }

      function sfname() {
        var d = new Date(), p = function(n) { return (n < 10 ? "0" : "") + n; };
        return "" + d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate())
             + "_" + p(d.getHours()) + p(d.getMinutes()) + "_session.iface";
      }
      saveBtn.addEventListener("click", async function() {
        var blob = new Blob([JSON.stringify(gather(), null, 1)], {type: "application/json"});
        var name = sfname();
        try {
          if (window.showSaveFilePicker) {
            var h = await window.showSaveFilePicker({suggestedName: name,
              types: [{description: "Interface session", accept: {"application/json": [".iface"]}}]});
            var w = await h.createWritable(); await w.write(blob); await w.close();
            note.textContent = "✓ saved " + h.name; return;
          }
        } catch (e) { if (e && e.name === "AbortError") return; }
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = name;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        setTimeout(function() { URL.revokeObjectURL(a.href); }, 1000);
        note.textContent = "✓ downloaded " + name + " → Downloads";
      });
      loadBtn.addEventListener("click", function() { fileIn.value = ""; fileIn.click(); });
      fileIn.addEventListener("change", function() {
        var file = fileIn.files && fileIn.files[0];
        if (!file) return;
        var r = new FileReader();
        r.onload = function() {
          try {
            var s = JSON.parse(r.result);
            apply(s);
            var nHid = (s.hiddenGenes || []).length + (s.hiddenCompounds || []).length;
            var nPin = (s.pinnedGenes || []).length + (s.pinnedCompounds || []).length;
            note.textContent = "✓ loaded " + file.name + " (" + nPin + " pinned, " + nHid + " hidden)";
          } catch (e) { note.textContent = "✗ could not load: " + ((e && e.message) || e); }
        };
        r.readAsText(file);
      });
    })();
  });
</script>
'''


def plot_target_3d(
    target_final,
    *,
    must_include=(),
    exclude_genes=(),
    max_fold_plot=500,
    top_n_highlight=50,
    min_r2_highlight=0.10,
    min_os_auto=0.60,
    top_n_hover=5,
    png_dir='data/srb_png',
    df_raw=None,
    volcano_size_px=350,
    volcano_xlim=(-5.0, 5.0),
    volcano_n_jobs=1,
    compound_meta_df=None,
    compound_meta_icons=None,
    gene_patents_df=None,
    gene_patents_top_n=5,
    depmap_url_template='https://depmap.org/portal/gene/{gene}',
    disease_area_colors=None,
    na_area_color='#bbbbbb',
    title='SAR predictability × disease relevance × MCS fold-enrichment',
    html_path=None,
    height=900,
    width=1500,
    nb_display=True,
):
    """
    3D scatter of (R², overall_score, fold) for the ``target_final`` shortlist.

    Highlights:
      * the top ``top_n_highlight`` genes closest to the (↑, ↑, ↑) corner,
      * all genes with overall_score > ``min_os_auto``,
      * everything in ``must_include`` (bypasses every filter, fold clipped for plotting).

    Genes below the R² noise floor (``min_r2_highlight``) are NOT auto-highlighted
    but still appear in the lightgrey backdrop. Highlighted points are coloured
    by ``disease_area``; genes outside the priority dict get ``na_area_color``.

    If ``html_path`` is set, also writes a standalone HTML with on-hover
    structure previews (top-N down-modulators per gene from ``top1_smiles``
    … ``topN_smiles``, embedded as base64 PNGs).

    :param df target_final: must contain at least ``gene``, ``R2``, ``overall_score``,
        ``fold``, ``disease_area``, and ``top1_compound``/``top1_logfc``/``top1_smiles``
        … ``topN_*`` columns (produced by the cell that adds top down-modulators).
    :return: ``(fig, highlighted)`` — the Plotly figure and the highlighted-set DataFrame.
    """
    import io, base64
    import plotly.graph_objects as go
    from rdkit import Chem
    from rdkit.Chem import Draw

    if disease_area_colors is None:
        disease_area_colors = {}

    # 1) filter target_final → plot_df, with must_include bypassing both filters
    required_cols = ['R2', 'overall_score', 'fold']
    missing = [c for c in required_cols if c not in target_final.columns]
    assert not missing, f'target_final is missing {missing}'

    plot_df = target_final.dropna(subset=required_cols).copy()
    n0 = len(plot_df)
    must_set = set(must_include)
    is_must = plot_df['gene'].isin(must_set)
    dropped_named = plot_df[plot_df['gene'].isin(exclude_genes) & ~is_must]
    dropped_fold  = plot_df[(plot_df['fold'] > max_fold_plot)
                              & ~plot_df['gene'].isin(exclude_genes)
                              & ~is_must]
    plot_df = plot_df[
        is_must
        | (~plot_df['gene'].isin(exclude_genes) & (plot_df['fold'] <= max_fold_plot))
    ]

    plot_df['fold_plot'] = plot_df['fold'].clip(upper=max_fold_plot)
    clipped = plot_df.loc[plot_df['fold'] > max_fold_plot, ['gene', 'fold']]

    print(f'> {len(plot_df):,} / {n0:,} genes after excluding outliers')
    if len(dropped_named):
        print(f'  [excluded by name]  {list(dropped_named["gene"])}')
    if len(dropped_fold):
        print(f'  [excluded fold>{max_fold_plot}]  '
              f'{dropped_fold[["gene", "fold"]].head(10).to_dict("records")}')
    if len(clipped):
        print(f'  [clipped fold>{max_fold_plot} for plotting (still shown)]  '
              f'{clipped.to_dict("records")}')

    # 2) corner-distance ranking (uses log10 of fold so the linear span doesn't dominate)
    plot_df['log_fold'] = np.log10(plot_df['fold'].clip(lower=0.01))
    def _norm01(s):
        return (s - s.min()) / (s.max() - s.min())
    xn = _norm01(plot_df['R2'])
    yn = _norm01(plot_df['overall_score'])
    zn = _norm01(plot_df['log_fold'])
    plot_df['_dist'] = np.sqrt((1 - xn) ** 2 + (1 - yn) ** 2 + (1 - zn) ** 2)

    candidates = plot_df[plot_df['R2'] >= min_r2_highlight]
    top_n   = candidates.nsmallest(top_n_highlight, '_dist')
    auto_os = candidates[candidates['overall_score'] > min_os_auto]
    must    = plot_df[plot_df['gene'].isin(must_set)]
    miss = [g for g in must_include if g not in plot_df['gene'].values]
    if miss:
        print(f'  [warn] must_include not found: {miss}')
    highlighted = pd.concat([top_n, auto_os, must]).drop_duplicates('gene')
    print(f'  [highlight] corner-top-{top_n_highlight}={len(top_n)}, '
          f'OS>{min_os_auto}: {len(auto_os)}, must={len(must)}, '
          f'union={len(highlighted)} (R² floor = {min_r2_highlight})')

    # 3) per-gene structure thumbnails -> customdata
    needed = [f'top{k}_{n}' for k in range(1, top_n_hover + 1)
                            for n in ('compound', 'logfc', 'smiles')]
    assert set(needed).issubset(highlighted.columns), (
        f'highlighted is missing top1..top{top_n_hover} columns'
    )

    # source-of-image preference: data/srb_png/<compound>.png  →  RDKit-from-SMILES
    _stats = {'png': 0, 'rdkit': 0, 'miss': 0}

    def _compound_b64(compound, smi, size=(170, 110)):
        if isinstance(compound, str) and compound and png_dir:
            p = os.path.join(png_dir, f'{compound}.png')
            if os.path.isfile(p):
                with open(p, 'rb') as fh:
                    _stats['png'] += 1
                    return base64.b64encode(fh.read()).decode()
        if isinstance(smi, str) and smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                img = Draw.MolToImage(m, size=size)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                _stats['rdkit'] += 1
                return base64.b64encode(buf.getvalue()).decode()
        _stats['miss'] += 1
        return ''

    # Pre-index compound metadata for fast O(1) lookup during the per-gene loop.
    # `compound_meta_icons` shape:
    #   { 'col_name': {
    #         'icon':    str,                  # emoji / character to render
    #         'color':   str,                  # css color
    #         'tooltip': str,                  # html title attr (optional)
    #         'show_if': callable(v)->bool,    # default: pd.notna(v) and bool(v)
    #         'label':   callable(v)->str,     # optional text after the icon
    #     }, ... }
    meta_index = None
    if compound_meta_df is not None and compound_meta_icons:
        # set_index('compound').to_dict('index') requires a unique index, so
        # dedup defensively. Take the first row per compound — caller can
        # pre-aggregate (groupby + agg) if smarter merging is needed.
        n0 = len(compound_meta_df)
        cm = compound_meta_df.drop_duplicates('compound', keep='first')
        if len(cm) < n0:
            print(f'  [warn] compound_meta_df: deduped {n0 - len(cm):,} duplicate compound rows '
                  f'(keeping first); pre-aggregate yourself for different semantics')
        meta_index = cm.set_index('compound').to_dict('index')

    def _meta_html(compound_id):
        if not meta_index or not compound_id:
            return ''
        row = meta_index.get(compound_id, {}) or {}
        parts = []
        for col, cfg in compound_meta_icons.items():
            v = row.get(col)
            tooltip = cfg.get('tooltip', col)

            # --- Mode A: state_map (always-show: render a fixed icon per value)
            #   {'state_map': {'yes': {'icon':'✅', 'color':'#2A9D8F'},
            #                  'no':  {'icon':'❌', 'color':'#E63946'},
            #                  '/':   {'icon':'❓', 'color':'#999'}}}
            # Useful when you want a consistent N-slot row regardless of value.
            state_map = cfg.get('state_map')
            if state_map is not None:
                key = (str(v).strip().lower() if pd.notna(v) else None)
                state = (state_map.get(key) or state_map.get(v)
                         or state_map.get('__default__', {'icon': '❓', 'color': '#bbb'}))
                icon  = state.get('icon', '❓')
                color = state.get('color', '#bbb')
                parts.append(
                    f'<span title="{tooltip}: {v}" '
                    f'style="color:{color};font-weight:600;margin-left:3px;">'
                    f'{icon}</span>'
                )
                continue

            # --- Mode B: show_if (legacy: show/hide a single icon)
            show_if = cfg.get('show_if', lambda x: pd.notna(x) and bool(x))
            try:
                ok = show_if(v)
            except Exception:
                ok = False
            if not ok:
                continue
            icon    = cfg.get('icon', '•')
            color   = cfg.get('color', '#666')
            label   = cfg.get('label', lambda _v: '')(v) if callable(cfg.get('label')) else cfg.get('label', '')
            parts.append(
                f'<span title="{tooltip}: {v}" '
                f'style="color:{color};font-weight:600;margin-left:3px;">'
                f'{icon}{label}</span>'
            )
        return ''.join(parts)

    custom = {}
    for _, row in highlighted.iterrows():
        triples = []
        # Index 0 is a per-gene META row: ['__META__', '', '<fisher_p str>', '', ''].
        # The hover JS detects '__META__' to populate the panel header; the
        # existing compound-render loop skips it because t[1] (b64) is empty.
        # Pad to 5 elements so all rows in customdata have a consistent shape.
        fp_val = row.get('fisher_p') if 'fisher_p' in highlighted.columns else None
        if fp_val is None or pd.isna(fp_val):
            fp_str = '—'
        else:
            fp_str = '< 0.0001' if fp_val < 0.0001 else f'{fp_val:.4f}'
        triples.append(['__META__', '', f'fisher_p={fp_str}', '', ''])
        for k in range(1, top_n_hover + 1):
            c = row.get(f'top{k}_compound')
            s = row.get(f'top{k}_smiles')
            l = row.get(f'top{k}_logfc')
            c_str = str(c) if pd.notna(c) else ''
            triples.append([
                c_str,
                _compound_b64(c if pd.notna(c) else None,
                              s if pd.notna(s) else None),
                f'{l:.2f}' if pd.notna(l) else '',
                # index 3 reserved for volcano b64 (filled in 3b below);
                # index 4 is the compound-meta HTML snippet
                '',
                _meta_html(c_str),
            ])
        custom[row['gene']] = triples

    n_thumbs = _stats['png'] + _stats['rdkit']
    print(f'> built {n_thumbs:,} structure thumbnails across {len(custom)} highlighted genes '
          f'(png={_stats["png"]}, rdkit-fallback={_stats["rdkit"]}, missing={_stats["miss"]}; '
          f'png_dir={png_dir!r})')

    # 3b) optional per-(gene, compound) volcano thumbnails. Each compound row
    #     already has index 3 reserved (set to '' during the build above). This
    #     step *fills in* that slot; padding is unnecessary since the slot
    #     exists. JS reads t[3] for the volcano payload.
    if df_raw is not None:
        # Build the task list once; rows without a compound id keep '' at idx 3.
        tasks = [
            (g, triples[i][0], i)
            for g, triples in custom.items()
            for i in range(1, len(triples)) if triples[i][0]
        ]
        n_expected = len(tasks)

        if n_expected == 0:
            pass
        elif volcano_n_jobs == 1:
            # ----- serial path -----
            import matplotlib.pyplot as plt
            pbar = tqdm(total=n_expected, desc='volcanoes',
                        unit='cmp', mininterval=0.5)
            for g, compound, i in tasks:
                fig_v, ax_v = plt.subplots(
                    figsize=(volcano_size_px / 100, volcano_size_px / 100),
                    dpi=100)
                try:
                    plot_volcano(df_raw, compound, g,
                                 xmin=volcano_xlim[0], xmax=volcano_xlim[1],
                                 ax=ax_v, title='')
                    buf = io.BytesIO()
                    fig_v.savefig(buf, format='PNG', bbox_inches='tight')
                    b64 = base64.b64encode(buf.getvalue()).decode()
                except Exception as e:
                    tqdm.write(f'  [warn] volcano render failed for {g}/{compound}: {e}')
                    b64 = ''
                finally:
                    plt.close(fig_v)
                custom[g][i][3] = b64               # fill the reserved slot
                pbar.update(1)
            pbar.close()
        else:
            # ----- parallel path -----
            import contextlib
            import joblib as _joblib
            from joblib import Parallel, delayed
            unique_cmps = sorted({c for _, c, _ in tasks})
            print(f'> pre-slicing df_raw for {len(unique_cmps):,} compounds '
                  f'(one groupby pass, was 300x boolean filters)...', flush=True)
            # single O(n) pass instead of one boolean filter per compound — the
            # old loop was the dominant cost when df_raw has millions of rows.
            _cols = ['compound', 'genes', 'logfc', 'pvalue']
            _filt = df_raw.loc[df_raw['compound'].isin(unique_cmps), _cols].dropna()
            sub_cache = {c: g for c, g in _filt.groupby('compound', sort=False)}
            print(f'> rendering {n_expected:,} volcanoes on {volcano_n_jobs} workers...',
                  flush=True)

            @contextlib.contextmanager
            def _tqdm_joblib(pbar):
                class _Cb(_joblib.parallel.BatchCompletionCallBack):
                    def __call__(self, *a, **kw):
                        pbar.update(n=self.batch_size)
                        return super().__call__(*a, **kw)
                prev = _joblib.parallel.BatchCompletionCallBack
                _joblib.parallel.BatchCompletionCallBack = _Cb
                try:
                    yield pbar
                finally:
                    _joblib.parallel.BatchCompletionCallBack = prev
                    pbar.close()

            pbar = tqdm(total=n_expected, desc='volcanoes',
                        unit='cmp', mininterval=0.5)
            with _tqdm_joblib(pbar):
                results = Parallel(n_jobs=volcano_n_jobs, backend='loky')(
                    delayed(_volcano_render_worker)(
                        (g, c, sub_cache[c], volcano_size_px,
                         volcano_xlim[0], volcano_xlim[1])
                    )
                    for g, c, _ in tasks
                )
            for (g, c, i), b64 in zip(tasks, results):
                custom[g][i][3] = b64               # fill the reserved slot
        print(f'> rendered {n_expected:,} volcanoes')
    # else: index 3 is already '' for every compound row — nothing to do.

    # 4) build figure
    def _hover_text(df):
        areas = (df['disease_area'].fillna('—') if 'disease_area' in df.columns
                 else pd.Series(['—'] * len(df), index=df.index))
        # Fisher's-exact p from per-gene MCS enrichment (cell 49e1bc56). Falls
        # back to '—' if the MCS_CSV merge step hasn't run yet.
        def _fmt_p(v):
            if v is None or pd.isna(v):
                return '—'
            return '< 0.0001' if v < 0.0001 else f'{v:.4f}'
        if 'fisher_p' in df.columns:
            fp = df['fisher_p'].apply(_fmt_p)
        else:
            fp = pd.Series(['—'] * len(df), index=df.index)
        return [
            f'<b>{g}</b><br>R²={r:.3f}<br>overall_score={s:.3f}<br>'
            f'fold={f}<br>fisher_p={p}<br>n={n}<br>area={a}'
            for g, r, s, f, p, n, a in zip(
                df['gene'], df['R2'], df['overall_score'],
                df['fold'].apply(lambda x: '∞' if not np.isfinite(x) else f'{x:.1f}'),
                fp,
                df.get('n', [None] * len(df)),
                areas)
        ]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=plot_df['R2'], y=plot_df['overall_score'], z=plot_df['fold_plot'],
        mode='markers',
        marker=dict(size=3, color='lightgrey', opacity=0.5, line=dict(width=0)),
        name=f'all ({len(plot_df):,})',
        text=_hover_text(plot_df), hoverinfo='text',
    ))

    assert 'disease_area' in highlighted.columns, 'expected disease_area column on highlighted'
    NA_LABEL = '— no priority area —'
    hl = highlighted.copy()
    hl['_area'] = hl['disease_area'].fillna(NA_LABEL)

    area_order = [a for a in disease_area_colors if a in hl['_area'].values]
    if NA_LABEL in hl['_area'].values:
        area_order.append(NA_LABEL)

    for area in area_order:
        grp = hl[hl['_area'] == area]
        color = disease_area_colors.get(area, na_area_color)
        fig.add_trace(go.Scatter3d(
            x=grp['R2'], y=grp['overall_score'], z=grp['fold_plot'],
            mode='markers+text',
            marker=dict(size=6, color=color, opacity=0.95,
                        line=dict(color='#333', width=1)),
            text=grp['gene'],
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertext=_hover_text(grp), hoverinfo='text',
            customdata=[custom[g] for g in grp['gene']],
            name=f'{area} ({len(grp)})',
        ))

    fig.update_layout(
        height=height, width=width,
        title=title,
        scene=dict(
            xaxis=dict(title='SAR predictability (R²)', showbackground=False,
                       gridcolor='lightgrey', zeroline=False),
            yaxis=dict(title='OpenTargets overall_score', showbackground=False,
                       gridcolor='lightgrey', zeroline=False),
            zaxis=dict(title='MCS fold-enrichment (log scale)', type='log',
                       showbackground=False, gridcolor='lightgrey', zeroline=False),
            bgcolor='white',
        ),
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # 5) optional standalone HTML with on-hover structure thumbnails
    if html_path:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        # 'directory' writes plotly.py's BUNDLED (offline) plotly.min.js next to the
        # HTML and references it relatively — no CDN fetch on every open (measured the
        # dominant real-world load cost) and fully offline. Keep plotly.min.js
        # alongside the HTML, like the _data.js / volcanoes_px / srb_png sidecars.
        fig.write_html(html_path, include_plotlyjs='directory')

        # Pre-build a per-gene patents-HTML lookup. Injected as a global JS
        # dict so the panel JS can render the table on hover/click without
        # bloating customdata.
        gene_patents_map = _build_gene_patents_html_map(
            gene_patents_df, gene_patents_top_n, depmap_url_template,
        )
        import json as _json
        inject_data = (
            '<script>window.__GENE_PATENTS__ = '
            + _json.dumps(gene_patents_map) + ';\n'
            'window.__DEPMAP_URL__ = '
            + _json.dumps(depmap_url_template) + ';</script>'
        )

        with open(html_path) as fh:
            html = fh.read()
        with open(html_path, 'w') as fh:
            fh.write(html.replace('</body>', inject_data + _HOVER_INJECT + '</body>'))
        print(f'wrote {html_path}  ({os.path.getsize(html_path) / 1e6:.1f} MB)')

    if nb_display:
        fig.show()

    return fig, highlighted


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Interactive 3D target browser (generalised axes)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_3d_interface(
    target_df,
    *,
    x_col='R2', y_col='association_score', z_col='ms_score',
    x_label='SAR predictability (R²)',
    y_label='OpenTargets association_score',
    z_label='MS score',
    axis_help=None,
    z_log=False,
    z_clip_upper=None,
    gene_col='gene',
    must_include=(),
    exclude_genes=(),
    top_n_highlight=50,
    min_x_highlight=None,
    min_y_auto=None,
    top_n_hover=5,
    compounds_df=None,
    plate_dates=None,
    plate_defaults=None,
    panels=None,
    return_panels=False,
    volcano_source=None,
    volcano_key='uniquecontrast',
    page_size=5,
    png_dir='data/srb_png',
    thumb_external=False,
    df_raw=None,
    volcano_size_px=350,
    volcano_xlim=(-5.0, 5.0),
    volcano_n_jobs=1,
    compound_meta_df=None,
    compound_meta_icons=None,
    gene_patents_df=None,
    gene_patents_top_n=5,
    depmap_url_template='https://depmap.org/portal/gene/{gene}',
    disease_area_colors=None,
    na_area_color='#bbbbbb',
    title='',
    range_sliders=False,
    range_defaults=None,
    activity_defaults=None,
    control_genes=(),
    control_compounds=(),
    control_default_on=True,   # control-compound tickbox starts checked (controls shown)
    contaminant_compounds=(),
    contaminant_default_on=True,   # contaminant tickbox starts checked (contaminants shown)
    gene_research=None,
    validated_targets=(),
    devalidated_targets=(),
    validated_label='Yes',
    devalidated_label='No',
    validated_compounds=(),
    devalidated_compounds=(),
    compound_validated_label='Yes',
    compound_devalidated_label='No',
    compound_validation_defaults=None,
    depmap_defaults=None,
    conf_defaults=None,
    lof_defaults=None,
    validation_defaults=None,
    volcano_significant=False,
    volcano_dir=None,
    html_path=None,
    height=900,
    width=1500,
    nb_display=True,
):
    """
    Interactive 3D target browser — generalised sibling of ``plot_target_3d``.

    Same interactive shell (hover a target → compound table top-right; click to
    pin; hover a pinned compound → its volcano; per-gene patents panel) but the
    three axes are now *configurable* via ``x_col`` / ``y_col`` / ``z_col`` so it
    can plot any (R², association_score, ms_score)-style triple rather than the
    hardwired (R², overall_score, fold). This is the function we extend with
    new interface features; ``plot_target_3d`` is left untouched.

    Highlights (coloured by ``disease_area``; everything else is a light-grey
    backdrop):
      * the top ``top_n_highlight`` genes closest to the (x↑, y↑, z↑) corner,
      * all genes with ``y_col`` > ``min_y_auto`` (if set),
      * everything in ``must_include`` (bypasses the filters).
    If ``min_x_highlight`` is set, only genes at/above it are eligible for the
    corner-distance and auto-y highlights (a noise-floor gate on the x-axis).

    Graceful degradation: ``disease_area``, the ``top1..topN`` compound columns,
    ``fisher_p`` and ``gene_patents_df`` are all optional. Missing compound
    columns simply disable the hover/pin panel and volcanoes; the 3D scatter
    with hover text always renders.

    The compound panel accepts two input modes:
      * **long-format** ``compounds_df`` (recommended) — one row per
        (gene, compound) with columns ``gene``, ``compound`` (display id),
        ``logfc``, ``smiles``; optionally a ``volcano_key`` column (default name
        ``'uniquecontrast'``; falls back to ``compound`` if absent) that picks
        which volcano to show, and ``sublabel`` (small caption line, e.g.
        plate/conc). ALL of a gene's compounds are kept, sorted by ascending
        ``logfc``, and the panel paginates ``page_size`` at a time (◀ / ▶).
        Volcanoes are sliced from ``volcano_source`` by ``volcano_key``.
      * **wide** ``top1..topN_{compound,logfc,smiles}`` columns on ``target_df``
        (legacy ``plot_target_3d`` shape) — fixed top-N panel, volcanoes from
        ``df_raw``.

    :param df target_df: one row per gene. Must contain ``gene_col`` plus the
        three axis columns; optionally ``disease_area`` (colour) and ``fisher_p``.
    :param str x_col, y_col, z_col: column names mapped to the X/Y/Z axes.
    :param str x_label, y_label, z_label: axis titles (also shown in the legend).
    :param bool z_log: render Z on a log axis (and rank in log space).
    :param float z_clip_upper: clip Z to this value for plotting (outliers stay
        but are pulled to the cap); ``None`` = no clipping.
    :param str gene_col: name of the gene-symbol column in ``target_df``.
    :param df compounds_df: long-format compound table (see above). When given,
        it supersedes the wide ``top*`` columns and enables pagination.
    :param df volcano_source: rows the volcanoes are drawn from (needs the
        ``volcano_key`` column plus ``genes``/``logfc``/``pvalue``). For the FBX
        interface this is ``FBX_MEASURE`` keyed by ``uniquecontrast``.
    :param str volcano_key: column in ``volcano_source`` / ``compounds_df`` that
        identifies one volcano (default ``'uniquecontrast'``).
    :param int page_size: compounds shown per page in the panel (default 5).
    :return: ``(fig, highlighted)`` — the Plotly figure and highlighted-set DataFrame.
    """
    import io, base64
    import plotly.graph_objects as go
    from rdkit import Chem
    from rdkit.Chem import Draw

    if disease_area_colors is None:
        disease_area_colors = {}

    # Per-axis explanations shown on hover over the axis legend (like plot_target_3d).
    # Defaults describe the FBX interface axes; override any via `axis_help`.
    _axis_help = {
        'x': ('SAR predictability (R²): 5-fold cross-validated R² between '
              'chemistry-predicted and observed per-compound logfc for this gene. '
              'Higher = the compound structure explains more of its effect on the '
              'target, i.e. the SAR is more learnable/predictable.'),
        'y': ('OpenTargets association_score: target–disease association score '
              '(max across the priority disease areas). Higher = more '
              'genetic/clinical/literature evidence linking the gene to disease.'),
        'z': ('MS score: the FBX mass-spec proteomics score for the target — its '
              'strongest down-modulation signal across compounds. Higher = a '
              'stronger / more reproducible significant down-regulation.'),
    }
    if axis_help:
        _axis_help.update(axis_help)

    # 0) normalise the gene column to 'gene' so the rest mirrors plot_target_3d
    df = target_df.copy()
    if gene_col != 'gene':
        assert gene_col in df.columns, f'gene_col {gene_col!r} not in target_df'
        df = df.rename(columns={gene_col: 'gene'})

    # 1) filter target_df → plot_df, with must_include bypassing the filters
    required_cols = [x_col, y_col, z_col, 'gene']
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f'target_df is missing {missing}'

    plot_df = df.dropna(subset=[x_col, y_col, z_col]).copy()
    n0 = len(plot_df)
    must_set = set(must_include)
    is_must = plot_df['gene'].isin(must_set)
    dropped_named = plot_df[plot_df['gene'].isin(exclude_genes) & ~is_must]
    keep = is_must | ~plot_df['gene'].isin(exclude_genes)
    if z_clip_upper is not None:
        keep = keep & (is_must | (plot_df[z_col] <= z_clip_upper))
    dropped_z = plot_df[~keep & ~is_must & ~plot_df['gene'].isin(exclude_genes)]
    plot_df = plot_df[keep]

    # Z used for plotting (optionally clipped so an outlier doesn't squash the axis)
    plot_df['_zplot'] = (plot_df[z_col].clip(upper=z_clip_upper)
                         if z_clip_upper is not None else plot_df[z_col])

    print(f'> {len(plot_df):,} / {n0:,} genes after filtering '
          f'(x={x_col}, y={y_col}, z={z_col})')
    if len(dropped_named):
        print(f'  [excluded by name]  {list(dropped_named["gene"])}')
    if z_clip_upper is not None and len(dropped_z):
        print(f'  [excluded {z_col}>{z_clip_upper}]  {len(dropped_z)} genes')

    # 2) corner-distance ranking on normalised (x, y, z) — z optionally in log space
    plot_df['_zrank'] = (np.log10(plot_df[z_col].clip(lower=1e-9))
                         if z_log else plot_df[z_col])

    def _norm01(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng else s * 0.0
    xn = _norm01(plot_df[x_col])
    yn = _norm01(plot_df[y_col])
    zn = _norm01(plot_df['_zrank'])
    plot_df['_dist'] = np.sqrt((1 - xn) ** 2 + (1 - yn) ** 2 + (1 - zn) ** 2)

    candidates = (plot_df if min_x_highlight is None
                  else plot_df[plot_df[x_col] >= min_x_highlight])
    top_n   = candidates.nsmallest(top_n_highlight, '_dist')
    auto_y  = (candidates[candidates[y_col] > min_y_auto]
               if min_y_auto is not None else candidates.iloc[0:0])
    must    = plot_df[plot_df['gene'].isin(must_set)]
    miss = [g for g in must_include if g not in plot_df['gene'].values]
    if miss:
        print(f'  [warn] must_include not found: {miss}')
    highlighted = pd.concat([top_n, auto_y, must]).drop_duplicates('gene')
    print(f'  [highlight] corner-top-{top_n_highlight}={len(top_n)}, '
          f'{y_col}>{min_y_auto}: {len(auto_y)}, must={len(must)}, '
          f'union={len(highlighted)}'
          + (f' (x floor = {min_x_highlight})' if min_x_highlight is not None else ''))

    # Keep the corner-distance subset — it seeds the slider default ranges so the
    # initial view matches the static highlight you'd otherwise get.
    corner_df = highlighted

    # Slider mode: highlighting is driven client-side by the 3 range sliders, so
    # EVERY plotted gene needs a compound panel (any of them can become in-range).
    # The colour/grey split is recomputed in the browser; here we just make sure
    # panels exist for all of them.
    if range_sliders:
        highlighted = plot_df.copy()
        print(f'  [range_sliders] panels built for all {len(highlighted)} plotted genes')

    # 3) per-gene compound panel -> customdata. Two input modes:
    #      * long-format `compounds_df` (variable length, paginated; volcanoes
    #        from `volcano_source` keyed by `volcano_key`)
    #      * wide `top1..topN_*` columns (fixed top-N; volcanoes from `df_raw`)
    use_long = compounds_df is not None
    have_compounds = use_long or all(f'top{k}_compound' in highlighted.columns
                                     for k in range(1, top_n_hover + 1))
    custom = {}
    tasks = []     # (gene, volcano_key_value, entry_index, plate_index|None) for volcano pass
    _vsrc = None   # frame the volcano pass slices by its 'compound' column
    all_plates = []      # ordered unique plate labels for the client-side filter checkboxes
    all_activities = []  # ordered activity levels (nr_down buckets) for the activity filter
    _thumb_ext = False   # thumbnail path-mode flag (set in the have_compounds block)
    _thumb_rel = 'srb_png'
    # External volcanoes share one long directory prefix; factor it into a single
    # __VOLCANO_BASE__ and store only the per-row filename in the compound blob
    # (prefix × 23k rows was ~2 MB). '' ⇒ embedded mode, JS prepends nothing.
    _volcano_base = ''
    # Precomputed panels (compound blobs + plate/activity lists + thumb/volcano modes) can be
    # passed in to SKIP the expensive build below — the referenced thumbnail/volcano files must
    # already exist on disk (written by the run that produced the panels). Cache via return_panels.
    if have_compounds and panels is not None:
        custom         = panels['custom']
        all_plates     = panels['all_plates']
        all_activities = panels['all_activities']
        _volcano_base  = panels['volcano_base']
        _thumb_ext     = bool(panels['thumb_ext'])
        _thumb_rel     = panels['thumb_rel']
        print(f'> panels: loaded {len(custom):,} gene panels from cache (skipped rebuild)')

    if have_compounds and panels is None:
        _stats = {'png': 0, 'rdkit': 0, 'miss': 0}

        # thumbnail mode: 'path' references PNGs copied next to the HTML (lazy <img>,
        # keeps the page tiny); 'b64' inlines them (legacy, FBX). External needs png_dir
        # + html_path. Used compound PNGs are copied to <html_dir>/srb_png/ on demand.
        import shutil
        _thumb_ext = bool(thumb_external) and bool(png_dir) and bool(html_path)
        _thumb_rel = 'srb_png'
        _thumb_copied = set()
        if _thumb_ext:
            _thumb_out = os.path.join(os.path.dirname(os.path.abspath(html_path)), _thumb_rel)
            os.makedirs(_thumb_out, exist_ok=True)

        def _compound_b64(compound, smi, size=(170, 110)):
            # PATH mode -> copy the compound PNG next to the HTML, return the compound id
            # (the client builds <img src="srb_png/<id>.png">); '' if no image available.
            if _thumb_ext:
                if isinstance(compound, str) and compound and png_dir:
                    p = os.path.join(png_dir, f'{compound}.png')
                    if os.path.isfile(p):
                        if compound not in _thumb_copied:
                            dst = os.path.join(_thumb_out, f'{compound}.png')
                            # refresh if missing or the source is newer (real PNG superseding a cached rdkit render)
                            if not os.path.exists(dst) or os.path.getmtime(p) > os.path.getmtime(dst):
                                shutil.copyfile(p, dst)
                            _thumb_copied.add(compound)
                        _stats['png'] += 1
                        return compound
                    if isinstance(smi, str) and smi:        # render+cache to the thumb dir
                        m = Chem.MolFromSmiles(smi)
                        if m is not None:
                            dst = os.path.join(_thumb_out, f'{compound}.png')
                            if not os.path.exists(dst):
                                Draw.MolToImage(m, size=size).save(dst, format='PNG')
                            _stats['rdkit'] += 1
                            return compound
                _stats['miss'] += 1
                return ''
            # B64 mode (legacy): inline the PNG / rdkit render as base64
            if isinstance(compound, str) and compound and png_dir:
                p = os.path.join(png_dir, f'{compound}.png')
                if os.path.isfile(p):
                    with open(p, 'rb') as fh:
                        _stats['png'] += 1
                        return base64.b64encode(fh.read()).decode()
            if isinstance(smi, str) and smi:
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    img = Draw.MolToImage(m, size=size)
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    _stats['rdkit'] += 1
                    return base64.b64encode(buf.getvalue()).decode()
            _stats['miss'] += 1
            return ''

        # Pre-index compound metadata for O(1) lookup during the per-gene loop.
        meta_index = None
        if compound_meta_df is not None and compound_meta_icons:
            cm0 = len(compound_meta_df)
            cm = compound_meta_df.drop_duplicates('compound', keep='first')
            if len(cm) < cm0:
                print(f'  [warn] compound_meta_df: deduped {cm0 - len(cm):,} duplicate rows')
            meta_index = cm.set_index('compound').to_dict('index')

        def _meta_html(compound_id):
            if not meta_index or not compound_id:
                return ''
            row = meta_index.get(compound_id, {}) or {}
            parts = []
            for col, cfg in compound_meta_icons.items():
                v = row.get(col)
                tooltip = cfg.get('tooltip', col)
                state_map = cfg.get('state_map')
                if state_map is not None:
                    key = (str(v).strip().lower() if pd.notna(v) else None)
                    state = (state_map.get(key) or state_map.get(v)
                             or state_map.get('__default__', {'icon': '❓', 'color': '#bbb'}))
                    parts.append(
                        f'<span title="{tooltip}: {v}" '
                        f'style="color:{state.get("color", "#bbb")};font-weight:600;'
                        f'margin-left:3px;">{state.get("icon", "❓")}</span>')
                    continue
                show_if = cfg.get('show_if', lambda x: pd.notna(x) and bool(x))
                try:
                    ok = show_if(v)
                except Exception:
                    ok = False
                if not ok:
                    continue
                icon  = cfg.get('icon', '•')
                color = cfg.get('color', '#666')
                label = (cfg.get('label', lambda _v: '')(v)
                         if callable(cfg.get('label')) else cfg.get('label', ''))
                parts.append(
                    f'<span title="{tooltip}: {v}" '
                    f'style="color:{color};font-weight:600;margin-left:3px;">'
                    f'{icon}{label}</span>')
            return ''.join(parts)

        def _fmt_fp(v):
            if v is None or pd.isna(v):
                return '—'
            return '< 0.0001' if v < 0.0001 else f'{v:.4f}'

        # Gene-level header meta. Only emit `fisher_p=…` when the column exists
        # (it doesn't for the FBX interface), so the panel header stays clean.
        _has_fp = 'fisher_p' in highlighted.columns
        fp_by_gene = (highlighted.set_index('gene')['fisher_p'].to_dict()
                      if _has_fp else {})

        def _meta_str(gene):
            return f'fisher_p={_fmt_fp(fp_by_gene.get(gene))}' if _has_fp else ''

        if use_long:
            need = {'gene', 'compound', 'logfc', 'smiles'}
            miss_c = need - set(compounds_df.columns)
            assert not miss_c, f'compounds_df missing {miss_c}'
            vkey_col = volcano_key if volcano_key in compounds_df.columns else 'compound'
            has_sub = 'sublabel' in compounds_df.columns
            # plate-aware mode: a `plate` column means one row per (gene, compound,
            # plate). We collapse to one entry per (gene, compound) carrying ALL its
            # plates; the panel JS filters/stacks them per the plate checkboxes.
            plate_aware = 'plate' in compounds_df.columns
            if volcano_source is not None:
                # drop any pre-existing 'compound' so renaming volcano_key -> 'compound'
                # can't create a duplicate column (volcano_source may already carry compound)
                _vsrc = (volcano_source.drop(columns=['compound'], errors='ignore')
                         .rename(columns={volcano_key: 'compound'})
                         if volcano_key != 'compound' else volcano_source)
            cdf = compounds_df[compounds_df['gene'].isin(set(highlighted['gene']))].copy()
            if plate_aware:
                all_plates = sorted(compounds_df['plate'].dropna().astype(str).unique())
                # Optional per-experiment activity level (nr_down bucket). Ordered
                # high→silent for the checkbox panel; only levels present are kept.
                has_act = 'activity' in compounds_df.columns
                has_ng = 'n_genes' in compounds_df.columns   # genes measured in the experiment
                has_mbid = 'molecule_batch_id' in compounds_df.columns  # per-plate batch id
                if has_act:
                    _ACT_ORDER = ['High (>25)', 'Medium (11-25)', 'Low (2-10)',
                                  'Single (1)', 'Silent']
                    _present = set(compounds_df['activity'].dropna().astype(str))
                    all_activities = ([a for a in _ACT_ORDER if a in _present]
                                      + sorted(_present - set(_ACT_ORDER)))
                # compounds ordered by their strongest (min) logfc; plates within a
                # compound ordered by logfc too.
                cdf['_best'] = cdf.groupby(['gene', 'compound'])['logfc'].transform('min')
                cdf = cdf.sort_values(['gene', '_best', 'compound', 'logfc'],
                                      ascending=True)
                for gene, gdf in tqdm(cdf.groupby('gene', sort=False), total=cdf['gene'].nunique(),
                                      desc='compound panels', unit='gene', mininterval=0.5):
                    entries = [['__META__', '', _meta_str(gene), '', '', '']]
                    for compound, cg in gdf.groupby('compound', sort=False):
                        lab = str(compound)
                        smi = (cg['smiles'].dropna().iloc[0]
                               if cg['smiles'].notna().any() else None)
                        ei = len(entries)
                        plate_rows = []   # [plate, logfc, volcano, activity, n_genes, mbid] per plate
                        for pi, (_, pr) in enumerate(cg.iterrows()):
                            plate_rows.append([
                                str(pr['plate']),
                                f"{pr['logfc']:.2f}" if pd.notna(pr['logfc']) else '',
                                '',   # volcano b64/path filled in the render pass below
                                (str(pr['activity']) if has_act and pd.notna(pr['activity'])
                                 else ''),
                                (str(int(pr['n_genes'])) if has_ng and pd.notna(pr['n_genes'])
                                 else ''),
                                (str(pr['molecule_batch_id']) if has_mbid
                                 and pd.notna(pr['molecule_batch_id']) else ''),
                            ])
                            vk = pr[vkey_col]
                            if pd.notna(vk):
                                tasks.append((gene, vk, ei, pi))
                        entries.append([
                            lab,
                            _compound_b64(lab or None, smi),
                            f"{cg['logfc'].min():.2f}",
                            plate_rows,
                            _meta_html(lab),
                            '',
                        ])
                    custom[gene] = entries
            else:
                # one entry per (gene, compound), single volcano, no plate filter
                cdf = cdf.sort_values(['gene', 'logfc'], ascending=[True, True])
                for gene, grp in tqdm(cdf.groupby('gene', sort=False), total=cdf['gene'].nunique(),
                                      desc='compound panels', unit='gene', mininterval=0.5):
                    entries = [['__META__', '', _meta_str(gene), '', '', '']]
                    for _, r in grp.iterrows():
                        lab = str(r['compound']) if pd.notna(r['compound']) else ''
                        sub = (str(r['sublabel']) if has_sub and pd.notna(r.get('sublabel'))
                               else '')
                        entries.append([
                            lab,
                            _compound_b64(lab or None,
                                          r['smiles'] if pd.notna(r['smiles']) else None),
                            f"{r['logfc']:.2f}" if pd.notna(r['logfc']) else '',
                            '',
                            _meta_html(lab),
                            sub,
                        ])
                        vk = r[vkey_col]
                        if pd.notna(vk):
                            tasks.append((gene, vk, len(entries) - 1, None))
                    custom[gene] = entries
            # Highlighted genes with no associated compound still need a (meta-only)
            # customdata slot so the figure's `customdata=[custom[g] ...]` never KeyErrors.
            for g in highlighted['gene']:
                if g not in custom:
                    custom[g] = [['__META__', '', _meta_str(g), '', '', '']]
            print(f'> long-format panel: '
                  f'{sum(len(v) - 1 for v in custom.values()):,} compound entries '
                  f'across {sum(len(v) > 1 for v in custom.values())} genes '
                  f'with compounds (page_size={page_size}'
                  + (f', {len(all_plates)} plates' if plate_aware else '') + ')')
        else:
            _vsrc = df_raw
            for _, row in highlighted.iterrows():
                gene = row['gene']
                entries = [['__META__', '', _meta_str(gene), '', '', '']]
                for k in range(1, top_n_hover + 1):
                    c = row.get(f'top{k}_compound')
                    s = row.get(f'top{k}_smiles')
                    l = row.get(f'top{k}_logfc')
                    c_str = str(c) if pd.notna(c) else ''
                    entries.append([
                        c_str,
                        _compound_b64(c if pd.notna(c) else None,
                                      s if pd.notna(s) else None),
                        f'{l:.2f}' if pd.notna(l) else '',
                        '',
                        _meta_html(c_str),
                        '',
                    ])
                    if c_str:
                        tasks.append((gene, c_str, len(entries) - 1, None))
                custom[gene] = entries

        n_thumbs = _stats['png'] + _stats['rdkit']
        print(f'> built {n_thumbs:,} structure thumbnails across {len(custom)} genes '
              f'(png={_stats["png"]}, rdkit={_stats["rdkit"]}, missing={_stats["miss"]})')

        # Fill a rendered volcano into the right customdata slot. plate_idx is
        # None for single-volcano entries (slot 3 is the b64 string) or an int
        # for plate-aware entries (slot 3 is a list of [plate, logfc, b64]).
        def _set_volcano(g, ei, plate_idx, b64):
            if plate_idx is None:
                custom[g][ei][3] = b64
            else:
                custom[g][ei][3][plate_idx][2] = b64

        # 3b) per-(gene, compound[, plate]) volcanoes from `_vsrc`, keyed by the
        #     volcano-key value carried in `tasks`. If `volcano_dir` is set (and
        #     we're writing HTML), PNGs are cached to that folder and referenced by
        #     relative path (lazy-loaded, tiny HTML, cached re-runs skip rendering);
        #     otherwise they're embedded as base64 in the customdata.
        if _vsrc is not None and tasks:
            import hashlib
            # Significant-only volcanoes render as INTERACTIVE SVG (rasterised grey
            # cloud + vector significant points carrying <title> hover tooltips);
            # otherwise plain PNG. SVGs are shown via <object>, PNGs via <img>.
            _sig = bool(volcano_significant) and ('significant' in _vsrc.columns)
            _ext = '.svg' if _sig else '.png'
            _external = bool(volcano_dir) and bool(html_path)
            if _external:
                os.makedirs(volcano_dir, exist_ok=True)
                _rel = os.path.relpath(
                    volcano_dir, os.path.dirname(os.path.abspath(html_path))).replace(os.sep, '/')
                _volcano_base = _rel   # emitted once; rows store only the filename

                def _vfname(g, vk):
                    return _volcano_cache_fname(g, vk, volcano_xlim, volcano_size_px, _ext)

                # cache hits: file already on disk -> reference it, skip render. List the
                # dir ONCE and test membership in memory — an os.path.exists per task is
                # ~26k stat calls, painfully slow on a Dropbox/networked mount (/mnt/c).
                _existing = set(os.listdir(volcano_dir))
                render = []
                for (g, vk, ei, pi) in tqdm(tasks, desc='volcano cache scan',
                                            unit='task', mininterval=0.5):
                    fn_ = _vfname(g, vk)
                    if fn_ in _existing:
                        _set_volcano(g, ei, pi, fn_)   # base prepended client-side
                    else:
                        render.append((g, vk, ei, pi, fn_))
                n_cached = len(tasks) - len(render)
            else:
                render = [(g, vk, ei, pi, None) for (g, vk, ei, pi) in tasks]
                n_cached = 0

            def _store(g, ei, pi, fn_, content):
                # content = SVG text (_sig) or base64 PNG. External: write the file
                # and store its relative path; embedded: store an inline value
                # (data-URI SVG, or raw base64 PNG). '' on failure.
                if not content:
                    _set_volcano(g, ei, pi, '')
                    return
                if _external:
                    mode_ = 'w' if _sig else 'wb'
                    data_ = content if _sig else base64.b64decode(content)
                    with open(os.path.join(volcano_dir, fn_), mode_,
                              **({'encoding': 'utf-8'} if _sig else {})) as _fh:
                        _fh.write(data_)
                    _set_volcano(g, ei, pi, fn_)   # base prepended client-side
                elif _sig:
                    _set_volcano(g, ei, pi, 'data:image/svg+xml;base64,'
                                 + base64.b64encode(content.encode()).decode())
                else:
                    _set_volcano(g, ei, pi, content)

            n_render = len(render)
            if n_render == 0:
                pass
            elif volcano_n_jobs == 1:
                import matplotlib.pyplot as plt
                pbar = tqdm(total=n_render, desc='volcanoes', unit='cmp', mininterval=0.5)
                for g, vk, ei, pi, fn_ in render:
                    try:
                        if _sig:
                            content = _volcano_svg_string(
                                _vsrc, vk, g, key='compound', sig_col='significant',
                                xmin=volcano_xlim[0], xmax=volcano_xlim[1],
                                size_px=volcano_size_px)
                        else:
                            fig_v, ax_v = plt.subplots(
                                figsize=(volcano_size_px / 100, volcano_size_px / 100), dpi=100)
                            try:
                                plot_volcano(_vsrc, vk, g,
                                             xmin=volcano_xlim[0], xmax=volcano_xlim[1],
                                             ax=ax_v, title='')
                                buf = io.BytesIO()
                                fig_v.savefig(buf, format='PNG', bbox_inches='tight')
                                content = base64.b64encode(buf.getvalue()).decode()
                            finally:
                                plt.close(fig_v)
                    except Exception as e:
                        tqdm.write(f'  [warn] volcano failed {g}/{vk}: {e}')
                        content = ''
                    _store(g, ei, pi, fn_, content)
                    pbar.update(1)
                pbar.close()
            else:
                import contextlib
                import joblib as _joblib
                from joblib import Parallel, delayed
                unique_keys = sorted({vk for _, vk, _, _, _ in render})
                _cols = ['compound', 'genes', 'logfc', 'pvalue'] + (['significant'] if _sig else [])
                _filt = _vsrc.loc[_vsrc['compound'].isin(unique_keys), _cols].dropna()
                sub_cache = {c: g for c, g in _filt.groupby('compound', sort=False)}
                _empty = _filt.iloc[0:0]
                print(f'> rendering {n_render:,} volcanoes on {volcano_n_jobs} workers'
                      + (f' ({n_cached:,} cached)' if _external else '')
                      + (' [significant SVG]' if _sig else '') + '...', flush=True)

                @contextlib.contextmanager
                def _tqdm_joblib(pbar):
                    class _Cb(_joblib.parallel.BatchCompletionCallBack):
                        def __call__(self, *a, **kw):
                            pbar.update(n=self.batch_size)
                            return super().__call__(*a, **kw)
                    prev = _joblib.parallel.BatchCompletionCallBack
                    _joblib.parallel.BatchCompletionCallBack = _Cb
                    try:
                        yield pbar
                    finally:
                        _joblib.parallel.BatchCompletionCallBack = prev
                        pbar.close()

                pbar = tqdm(total=n_render, desc='volcanoes', unit='cmp', mininterval=0.5)
                with _tqdm_joblib(pbar):
                    results = Parallel(n_jobs=volcano_n_jobs, backend='loky')(
                        delayed(_volcano_render_worker)(
                            (g, vk, sub_cache.get(vk, _empty), volcano_size_px,
                             volcano_xlim[0], volcano_xlim[1], _sig))
                        for g, vk, _, _, _ in render)
                for (g, vk, ei, pi, fn_), content in zip(render, results):
                    _store(g, ei, pi, fn_, content)
            print(f'> volcanoes: {n_cached:,} cached, {n_render:,} rendered'
                  + (' [interactive SVG]' if _sig else '')
                  + (f' -> {volcano_dir}' if _external else ' (embedded)'))
        elif _vsrc is None:
            print('> no volcano source (pass df_raw or volcano_source) — volcanoes disabled')
    else:
        print('> no compound panel (provide compounds_df or top1..topN columns) — '
              'scatter + hover text only')

    # bundle the (built or loaded) panel data so callers can cache + replay it (return_panels)
    _panels_out = ({'custom': custom, 'all_plates': list(all_plates),
                    'all_activities': list(all_activities), 'volcano_base': _volcano_base,
                    'thumb_ext': bool(_thumb_ext), 'thumb_rel': _thumb_rel}
                   if have_compounds else None)

    # 4) build figure
    def _hover_text(d):
        areas = (d['disease_area'].fillna('—') if 'disease_area' in d.columns
                 else pd.Series(['—'] * len(d), index=d.index))
        return [
            f'<b>{g}</b><br>{x_label}={xx:.3f}<br>{y_label}={yy:.3f}<br>'
            f'{z_label}={zz:.3f}<br>area={a}'
            for g, xx, yy, zz, a in zip(
                d['gene'], d[x_col], d[y_col], d[z_col], areas)
        ]

    fig = go.Figure()
    # Invisible backdrop of ALL genes. opacity=0 hides it visually, but the trace stays
    # visible=True so its full-extent points still define the scene's autorange — that's
    # what keeps the coloured dots anchored in place as the sliders filter them in/out.
    # Hover + legend are off and the (now unused) full-genome hover text is dropped.
    fig.add_trace(go.Scatter3d(
        x=plot_df[x_col], y=plot_df[y_col], z=plot_df['_zplot'],
        mode='markers',
        marker=dict(size=3, color='lightgrey', opacity=0, line=dict(width=0)),
        name=f'all ({len(plot_df):,})',
        hoverinfo='skip', showlegend=False,
    ))

    NA_LABEL = '— no priority area —'
    hl = highlighted.copy()
    hl['_area'] = (hl['disease_area'].fillna(NA_LABEL)
                   if 'disease_area' in hl.columns else NA_LABEL)
    # Control targets (e.g. GAK) — genes whose only significant compound(s) are
    # controls — are drawn as grey diamonds in a dedicated "control" trace, pulled
    # out of their disease-area group. (Plotly 3D has no 'star' marker; 'diamond'
    # is the closest distinct symbol.)
    hl['_ctrl'] = hl['gene'].isin(set(control_genes))
    area_order = [a for a in disease_area_colors if a in hl.loc[~hl['_ctrl'], '_area'].values]
    if NA_LABEL in hl.loc[~hl['_ctrl'], '_area'].values:
        area_order.append(NA_LABEL)

    area_data = []          # plain coord arrays per colour trace, for the slider JS
    area_trace_indices = []  # trace indices that the sliders restyle (colour traces)

    def _add_colour_trace(grp, name, color, symbol='circle', size=6):
        # Rendering a gene-name text label for every point is the dominant cost in a
        # gl3d scatter (thousands of 3D text sprites can take tens of seconds to lay
        # out at the initial Plotly.newPlot). When the range sliders are present, start
        # markers-only (fast paint); applyRanges() re-enables 'markers+text' for the
        # in-range subset only when it's small enough to be readable (<= labelMax).
        # The label data/styling is still emitted so the JS can switch text on with no
        # re-layout of the data. Without sliders, keep the original always-on labels.
        _init_mode = 'markers' if range_sliders else 'markers+text'
        trace_kw = dict(
            x=grp[x_col], y=grp[y_col], z=grp['_zplot'],
            mode=_init_mode,
            marker=dict(size=size, color=color, symbol=symbol, opacity=0.95,
                        line=dict(color='#333', width=1)),
            text=grp['gene'], textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertext=_hover_text(grp), hoverinfo='text',
            name=name,
        )
        if have_compounds:
            # customdata = just the gene name; heavy entries live in __GENE_COMPOUNDS__.
            trace_kw['customdata'] = list(grp['gene'])
        fig.add_trace(go.Scatter3d(**trace_kw))
        area_trace_indices.append(len(fig.data) - 1)
        area_data.append({
            'x': [float(v) for v in grp[x_col]],
            'y': [float(v) for v in grp[y_col]],
            'z': [float(v) for v in grp['_zplot']],
            'gene': list(grp['gene']),
            'hover': list(_hover_text(grp)),
        })

    for area in area_order:
        grp = hl[(hl['_area'] == area) & (~hl['_ctrl'])]
        if grp.empty:
            continue
        _add_colour_trace(grp, f'{area} ({len(grp)})',
                          disease_area_colors.get(area, na_area_color))

    ctrl_grp = hl[hl['_ctrl']]
    if not ctrl_grp.empty:
        _add_colour_trace(ctrl_grp, f'control ({len(ctrl_grp)})',
                          '#9e9e9e', symbol='diamond', size=7)

    # Pinned-genes overlay — initially empty; the search box drives it (JS sets
    # x/y/z/text/customdata on pin). Gold diamond + always-on label; deliberately NOT
    # added to area_trace_indices, so the range sliders / filters never touch it —
    # a pinned gene stays visible regardless of every filter.
    # markers-only: the gene-name labels are drawn via scene.annotations (refreshLabels),
    # the SAME 11px SVG path as every other gene — gl3d trace-text renders oversized.
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode='markers',
        marker=dict(size=6, color='#1D3557', symbol='diamond',
                    opacity=1.0, line=dict(color='#1D3557', width=1.5)),
        text=[], customdata=[], hovertext=[], hoverinfo='text', name='★ pinned',
    ))
    pin_trace_index = len(fig.data) - 1

    # Range-slider config. The colour traces are indices 1..N (trace 0 = grey
    # backdrop); the JS slices them to the in-range subset on each slider move.
    ranges_cfg = None
    if range_sliders:
        # Sliders span the full plotted range. Default handles = a box that keeps
        # ~the corner-subset SIZE of genes "high on all three axes" (an axis-aligned
        # box can't reproduce the distance-based corner set — its bounding box is
        # far larger — so we binary-search a common lower percentile that yields
        # roughly `target` genes in the x≥·∧y≥·∧z≥· intersection).
        target = max(1, min(len(corner_df), len(plot_df)))
        xv = plot_df[x_col].to_numpy(dtype=float)
        yv = plot_df[y_col].to_numpy(dtype=float)
        zv = plot_df['_zplot'].to_numpy(dtype=float)

        def _count(p):
            return int(((xv >= np.quantile(xv, p)) & (yv >= np.quantile(yv, p))
                        & (zv >= np.quantile(zv, p))).sum())
        _plo, _phi = 0.0, 0.99
        for _ in range(40):
            _pm = (_plo + _phi) / 2
            if _count(_pm) > target:
                _plo = _pm
            else:
                _phi = _pm
        _p = _plo

        def _axis_cfg(v, label):
            lo, hi = float(v.min()), float(v.max())
            return {'min': lo, 'max': hi, 'step': (hi - lo) / 200 if hi > lo else 1.0,
                    'lo': float(np.quantile(v, _p)), 'hi': hi, 'label': label}
        ranges_cfg = {
            'x': _axis_cfg(xv, x_label),
            'y': _axis_cfg(yv, y_label),
            'z': _axis_cfg(zv, z_label),
            'areaTraces': area_trace_indices,
        }
        # Per-axis default lower-handle overrides, e.g. {'y': 0.35} to start the
        # association handle at 0.35 (clamped to the axis range).
        for _ax, _lo in (range_defaults or {}).items():
            if _ax in ranges_cfg:
                ranges_cfg[_ax]['lo'] = float(np.clip(_lo, ranges_cfg[_ax]['min'],
                                                      ranges_cfg[_ax]['max']))
        # labelMax: a readable ceiling on how many gene-name labels render at once
        # (drawing them all — ~4,600+ SVG annotations — was the ~30 s cost). Floor at 800
        # so widening the ranges past the initial focus box actually surfaces more labels
        # (previously it stayed frozen at the small focus-box count); the initial focus box
        # can label more if it's bigger; hard ceiling 1000. refreshLabels() spreads this
        # budget EVENLY across the MS-score range rather than clustering at the top.
        _lx, _ly, _lz = ranges_cfg['x']['lo'], ranges_cfg['y']['lo'], ranges_cfg['z']['lo']
        _focus_n = int(((xv >= _lx) & (yv >= _ly) & (zv >= _lz)).sum())
        ranges_cfg['labelMax'] = int(min(max(_focus_n + 20, 800), 1000))
        print(f'  [range_sliders] default box ≈ {_focus_n} genes; '
              f'gene labels shown while ≤ {ranges_cfg["labelMax"]} in range '
              f'(drag handles to widen/narrow)')

    fig.update_layout(
        height=height, width=width, title=title,
        scene=dict(
            xaxis=dict(title=x_label, showbackground=False,
                       gridcolor='lightgrey', zeroline=False),
            yaxis=dict(title=y_label, showbackground=False,
                       gridcolor='lightgrey', zeroline=False),
            zaxis=dict(title=z_label, type=('log' if z_log else 'linear'),
                       showbackground=False, gridcolor='lightgrey', zeroline=False),
            bgcolor='white',
        ),
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, b=0, t=(40 if title else 10)),
    )

    # 5) optional standalone HTML with on-hover structure thumbnails
    if html_path:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        # 'directory' writes plotly.py's BUNDLED (offline) plotly.min.js next to the
        # HTML and references it relatively — no CDN fetch on every open (measured the
        # dominant real-world load cost) and fully offline. Keep plotly.min.js
        # alongside the HTML, like the _data.js / volcanoes_px / srb_png sidecars.
        fig.write_html(html_path, include_plotlyjs='directory')
        gene_patents_map = _build_gene_patents_html_map(
            gene_patents_df, gene_patents_top_n, depmap_url_template)
        import json as _json
        # Emit large data via JSON.parse("...") not as JS object literals: the engine
        # scans a string literal cheaply and the native JSON parser is far faster than
        # parsing a multi-MB object literal as source (big page load-time win).
        def _jsp(obj):
            s = _json.dumps(obj).replace('</', '<\\/')   # neutralise premature </script>
            return 'JSON.parse(' + _json.dumps(s) + ')'

        # Resolve activity_defaults against the ACTUAL level labels (e.g. the data
        # uses 'Single (1)', 'Low (2-10)' — not bare 'Single'/'Low'). Match case-
        # insensitively by substring so ['Single','Low'] hits 'Single (1)' etc.
        # If nothing matches, fall back to all-ticked (None) rather than an empty
        # set — an empty JS array is truthy and would untick everything.
        _act_def = None
        if activity_defaults:
            _wanted = [str(a).strip().lower() for a in activity_defaults]
            _act_def = [a for a in all_activities
                        if any(w in a.lower() for w in _wanted)]
            if _act_def:
                print(f'  [activity_defaults] focus view starts on {_act_def}')
            else:
                print(f'  [activity_defaults] none of {list(activity_defaults)} '
                      f'matched present levels {list(all_activities)} — '
                      f'defaulting to all activities ticked')
                _act_def = None

        # Research-derived gene-level filters (Target filters section). Each buckets a
        # free-text field of the gene_research record into a canonical category for a
        # tickbox filter, adding a "(no data)" bucket only if some plotted (highlighted)
        # gene lacks a record. All three share the one gene_research pass below.
        _dep_order = ['Pan-essential', 'Selective', 'Non-essential', 'Other']
        _conf_order = ['High', 'Med', 'Low', 'Other']        # research confidence
        _lof_order = ['Yes', 'No', 'Maybe', 'Other']         # LoF therapeutic benefit

        def _prefix_cat(s, order):
            # canonical category = first entry in `order` whose lowercase prefix the
            # free text starts with; 'Other' if non-empty but unmatched, '' if empty.
            s = str(s or '').strip().lower()
            if not s:
                return ''
            for c in order:
                if c != 'Other' and s.startswith(c.lower()):
                    return c
            return 'Other'

        def _depmap_cat(s):
            s = str(s or '').strip().lower()
            if not s:
                return ''
            if s.startswith('pan-essential'):
                return 'Pan-essential'
            if s.startswith('non-essential'):
                return 'Non-essential'
            if s.startswith('selective'):
                return 'Selective'
            return 'Other'
        # Research is only ever shown for a PLOTTED gene (hover/click/pin), so drop any
        # record whose gene isn't a dot — a whole-genome research file (~10K genes) would
        # otherwise ship records for thousands of unreachable genes in the sidecar. This
        # keeps research available for every plotted gene while bounding the payload.
        if gene_research:
            _plotted = set(plot_df['gene'])
            _n_all = len(gene_research)
            gene_research = {g: r for g, r in gene_research.items() if g in _plotted}
            _mb = len(_json.dumps(gene_research)) / 1e6
            print(f'  [gene_research] {len(gene_research)}/{_n_all} records kept '
                  f'(plotted genes only) — {_mb:.1f} MB injected')
        gene_depmap, gene_conf, gene_lof = {}, {}, {}
        if gene_research:
            for _g, _rec in gene_research.items():
                if isinstance(_rec, dict):
                    _c = _depmap_cat(_rec.get('depmap_dependency'))
                    if _c:
                        gene_depmap[_g] = _c
                    _cc = _prefix_cat(_rec.get('confidence'), _conf_order)
                    if _cc:
                        gene_conf[_g] = _cc
                    _lc = _prefix_cat(_rec.get('lof_therapeutic_benefit'), _lof_order)
                    if _lc:
                        gene_lof[_g] = _lc

        def _cats_with_nodata(gene_map, order):
            cats = [c for c in order if c in set(gene_map.values())]
            if cats and any(g not in gene_map for g in highlighted['gene']):
                cats = cats + ['(no data)']
            return cats
        depmap_cats = _cats_with_nodata(gene_depmap, _dep_order)
        conf_cats = _cats_with_nodata(gene_conf, _conf_order)
        lof_cats = _cats_with_nodata(gene_lof, _lof_order)

        # Resolve which category boxes start ticked. None => all ticked; a list keeps
        # only the matching present categories (exact or prefix, case-insensitive);
        # an empty list => none ticked. Returned to JS as a list, or 'null' for all.
        def _resolve_defaults(defaults, cats):
            if defaults is None:
                return None
            wanted = [str(d).strip().lower() for d in defaults]
            return [c for c in cats
                    if any(c.lower() == w or c.lower().startswith(w) for w in wanted)]
        depmap_def = _resolve_defaults(depmap_defaults, depmap_cats)
        conf_def = _resolve_defaults(conf_defaults, conf_cats)
        lof_def = _resolve_defaults(lof_defaults, lof_cats)

        # Target validation (Y/N) filter: "Yes" -> validated_targets, "No" ->
        # devalidated_targets. Only emit a category if at least one plotted gene
        # carries it, so the dropdown appears only when it can do something.
        _hl_genes = set(highlighted['gene'])
        validated_set = [str(g) for g in (validated_targets or [])]
        devalidated_set = [str(g) for g in (devalidated_targets or [])]
        validation_cats = []
        if _hl_genes & set(validated_set):
            validation_cats.append(validated_label)
        if _hl_genes & set(devalidated_set):
            validation_cats.append(devalidated_label)
        validation_def = _resolve_defaults(validation_defaults, validation_cats)

        # Compound-validation (FBXO31 dependent/independent) filter: mirror of the target
        # one but keyed by compound id. Restrict the injected sets + categories to compounds
        # actually present in the panel, so the dropdown only shows when it can act.
        _present_cmps = (set(str(c) for c in compounds_df['compound'])
                         if (compounds_df is not None and len(compounds_df)
                             and 'compound' in compounds_df.columns) else set())
        cmp_validated_set = [c for c in (str(x) for x in (validated_compounds or [])) if c in _present_cmps]
        cmp_devalidated_set = [c for c in (str(x) for x in (devalidated_compounds or [])) if c in _present_cmps]
        cmp_validation_cats = []
        if cmp_validated_set:
            cmp_validation_cats.append(compound_validated_label)
        if cmp_devalidated_set:
            cmp_validation_cats.append(compound_devalidated_label)
        cmp_validation_def = _resolve_defaults(compound_validation_defaults, cmp_validation_cats)

        # The data blobs (compound panels, patents, research, area metadata, plate/
        # activity/range config) are NOT needed to first-paint the 3D plot — only
        # once a gene is hovered/clicked or a filter is touched. Emitting them inline
        # forces the browser to tokenise several MB (dominated by __GENE_COMPOUNDS__)
        # before the plot can appear. Instead write them to a DEFERRED sidecar .js:
        #   * the main document stays small (Plotly figure + handlers) → fast parse
        #     and first-paint of the dots;
        #   * the sidecar downloads in parallel and runs right before DOMContentLoaded,
        #     so every global is set by the time the handlers (which all wait on
        #     DOMContentLoaded) wire up — nothing is lost, it's the same globals.
        # A <script defer src> (not fetch) is used deliberately: the HTML is opened
        # via file:// double-click, where fetch() of a local file is CORS-blocked but
        # a same-folder external script loads fine.
        # plate -> date (string) for the nested-by-date Plates filter; only the plates
        # actually present in the panel, undated ones omitted (JS buckets them as "(no date)").
        _plate_dates_map = ({str(p): str(plate_dates[p]) for p in all_plates
                             if plate_dates.get(p) is not None}
                            if plate_dates else {})
        # plates to start ticked (None -> all). Restrict to plates actually present.
        _plate_def = (None if plate_defaults is None
                      else [str(p) for p in plate_defaults if str(p) in set(map(str, all_plates))])
        # gene -> plotted [x, y, z] (z is _zplot, the rendered z) + sorted name list,
        # for the search box / pin overlay. Built from plot_df = ALL genes, so any gene
        # is pinnable (incl. zero-R² genes greyed out of the default range).
        _gene_xyz = {str(g): [float(x), float(y), float(z)] for g, x, y, z in
                     zip(plot_df['gene'], plot_df[x_col], plot_df[y_col], plot_df['_zplot'])}
        _all_genes = sorted(_gene_xyz)
        # gene -> marker colour for the pin overlay, matching the area-trace colouring:
        # control genes grey, else disease_area colour (na_area_color when missing).
        _ctrl_set = set(control_genes)
        _areas_col = (plot_df['disease_area'] if 'disease_area' in plot_df.columns
                      else pd.Series([None] * len(plot_df), index=plot_df.index))
        _gene_color = {str(g): ('#9e9e9e' if g in _ctrl_set else
                                (disease_area_colors.get(a, na_area_color) if pd.notna(a) else na_area_color))
                       for g, a in zip(plot_df['gene'], _areas_col)}
        # compound -> target genes (pinning a compound pins its target genes) + sorted
        # compound list for the search autocomplete. Built from compounds_df.
        _compound_genes = {}
        if (compounds_df is not None and len(compounds_df)
                and {'compound', 'gene'} <= set(compounds_df.columns)):
            for _c, _sub in compounds_df.groupby('compound'):
                _compound_genes[str(_c)] = sorted({str(g) for g in _sub['gene']})
        _all_compounds = sorted(_compound_genes)
        data_js = (
            'window.__GENE_COMPOUNDS__ = ' + _jsp(custom if have_compounds else {}) + ';\n'
            'window.__GENE_PATENTS__ = ' + _jsp(gene_patents_map) + ';\n'
            'window.__DEPMAP_URL__ = ' + _json.dumps(depmap_url_template) + ';\n'
            'window.__PAGE_SIZE__ = ' + str(int(page_size)) + ';\n'
            'window.__PLATES__ = ' + _jsp(list(all_plates)) + ';\n'
            'window.__PLATE_DATES__ = ' + _jsp(_plate_dates_map) + ';\n'
            'window.__PLATE_DEFAULTS__ = ' + (_jsp(_plate_def) if _plate_def is not None else 'null') + ';\n'
            'window.__ACTIVITIES__ = ' + _jsp(list(all_activities)) + ';\n'
            'window.__ACTIVITY_DEFAULTS__ = ' + (_jsp(_act_def) if _act_def else 'null') + ';\n'
            'window.__CONTROL_COMPOUNDS__ = ' + _jsp([str(c) for c in (control_compounds or [])]) + ';\n'
            'window.__CONTROL_DEFAULT_ON__ = ' + _json.dumps(bool(control_default_on)) + ';\n'
            'window.__CONTAMINANT_COMPOUNDS__ = ' + _jsp([str(c) for c in (contaminant_compounds or [])]) + ';\n'
            'window.__CONTAMINANT_DEFAULT_ON__ = ' + _json.dumps(bool(contaminant_default_on)) + ';\n'
            'window.__RANGES__ = ' + _jsp(ranges_cfg) + ';\n'
            'window.__AREA_DATA__ = ' + _jsp(area_data) + ';\n'
            'window.__PIN_TRACE__ = ' + str(int(pin_trace_index)) + ';\n'
            'window.__ALL_GENES__ = ' + _jsp(_all_genes) + ';\n'
            'window.__GENE_XYZ__ = ' + _jsp(_gene_xyz) + ';\n'
            'window.__GENE_COLOR__ = ' + _jsp(_gene_color) + ';\n'
            'window.__ALL_COMPOUNDS__ = ' + _jsp(_all_compounds) + ';\n'
            'window.__COMPOUND_GENES__ = ' + _jsp(_compound_genes) + ';\n'
            'window.__VOLCANO_MODE__ = '
            + _json.dumps('svg' if (volcano_significant and 'significant'
                                    in (volcano_source.columns if volcano_source is not None else []))
                          else ('path' if (volcano_dir and html_path) else 'b64')) + ';\n'
            'window.__VOLCANO_BASE__ = ' + _json.dumps(_volcano_base) + ';\n'
            'window.__THUMB_MODE__ = ' + _json.dumps('path' if _thumb_ext else 'b64') + ';\n'
            'window.__THUMB_DIR__ = ' + _json.dumps(_thumb_rel) + ';\n'
            'window.__AXIS_LABELS__ = '
            + _jsp({'x': x_label, 'y': y_label, 'z': z_label}) + ';\n'
            'window.__AXIS_HELP__ = ' + _jsp(_axis_help) + ';\n'
            'window.__GENE_RESEARCH__ = ' + _jsp(gene_research or {}) + ';\n'
            'window.__GENE_DEPMAP__ = ' + _jsp(gene_depmap) + ';\n'
            'window.__DEPMAP_CATS__ = ' + _jsp(depmap_cats) + ';\n'
            'window.__DEPMAP_DEFAULTS__ = ' + (_jsp(depmap_def) if depmap_def is not None else 'null') + ';\n'
            'window.__GENE_CONF__ = ' + _jsp(gene_conf) + ';\n'
            'window.__CONF_CATS__ = ' + _jsp(conf_cats) + ';\n'
            'window.__CONF_DEFAULTS__ = ' + (_jsp(conf_def) if conf_def is not None else 'null') + ';\n'
            'window.__GENE_LOF__ = ' + _jsp(gene_lof) + ';\n'
            'window.__LOF_CATS__ = ' + _jsp(lof_cats) + ';\n'
            'window.__LOF_DEFAULTS__ = ' + (_jsp(lof_def) if lof_def is not None else 'null') + ';\n'
            'window.__VALIDATED_TARGETS__ = ' + _jsp(validated_set) + ';\n'
            'window.__DEVALIDATED_TARGETS__ = ' + _jsp(devalidated_set) + ';\n'
            'window.__VALIDATION_CATS__ = ' + _jsp(validation_cats) + ';\n'
            'window.__VALIDATION_DEFAULTS__ = ' + (_jsp(validation_def) if validation_def is not None else 'null') + ';\n'
            'window.__VAL_LABEL_YES__ = ' + _json.dumps(str(validated_label)) + ';\n'
            'window.__VAL_LABEL_NO__ = ' + _json.dumps(str(devalidated_label)) + ';\n'
            'window.__VALIDATED_COMPOUNDS__ = ' + _jsp(cmp_validated_set) + ';\n'
            'window.__DEVALIDATED_COMPOUNDS__ = ' + _jsp(cmp_devalidated_set) + ';\n'
            'window.__CMP_VALIDATION_CATS__ = ' + _jsp(cmp_validation_cats) + ';\n'
            'window.__CMP_VALIDATION_DEFAULTS__ = ' + (_jsp(cmp_validation_def) if cmp_validation_def is not None else 'null') + ';\n'
            'window.__CMP_VAL_LABEL_YES__ = ' + _json.dumps(str(compound_validated_label)) + ';\n'
            'window.__CMP_VAL_LABEL_NO__ = ' + _json.dumps(str(compound_devalidated_label)) + ';\n')
        _data_name = os.path.splitext(os.path.basename(html_path))[0] + '_data.js'
        _data_path = os.path.join(os.path.dirname(html_path), _data_name)
        with open(_data_path, 'w') as fh:
            fh.write(data_js)
        inject_data = '<script defer src="' + _data_name + '"></script>'
        with open(html_path) as fh:
            html = fh.read()
        with open(html_path, 'w') as fh:
            fh.write(html.replace('</body>', inject_data + _INTERFACE_INJECT + '</body>'))
        print(f'wrote {html_path}  ({os.path.getsize(html_path) / 1e6:.1f} MB main doc)'
              f'  +  {_data_name}  ({os.path.getsize(_data_path) / 1e6:.1f} MB, deferred)')

    if nb_display:
        fig.show()

    if return_panels:
        return fig, highlighted, _panels_out
    return fig, highlighted


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Per-compound volcano plot (one gene highlighted)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_volcano(df, compound, gene,
                 *,
                 fc_thresh=1.0, p_thresh=0.05,
                 xmin=-5.0, xmax=5.0,
                 figsize=(6, 6), dpi=100,
                 up_color='#008bfb', down_color='#ff0051',
                 ax=None, title=None):
    """
    Volcano plot for a single compound, with one target gene highlighted.

    For the given ``compound``, collapse multi-batch/plate replicates per gene
    using mean ``logfc`` and min ``pvalue``. Genes are coloured by significance
    bucket (up / down / ns) at the supplied thresholds, and ``gene`` is ringed
    + annotated so you can see where the target of interest lands relative to
    the rest of the proteome.

    :param df df: must contain columns ``compound``, ``genes``, ``logfc``, ``pvalue``.
    :param str compound: e.g. ``'SRB-0000615'``.
    :param str gene: gene symbol to highlight (e.g. ``'KDM1B'``); silently
        ignored if not measured for that compound.
    :param float fc_thresh, p_thresh: logfc / p-value thresholds for the
        significance buckets and the dashed reference lines.
    :param float xmin, xmax: x-axis limits (logfc range).
    :param tuple figsize: figure size in inches, used only when ``ax is None``.
    :param int dpi: DPI for the new figure, used only when ``ax is None``.
    :param str up_color, down_color: hex strings for significantly up/down dots.
    :param Axes ax: existing matplotlib Axes to draw into; if ``None`` a new
        figure is created.
    :param str title: optional custom title; default = ``f'{compound}  (N genes)'``.
    :return df: the per-gene aggregate frame
        (``genes``, ``logfc``, ``pvalue``, ``nlog10p``), useful for downstream
        filtering of the volcano data without recomputing the aggregation.
    """
    import matplotlib.pyplot as plt

    sub = df[df['compound'] == compound][['genes', 'logfc', 'pvalue']].dropna()
    if sub.empty:
        print(f'> {compound}: no rows in df_raw')
        return None
    # collapse multi-batch/plate replicates per gene: mean logfc, min p
    agg = (sub.groupby('genes')
              .agg(logfc=('logfc', 'mean'), pvalue=('pvalue', 'min'))
              .reset_index())
    agg['nlog10p'] = -np.log10(agg['pvalue'].clip(lower=1e-300))

    # classify
    up   = (agg['logfc'] >=  fc_thresh) & (agg['pvalue'] <= p_thresh)
    down = (agg['logfc'] <= -fc_thresh) & (agg['pvalue'] <= p_thresh)
    ns   = ~(up | down)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(agg.loc[ns,   'logfc'], agg.loc[ns,   'nlog10p'],
               s=8,  c='lightgrey', edgecolor='none', alpha=0.6,
               label=f'ns ({ns.sum()})')
    ax.scatter(agg.loc[up,   'logfc'], agg.loc[up,   'nlog10p'],
               s=10, c=up_color,    edgecolor='none', alpha=0.85,
               label=f'up ({up.sum()})')
    ax.scatter(agg.loc[down, 'logfc'], agg.loc[down, 'nlog10p'],
               s=10, c=down_color,  edgecolor='none', alpha=0.85,
               label=f'down ({down.sum()})')

    # threshold guides
    ax.axhline(-np.log10(p_thresh), ls='--', lw=0.7, c='#888')
    ax.axvline(+fc_thresh,          ls='--', lw=0.7, c='#888')
    ax.axvline(-fc_thresh,          ls='--', lw=0.7, c='#888')

    # highlight target gene
    tg = agg[agg['genes'] == gene]
    if tg.empty:
        print(f'> {gene} not measured for {compound}')
    else:
        ax.scatter(tg['logfc'], tg['nlog10p'],
                   s=70, facecolor='none', edgecolor='black', lw=1.5, zorder=5)
        ax.annotate(gene,
                    xy=(tg['logfc'].iat[0], tg['nlog10p'].iat[0]),
                    xytext=(8, 6), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', lw=0.7))

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('logfc')
    ax.set_ylabel('-log10(p-value)')
    # title=None -> default caption; title='' -> no title (the interface labels
    # the volcano in its HTML panel instead); any other string -> used verbatim.
    ax.set_title(f'{compound}  ({len(agg):,} genes)' if title is None else title)
    ax.legend(loc='best', fontsize=8, frameon=False)
    plt.tight_layout()
    return agg


# Coarse functional categories for colouring volcanoes "by function" instead of
# by up/down direction. Priority-ordered: the FIRST category whose any keyword is
# a substring of a gene's concatenated GO/Reactome term names wins (so specific
# processes beat generic ones). Keywords match term_name from gene2term.parquet.
CATEGORY_KEYWORDS = [
    ('DNA replication',           ['dna replication', 'pre-replicative', 'replication fork',
                                    'replication origin', 'origin of replication', 'dna-dependent dna']),
    ('DNA repair / HR',           ['dna repair', 'homologous recombination', 'double-strand break',
                                   'mismatch repair', 'excision repair', 'dna damage', 'brca', 'atr ']),
    ('Cell cycle / mitosis',      ['cell cycle', 'mitotic', 'mitosis', 'chromosome segregation',
                                   'spindle', 'cytokinesis', 'kinetochore', 'checkpoint']),
    ('Chromatin / transcription', ['chromatin', 'histone', 'nucleosome', 'transcription',
                                   'rna polymerase', 'methylation', 'demethyl', 'acetylation']),
    ('RNA processing / splicing', ['splic', 'mrna processing', 'rna processing', 'spliceosom',
                                   'rna export', 'mrna stability']),
    ('Translation / ribosome',    ['translation', 'ribosom', 'trna', 'rrna']),
    ('Proteostasis / UPR',        ['unfolded protein', 'proteasom', 'ubiquitin', 'protein folding',
                                   'endoplasmic reticulum stress', 'erad', 'autophag', 'chaperone']),
    ('Lipid / cholesterol',       ['lipid metabolic', 'lipid biosynthe', 'lipid catabolic',
                                   'cholesterol', 'sterol', 'fatty acid', 'lipoprotein',
                                   'ppar', 'triglyceride']),
    ('Xenobiotic / oxidation',    ['cytochrome', 'xenobiotic metabolic', 'drug metab', 'p450',
                                   'biological oxidation', 'phase i -', 'phase ii', 'glutathione']),
    ('ECM / adhesion',            ['extracellular matrix', 'collagen', 'elastic fib', 'elastin',
                                   'cell adhesion', 'cell-matrix', 'integrin', 'laminin',
                                   'basement membrane']),
    ('Cytoskeleton',              ['cytoskeleton', 'actin', 'microtubule', 'intermediate filament',
                                   'tubulin']),
    ('Transport / vesicle',       ['transmembrane transport', 'ion transport', 'vesicle',
                                   'trafficking', 'endocytosis', 'exocytosis', 'slc-mediated',
                                   'solute', 'golgi']),
    ('Signaling',                 ['signal transduction', 'signaling pathway', 'signaling by',
                                   'mapk cascade', 'kinase cascade']),
    ('Immune / inflammation',     ['immune', 'interferon', 'inflammat', 'cytokine', 'antigen',
                                   'complement', 'interleukin']),
    ('Metabolism (other)',        ['metabolic process', 'biosynthetic process', 'catabolic process',
                                   'tca cycle', 'glycolysis', 'oxidative phosphoryl', 'nucleotide']),
]

CATEGORY_COLORS = {
    'DNA replication':           '#1f77b4',
    'DNA repair / HR':           '#0b3d91',
    'Cell cycle / mitosis':      '#17becf',
    'Chromatin / transcription': '#9467bd',
    'RNA processing / splicing': '#c5b0d5',
    'Translation / ribosome':    '#8c564b',
    'Proteostasis / UPR':        '#e377c2',
    'Lipid / cholesterol':       '#bcbd22',
    'Xenobiotic / oxidation':    '#ff7f0e',
    'ECM / adhesion':            '#2ca02c',
    'Cytoskeleton':              '#98df8a',
    'Transport / vesicle':       '#7f7f7f',
    'Signaling':                 '#d62728',
    'Immune / inflammation':     '#e7969c',
    'Metabolism (other)':        '#ffbb78',
    'Other':                     '#cfcfcf',
}


def _term_category(term_name_lower, keywords):
    """Category for a single term name (first keyword match by priority), or None."""
    for cat, kws in keywords:
        if any(k in term_name_lower for k in kws):
            return cat
    return None


def categorize_genes(gene2term, genes=None, keywords=CATEGORY_KEYWORDS, default='Other'):
    """Map each gene to ONE coarse functional category from its GO/Reactome
    annotations, for colouring volcanoes / labelling proteins by function.

    Assignment is **specificity-weighted consensus**, not first-keyword-wins:
    every term is mapped to a category (via the priority-ordered ``keywords``
    table), and each category accumulates a score of ``1/sqrt(term_size)`` over
    the gene's terms (``term_size`` = #genes annotated, so specific terms count
    far more than broad GO ancestors). The gene takes the highest-scoring
    category. This way many supporting specific terms (RAD51's dozen HR/DSB-repair
    terms) beat a single tiny incidental one (an actin term), and broad ancestors
    ("DNA replication", "response to stimulus") barely contribute. Category
    priority breaks ties; genes with no categorised term map to ``default``.

    :param df gene2term: long table from build_cell_signature_annotations
        (``output/cell_signature/gene2term.parquet``) with columns ``gene``,
        ``term_id``, ``term_name``.
    :param genes: optional iterable to restrict to (and guarantee a key for);
        ``None`` categorises every gene present in ``gene2term``.
    :return dict: ``{gene: category}``.
    """
    # term specificity = number of distinct genes annotated (smaller = more specific)
    term_size = gene2term.groupby('term_id')['gene'].nunique()
    g2t = gene2term
    if genes is not None:
        genes = set(genes)
        g2t = g2t[g2t['gene'].isin(genes)]
    prio = {cat: i for i, (cat, _) in enumerate(keywords)}

    # map each unique term -> (category, size, priority); skip uncategorised terms
    terms = g2t[['term_id', 'term_name']].drop_duplicates('term_id')
    tcat = {}
    for tid, tname in terms.itertuples(index=False):
        c = _term_category(str(tname).lower(), keywords)
        if c is not None:
            tcat[tid] = (c, int(term_size.get(tid, 10 ** 9)), prio[c])

    # per gene: specificity-weighted vote (sum 1/sqrt(size)); tie -> priority
    import math
    from collections import defaultdict
    out = {}
    cats_only = g2t[g2t['term_id'].isin(tcat)]
    for gene, grp in cats_only.groupby('gene'):
        scores = defaultdict(float)
        for tid in grp['term_id'].values:
            c, sz, _ = tcat[tid]
            scores[c] += 1.0 / math.sqrt(sz)
        out[gene] = max(scores, key=lambda c: (scores[c], -prio[c]))

    pool = genes if genes is not None else set(g2t['gene'].unique())
    for g in pool:
        out.setdefault(g, default)
    return out


def gene_category_long(gene_category, collection='Function'):
    """Reshape a ``{gene: category}`` map (from :func:`categorize_genes`) into a
    ``gene2term``-shaped long frame so the coarse functional **categories** can
    be fed to :func:`ora_enrichment` / :func:`gsea_preranked` as gene sets —
    giving one enrichment score per function rather than per GO/Reactome term.

    Pass the result with ``collections=('Function',)``; remember categories are
    large sets, so relax the size caps (``max_term_size`` for ORA, ``max_size``
    for GSEA). The ``'Other'`` bucket is kept so the background ``N`` stays the
    full measured proteome — just ignore its row in the output.

    :param dict gene_category: ``{gene: category}``.
    :param str collection: value for the ``collection`` column.
    :return df: columns ``gene``, ``collection``, ``term_id``, ``term_name``
        (``term_id`` == ``term_name`` == the category).
    """
    items = list(gene_category.items())
    cats = [c for _, c in items]
    return pd.DataFrame({
        'gene':       [g for g, _ in items],
        'collection': collection,
        'term_id':    cats,
        'term_name':  cats,
    })


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR (q-values) for a 1-D array of p-values."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    q = np.empty(m)
    q[order] = (p[order] * m / np.arange(1, m + 1))
    # enforce monotonicity from the largest p downward
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    return np.clip(q, 0, 1)


def load_gmt(gmt_paths, *, gene_upper=True):
    """Parse MSigDB-style ``.gmt`` files into a ``gene2term`` long table for
    :func:`gsea_preranked` / :func:`ora_enrichment` — so cluster signatures can be
    qualified against curated gene sets (Hallmark, Reactome, ...) fully locally.

    Each ``.gmt`` line is ``term_name <tab> description <tab> gene1 <tab> gene2 ...``.

    :param dict gmt_paths: ``{collection: path | glob | [paths]}``; ``collection``
        (e.g. ``'Hallmark'``) becomes the ``collection`` column. Globs are expanded
        so version-stamped filenames need not be hard-coded.
    :param bool gene_upper: upper-case gene symbols (match HGNC / MSigDB symbols).
    :raises FileNotFoundError: if a collection's pattern matches no file.
    :return df: columns ``gene``, ``collection``, ``term_id``, ``term_name``
        (one row per gene-in-term; ``term_id`` == ``term_name`` == the set name).
    """
    import glob
    rows = []
    for collection, spec in gmt_paths.items():
        paths = list(spec) if isinstance(spec, (list, tuple)) else glob.glob(str(spec))
        if not paths:
            raise FileNotFoundError(f'no .gmt for collection {collection!r} matching {spec!r}')
        for path in paths:
            with open(path) as fh:
                for line in fh:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        continue
                    term, _desc, *genes = parts
                    for g in genes:
                        if g:
                            rows.append((g.upper() if gene_upper else g, collection, term, term))
    return pd.DataFrame(rows, columns=['gene', 'collection', 'term_id', 'term_name'])


def mean_logfc_rank(df_raw, compounds, *, compound_col='compound', gene_col='genes',
                    logfc_col='logfc', gene_upper=True):
    """Mean per-gene logFC across a set of compounds — the 'mean proteome change'
    of a signature cluster, as a signed ranking for :func:`gsea_preranked`.

    :param df df_raw: per-(compound, gene) differential table.
    :param iterable compounds: compound ids defining the cluster.
    :param bool gene_upper: upper-case gene symbols (match the gene-set table).
    :return Series: index = gene, value = mean logFC across the cluster.
    """
    sub = df_raw[df_raw[compound_col].isin(set(compounds))]
    r = sub.groupby(gene_col)[logfc_col].mean()
    if gene_upper:
        r.index = r.index.astype(str).str.upper()
        r = r.groupby(level=0).mean()
    return r.dropna()


def _term_members(gene2term, collections, universe):
    """{(collection, term_id, term_name): frozenset(genes ∩ universe)} restricted
    to ``collections`` and the gene ``universe`` (set)."""
    g = gene2term[gene2term['collection'].isin(collections) & gene2term['gene'].isin(universe)]
    out = {}
    for key, sub in g.groupby(['collection', 'term_id', 'term_name'], sort=False):
        out[key] = frozenset(sub['gene'])
    return out


def ora_enrichment(gene_set, background, gene2term, *,
                   collections=('GO_BP', 'Reactome'),
                   min_overlap=3, max_term_size=500, fdr=None, top_n=None):
    """Over-representation analysis via the **hypergeometric test** (one-tailed
    Fisher's exact): is each GO/Reactome term over-represented in ``gene_set``
    vs ``background``?

    Operates on a *thresholded* set (e.g. the significant-down genes of a
    volcano). The background should be the **measured proteome** — using the
    whole genome inflates membrane/secreted terms. Per collection the universe
    ``N`` is the background genes carrying ≥1 annotation in that collection, and
    ``n`` the gene_set genes within it; ``K``/``k`` are the term's hits in
    background / gene_set. p = ``hypergeom.sf(k-1, N, K, n)``; BH-FDR is pooled
    across all tested terms.

    :param gene_set: iterable of query genes (one direction at a time).
    :param background: iterable of measured genes (the universe).
    :param df gene2term: long table (gene, collection, term_id, term_name).
    :param collections: which annotation collections to test.
    :param int min_overlap: drop terms with < this many query hits (k).
    :param int max_term_size: drop terms broader than this (K) — generic noise.
    :param float fdr: if given, keep only rows with q <= fdr.
    :param int top_n: if given, return only the top_n by p-value.
    :return df: columns collection, term_id, term_name, k, K, n, N, p, fdr,
        overlap_genes — sorted by p-value.
    """
    from scipy.stats import hypergeom
    bg = set(background)
    gs = set(gene_set) & bg
    rows = []
    for coll in collections:
        members = _term_members(gene2term, [coll], bg)
        annot_bg = set().union(*members.values()) if members else set()
        N = len(annot_bg)
        n = len(gs & annot_bg)
        if N == 0 or n == 0:
            continue
        for (c, tid, tname), genes in members.items():
            K = len(genes)
            if K > max_term_size:
                continue
            k = len(genes & gs)
            if k < min_overlap:
                continue
            p = float(hypergeom.sf(k - 1, N, K, n))
            rows.append((c, tid, tname, k, K, n, N, p, sorted(genes & gs)))
    if not rows:
        return pd.DataFrame(columns=['collection', 'term_id', 'term_name',
                                     'k', 'K', 'n', 'N', 'p', 'fdr', 'overlap_genes'])
    out = pd.DataFrame(rows, columns=['collection', 'term_id', 'term_name',
                                      'k', 'K', 'n', 'N', 'p', 'overlap_genes'])
    out['fdr'] = _bh_fdr(out['p'].values)
    out = out.sort_values('p').reset_index(drop=True)
    out = out[['collection', 'term_id', 'term_name', 'k', 'K', 'n', 'N', 'p', 'fdr', 'overlap_genes']]
    if fdr is not None:
        out = out[out['fdr'] <= fdr].reset_index(drop=True)
    if top_n is not None:
        out = out.head(top_n).reset_index(drop=True)
    return out


def _running_es(pos, w, N):
    """Weighted GSEA running enrichment score. ``pos`` = ascending member
    positions in the ranked list, ``w`` = |stat| weights (len N). Returns
    (ES, peak_index)."""
    k = pos.size
    inc = np.zeros(N)
    inc[pos] = w[pos]
    s = inc.sum()
    if s == 0:
        return 0.0, 0
    inc /= s
    dec = np.full(N, 1.0 / (N - k))
    dec[pos] = 0.0
    run = np.cumsum(inc - dec)
    peak = int(np.argmax(np.abs(run)))
    return float(run[peak]), peak


def _null_es_for_size(k, n_perm, w, rng):
    """Vectorised null ES for random gene sets of size ``k`` against weights
    ``w`` — the null depends only on size, so callers cache by ``k``."""
    N = w.size
    rand = rng.random((n_perm, N))
    sel = np.argpartition(rand, k - 1, axis=1)[:, :k]          # n_perm random size-k sets
    rows = np.repeat(np.arange(n_perm), k)
    cols = sel.ravel()
    inc = np.zeros((n_perm, N))
    inc[rows, cols] = w[cols]
    inc /= inc.sum(axis=1, keepdims=True)
    dec = np.full((n_perm, N), 1.0 / (N - k))
    dec[rows, cols] = 0.0
    run = np.cumsum(inc - dec, axis=1)
    idx = np.argmax(np.abs(run), axis=1)
    return run[np.arange(n_perm), idx]


def gsea_preranked(ranks, gene2term, *,
                   collections=('GO_BP', 'Reactome'),
                   min_size=10, max_size=300, n_perm=1000,
                   weight=1.0, seed=0, fdr=None, top_n=None):
    """**GSEA-preranked** (Subramanian 2005), threshold-free: rank *all* measured
    genes by a signed statistic and test whether each term is concentrated at
    the top (induced) or bottom (suppressed) of the ranking. Catches coordinated
    subtle shifts (a whole complex nudged down) that no single gene clears the
    significance cutoff for — the complement to :func:`ora_enrichment`.

    Weighted running ES (``weight``=1). Significance from a **size-matched
    permutation null** (random gene sets of equal size; the null depends only on
    size, so it is computed once per size and reused). NES = ES / mean(|same-sign
    null|); nominal p = fraction of same-sign null with |ES| ≥ |observed|;
    BH-FDR across tested terms.

    :param Series ranks: index = gene, value = signed statistic (e.g.
        ``sign(logfc) * -log10(pvalue)``) over the measured proteome.
    :param df gene2term: long table (gene, collection, term_id, term_name).
    :param int min_size, max_size: term size bounds (genes present in ranking).
    :param int n_perm: permutations for the null (1000 default).
    :param float weight: ES weighting exponent on |stat| (GSEA default 1).
    :param int seed: RNG seed (reproducible).
    :param float fdr: if given, keep only rows with q <= fdr.
    :param int top_n: if given, return only the top_n by |NES|.
    :return df: collection, term_id, term_name, size, ES, NES, p, fdr,
        direction ('up'/'down'), leading_edge — sorted by p then |NES|.
    """
    ranks = pd.Series(ranks).dropna().sort_values(ascending=False)
    genes_sorted = list(ranks.index)
    pos_of = {g: i for i, g in enumerate(genes_sorted)}
    w = np.abs(ranks.values.astype(float)) ** weight
    N = len(genes_sorted)
    universe = set(genes_sorted)
    members = _term_members(gene2term, collections, universe)

    rng = np.random.default_rng(seed)
    null_cache = {}
    rows = []
    for (coll, tid, tname), genes in members.items():
        size = len(genes)
        if size < min_size or size > max_size:
            continue
        pos = np.sort(np.fromiter((pos_of[g] for g in genes), dtype=int, count=size))
        es, peak = _running_es(pos, w, N)
        if size not in null_cache:
            null_cache[size] = _null_es_for_size(size, n_perm, w, rng)
        null = null_cache[size]
        same = null[null > 0] if es >= 0 else null[null < 0]
        if same.size == 0:
            nes, p = np.nan, 1.0
        else:
            nes = es / np.abs(same).mean()
            p = (np.sum(np.abs(same) >= abs(es)) + 1) / (same.size + 1)
        # leading edge = members driving the peak
        if es >= 0:
            le = [genes_sorted[i] for i in pos if i <= peak]
        else:
            le = [genes_sorted[i] for i in pos if i >= peak]
        rows.append((coll, tid, tname, size, es, nes, p,
                     'up' if es >= 0 else 'down', le))
    if not rows:
        return pd.DataFrame(columns=['collection', 'term_id', 'term_name', 'size',
                                     'ES', 'NES', 'p', 'fdr', 'direction', 'leading_edge'])
    out = pd.DataFrame(rows, columns=['collection', 'term_id', 'term_name', 'size',
                                      'ES', 'NES', 'p', 'direction', 'leading_edge'])
    out['fdr'] = _bh_fdr(out['p'].values)
    out = out.sort_values(['p', 'NES'], key=lambda s: s if s.name != 'NES' else -s.abs())
    out = out.reset_index(drop=True)
    out = out[['collection', 'term_id', 'term_name', 'size', 'ES', 'NES', 'p',
               'fdr', 'direction', 'leading_edge']]
    if fdr is not None:
        out = out[out['fdr'] <= fdr].reset_index(drop=True)
    if top_n is not None:
        out = out.head(top_n).reset_index(drop=True)
    return out


def _function_enrich_one(cmpd, sub, g2cat, categories, *,
                         gene_col, logfc_col, p_col, sig_col,
                         n_perm, seed, min_overlap, run_ora, run_gsea):
    """Per-compound function-level enrichment (module-level so joblib pickles it
    cleanly). Collapses plate replicates per gene, then scores each functional
    category with ORA (on the significant down/up sets) and GSEA-preranked
    (signed -log10 p). Returns one row per category for this compound, or None."""
    sub = sub.dropna(subset=[gene_col, logfc_col, p_col])
    if sub.empty:
        return None
    a = (sub.groupby(gene_col)
            .agg(logfc=(logfc_col, 'mean'), pvalue=(p_col, 'min'), sig=(sig_col, 'max'))
            .reset_index())
    bg = set(a[gene_col])
    down = set(a.loc[(a['sig'] > 0) & (a['logfc'] < 0), gene_col])
    up = set(a.loc[(a['sig'] > 0) & (a['logfc'] > 0), gene_col])
    rec = {c: {'compound': cmpd, 'function': c, 'n_down': 0, 'n_up': 0,
               'ora_down_fdr': np.nan, 'ora_up_fdr': np.nan, 'gsea_NES': np.nan,
               'gsea_fdr': np.nan, 'gsea_direction': None, 'n_measured': len(bg)}
           for c in categories}

    g2c = dict(zip(g2cat['gene'], g2cat['term_name']))
    for g in down:
        c = g2c.get(g)
        if c in rec:
            rec[c]['n_down'] += 1
    for g in up:
        c = g2c.get(g)
        if c in rec:
            rec[c]['n_up'] += 1

    if run_ora:
        if down:
            od = ora_enrichment(down, bg, g2cat, collections=('Function',),
                                min_overlap=min_overlap, max_term_size=10 ** 9)
            for _, r in od.iterrows():
                if r['term_name'] in rec:
                    rec[r['term_name']]['ora_down_fdr'] = r['fdr']
        if up:
            ou = ora_enrichment(up, bg, g2cat, collections=('Function',),
                                min_overlap=min_overlap, max_term_size=10 ** 9)
            for _, r in ou.iterrows():
                if r['term_name'] in rec:
                    rec[r['term_name']]['ora_up_fdr'] = r['fdr']
    if run_gsea:
        ranks = pd.Series((np.sign(a['logfc']) * -np.log10(a['pvalue'].clip(lower=1e-300))).values,
                          index=a[gene_col])
        gs = gsea_preranked(ranks, g2cat, collections=('Function',),
                            min_size=5, max_size=10 ** 9, n_perm=n_perm, seed=seed)
        for _, r in gs.iterrows():
            if r['term_name'] in rec:
                rec[r['term_name']]['gsea_NES'] = r['NES']
                rec[r['term_name']]['gsea_fdr'] = r['fdr']
                rec[r['term_name']]['gsea_direction'] = r['direction']
    return pd.DataFrame(list(rec.values()))


def function_enrichment_all(df_raw, gene_category, *,
                            compound_col='compound', gene_col='genes',
                            logfc_col='logfc', p_col='pvalue', sig_col='significant',
                            n_perm=1000, n_jobs=8, run_ora=True, run_gsea=True,
                            min_overlap=3, seed=0, drop_other=True, verbose=True):
    """Per-compound enrichment of the coarse **functional categories**, for every
    compound in ``df_raw`` — parallelised across compounds with joblib.

    For each compound: collapse plate replicates per gene (mean logfc / min
    pvalue / significant-if-any), then score every function with ORA
    (hypergeometric on the significant down/up sets, measured proteome as
    background) and GSEA-preranked (signed -log10 p, threshold-free). Both FDRs
    are BH-corrected across the ~15 functions within each compound.

    :param df df_raw: per-(compound, gene[, plate]) table with ``compound_col``,
        ``gene_col``, ``logfc_col``, ``p_col``, ``sig_col``.
    :param dict gene_category: ``{gene: function}`` from :func:`categorize_genes`.
    :param int n_perm: GSEA permutations per compound (1000 default; 500 ~2x faster).
    :param int n_jobs: parallel workers (joblib loky).
    :param bool run_ora, run_gsea: toggle either test.
    :param bool drop_other: drop the ``'Other'`` category rows from the output.
    :return df: tidy ``compound × function`` table — columns ``compound``,
        ``function``, ``n_down``, ``n_up``, ``ora_down_fdr``, ``ora_up_fdr``,
        ``gsea_NES``, ``gsea_fdr``, ``gsea_direction``, ``n_measured``.
    """
    from joblib import Parallel, delayed
    import contextlib
    import joblib

    g2cat = gene_category_long(gene_category)
    categories = sorted(g2cat['term_name'].unique())
    groups = [(c, g[[gene_col, logfc_col, p_col, sig_col]])
              for c, g in df_raw[[compound_col, gene_col, logfc_col, p_col, sig_col]]
              .groupby(compound_col)]

    @contextlib.contextmanager
    def _tqdm_joblib(pbar):
        class _Cb(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *a, **k):
                pbar.update(n=self.batch_size)
                return super().__call__(*a, **k)
        old = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = _Cb
        try:
            yield pbar
        finally:
            joblib.parallel.BatchCompletionCallBack = old
            pbar.close()

    def _run():
        return Parallel(n_jobs=n_jobs)(
            delayed(_function_enrich_one)(
                c, sub, g2cat, categories,
                gene_col=gene_col, logfc_col=logfc_col, p_col=p_col, sig_col=sig_col,
                n_perm=n_perm, seed=seed, min_overlap=min_overlap,
                run_ora=run_ora, run_gsea=run_gsea)
            for c, sub in groups)

    if verbose:
        with _tqdm_joblib(tqdm(total=len(groups), desc='function enrichment', unit='cmp')):
            results = _run()
    else:
        results = _run()

    out = pd.concat([r for r in results if r is not None], ignore_index=True)
    if drop_other:
        out = out[out['function'] != 'Other'].reset_index(drop=True)
    return out


def plot_function_enrichment(df, *, nes_col='gsea_NES', fdr_col='gsea_fdr',
                             label_col='function', sig=0.05,
                             show_counts=False, down_count_col='n_down',
                             up_count_col='n_up',
                             down_color='#1f77b4', up_color='#d62728',
                             ns_color='#cfcfcf', ax=None, title=None,
                             width=7, height=None):
    """Diverging **lollipop** of per-function enrichment for ONE compound — a
    readable replacement for the flat enrichment table.

    Each function is a stem from 0 to its GSEA NES, sorted so the most
    suppressed sit at the bottom and the most induced at the top. Colour encodes
    direction (suppressed = ``down_color``, induced = ``up_color``); bars are
    full-colour + dotted ``*`` when significant (``fdr_col < sig``) and faded
    grey otherwise, so the eye goes straight to the real signal.

    :param df df: one compound's rows from :func:`function_enrichment_all`
        (or the ``func_enrich`` table) — needs ``label_col``, ``nes_col``,
        ``fdr_col``.
    :param float sig: FDR threshold for the "significant" styling.
    :param bool show_counts: annotate each bar with ``n_down↓ n_up↑`` (the
        significant-gene counts) just above the stem; bars with both zero are
        left unlabelled to reduce clutter. Needs ``down_count_col`` /
        ``up_count_col`` on ``df``.
    :param Axes ax: draw into an existing Axes; new figure if ``None``.
    :return Axes: the axis drawn into.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    has_counts = show_counts and {down_count_col, up_count_col}.issubset(df.columns)
    keep = [label_col, nes_col, fdr_col] + ([down_count_col, up_count_col] if has_counts else [])
    d = df[keep].dropna(subset=[nes_col]).copy()
    d = d.sort_values(nes_col).reset_index(drop=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(width, height or max(3.0, 0.4 * len(d))), dpi=110)

    for yi, r in d.iterrows():
        nes, fdr = r[nes_col], r[fdr_col]
        signif = pd.notna(fdr) and fdr < sig
        base = up_color if nes >= 0 else down_color
        col = base if signif else ns_color
        ax.plot([0, nes], [yi, yi], color=col, lw=2, alpha=0.9 if signif else 0.55, zorder=1)
        ax.scatter([nes], [yi], s=95 if signif else 40, color=col,
                   edgecolor='black' if signif else 'none', lw=0.8, zorder=2)
        if signif:
            ax.text(nes + (0.05 if nes >= 0 else -0.05), yi, '*', fontsize=14,
                    fontweight='bold', va='center', ha='left' if nes >= 0 else 'right')
        if has_counts:
            nd = 0 if pd.isna(r[down_count_col]) else int(r[down_count_col])
            nu = 0 if pd.isna(r[up_count_col]) else int(r[up_count_col])
            if nd or nu:
                ax.text(nes / 2, yi + 0.30, f'{nd}↓ {nu}↑', ha='center',
                        va='bottom', fontsize=6.5, color='#333')

    ax.axvline(0, color='#444', lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d[label_col])
    ax.set_xlabel('GSEA NES   (← suppressed     induced →)')
    m = max(abs(d[nes_col].min()), abs(d[nes_col].max())) * 1.18
    ax.set_xlim(-m, m)
    ax.grid(axis='x', ls=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title or 'Per-function enrichment')
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=down_color,
               markeredgecolor='black', markersize=8, label=f'suppressed (FDR<{sig:g})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=up_color,
               markeredgecolor='black', markersize=8, label=f'induced (FDR<{sig:g})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ns_color,
               markersize=7, label='n.s.'),
    ], loc='lower right', fontsize=8, frameon=False)
    plt.tight_layout()
    return ax


def select_strong_signature_compounds(
        func_enrich_all, MS, *,
        activity_bins=('Low (2-10)', 'Medium (11-25)', 'High (>25)'),
        max_abs_nes=2.0, nes_col='gsea_NES', compound_col='compound',
        activity_col='activity', verbose=True):
    """
    Restrict the per-(compound, function) enrichment table to compounds with a
    **strong, well-defined cell signature** — the cohort for the
    structure↔cell-state analysis (signature clustering + chemistry→class ML).
    Weak/diffuse compounds otherwise contribute near-noise NES vectors that blur
    the very association being tested.

    A compound qualifies only if BOTH hold:

      1. **Response-magnitude floor** — its MS ``activity_col`` is in
         ``activity_bins`` (``ndown ≥ 2`` by default), dropping Silent / Single.
      2. **Dominant signature axis** — its peak ``|NES|`` across functions is
         ``≥ max_abs_nes``, i.e. the signature has a clear leading direction.

    Deliberately **not** gated on ``gsea_fdr``: GSEA-preranked flags a
    "significant" function for ~all compounds (including Silent), so the FDR
    can't separate signal from noise here; magnitude + peak NES can.

    :param df func_enrich_all: tidy output of :func:`function_enrichment_all`.
    :param df MS: unified MS metadata (``compound`` | ``activity`` | …).
    :param activity_bins: MS activity labels to keep.
    :param float max_abs_nes: minimum peak ``|NES|`` for a compound to qualify.
    :param str nes_col: GSEA NES column in ``func_enrich_all``.
    :param str compound_col: compound id column (in both frames).
    :param str activity_col: activity column in ``MS``.
    :param bool verbose: print aggregate before/after counts (no compound IDs).
    :return: row subset of ``func_enrich_all`` for the qualifying compounds.
    """
    active = set(MS.loc[MS[activity_col].isin(list(activity_bins)), compound_col])
    peak = func_enrich_all.groupby(compound_col)[nes_col].apply(
        lambda s: np.nanmax(np.abs(s.to_numpy())) if s.notna().any() else np.nan)
    strong = set(peak.index[peak >= max_abs_nes])
    keep = active & strong
    out = func_enrich_all[func_enrich_all[compound_col].isin(keep)]
    if verbose:
        n_in, n_out = func_enrich_all[compound_col].nunique(), out[compound_col].nunique()
        print(f'> strong-signature cohort: activity in {list(activity_bins)} '
              f'AND max|NES| >= {max_abs_nes}')
        print(f'  {n_in:,} -> {n_out:,} compounds '
              f'(magnitude floor: {len(active):,} | peak-NES: {len(strong):,})')
    return out


def label_signature_clusters(means, prefix='C'):
    """
    Name each signature cluster by its single most-extreme function (largest
    ``|mean NES|``) with a direction arrow — e.g. ``C0: Cell cycle ↓``.

    Replaces the older ``down {idxmin} | up {idxmax}`` two-sided label. On a
    dominant single up/down axis (small K) the clusters are mirror images, so the
    secondary pole (e.g. 'Transport / vesicle') appears as ``up`` in one cluster
    and ``down`` in the other and reads as noise; the strongest pole is the
    defining phenotype. K-agnostic: one label per row of ``means``.

    :param df means: cluster × function mean-NES table (index = cluster id), as
        produced by ``NES.groupby(labels).mean()``.
    :param str prefix: cluster-id prefix (``C`` -> ``C0``, ``C1`` …).
    :return: dict ``{cluster_id: label}`` keyed by the index of ``means``.
    """
    def _dom(row):
        f = row.abs().idxmax()                                 # strongest |NES| function
        return f"{f.split(' / ')[0]} {'↓' if row[f] < 0 else '↑'}"
    return {c: f'{prefix}{c}: {_dom(means.loc[c])}' for c in means.index}


def signature_matrix_from_enrichment(func_enrich_all, *, value_col='gsea_NES',
                                     compound_col='compound', func_col='function',
                                     fill=0.0):
    """Pivot :func:`function_enrichment_all` output into a compound × function
    **fingerprint matrix** — each compound a row, each functional category a
    column, value = ``value_col`` (GSEA NES by default). This is the per-compound
    "cell signature" used for similarity. Missing function/compound cells (e.g. a
    function dropped for a small measured set) are filled with ``fill`` (0 =
    neutral NES).

    :param df func_enrich_all: tidy output of :func:`function_enrichment_all`.
    :return df: index = compound, columns = function, values = NES.
    """
    M = func_enrich_all.pivot_table(index=compound_col, columns=func_col, values=value_col)
    return M.fillna(fill)


def compound_distance_matrix(features, *, metric='cosine', compound_col=None,
                             exclude_self=True):
    """Pairwise compound × compound **distance** matrix (smaller = more similar),
    in the same DataFrame layout as ``Rdkit_tools.get_*_distance_matrix`` so it
    feeds straight into ``Rdkit_tools.get_NN_from_dist_matrix(d, top=N)``.

    Works on any per-compound feature matrix — the functional fingerprint from
    :func:`signature_matrix_from_enrichment` (cosine on the 15-D NES vector =
    "same cell signature"), or a gene-level logfc table (use ``metric='correlation'``
    for a CMap-style connectivity distance).

    ``cosine`` distance (``1 - cosine_similarity``) is the default: it compares
    the *pattern* of up/down functions and is invariant to overall signature
    magnitude, so a strong and a mild proliferation-arrest compound still score
    as near neighbours.

    :param features: either a DataFrame indexed by compound (feature columns
        only), or one with a ``compound_col`` column + feature columns.
    :param str metric: any ``sklearn.metrics.pairwise_distances`` metric
        (``'cosine'``, ``'euclidean'``, ``'correlation'``, ...).
    :param str compound_col: name of the compound-id column if ``features`` isn't
        already indexed by compound; ``None`` -> use the index.
    :param bool exclude_self: set the diagonal to NaN so a compound's own row is
        dropped from nearest-neighbour queries (NaN sorts last in
        ``get_NN_from_dist_matrix``).
    :return df: square distance matrix, index = columns = compound ids.
    """
    from sklearn.metrics import pairwise_distances
    if compound_col is not None and compound_col in getattr(features, 'columns', []):
        idx = list(features[compound_col])
        X = features.drop(columns=[compound_col]).to_numpy(dtype=float)
    else:
        idx = list(features.index)
        X = features.to_numpy(dtype=float)
    D = pairwise_distances(X, metric=metric)
    if exclude_self:
        np.fill_diagonal(D, np.nan)          # fill on the writable ndarray
    return pd.DataFrame(D, index=idx, columns=idx)


def per_class_report(y_true, y_pred, proba, classes, names=None, sep_width=82):
    """One-vs-rest per-class metrics (Accuracy / F1 / ROC_auc / PR_auc / MCC) plus
    a MACRO average, printed in the project's standard format::

        > <label>:	 Accuracy: .., F1: .., ROC_auc: .., PR_auc: .., MCC: ..
        ----
        >> MACRO:	 Accuracy: .., F1: .., ROC_auc: .., PR_auc: .., MCC: ..

    Each class is scored as a binary one-vs-rest problem. ``proba`` columns must
    align to ``classes`` order (e.g. from ``cross_val_predict(method='predict_proba')``
    or ``clf.predict_proba``). ``names`` optionally maps class -> display label.

    :param y_true, y_pred: arrays of class labels.
    :param proba: (n_samples, n_classes) probability matrix aligned to ``classes``.
    :param classes: ordered class labels matching ``proba`` columns.
    :param dict names: optional {class: display label}.
    :return df: per-class + MACRO metrics table (also printed).
    """
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 average_precision_score, matthews_corrcoef)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sep = '-' * sep_width
    rows = []
    for i, c in enumerate(classes):
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        pc = proba[:, i]
        a = accuracy_score(yt, yp)
        f = f1_score(yt, yp, zero_division=0)
        r = roc_auc_score(yt, pc) if 0 < yt.sum() < len(yt) else float('nan')
        p = average_precision_score(yt, pc) if yt.sum() > 0 else float('nan')
        m = matthews_corrcoef(yt, yp) if yt.sum() > 0 else float('nan')
        lab = (names or {}).get(c, c)
        print(f'> {lab}:\t Accuracy: {a:.2f}, F1: {f:.2f}, ROC_auc: {r:.2f}, PR_auc: {p:.2f}, MCC: {m:.2f}')
        print(sep)
        rows.append({'class': lab, 'Accuracy': a, 'F1': f, 'ROC_auc': r, 'PR_auc': p, 'MCC': m})
    out = pd.DataFrame(rows)
    macro = out[['Accuracy', 'F1', 'ROC_auc', 'PR_auc', 'MCC']].mean()
    print(sep)
    print(f">> MACRO:\t Accuracy: {macro['Accuracy']:.2f}, F1: {macro['F1']:.2f}, "
          f"ROC_auc: {macro['ROC_auc']:.2f}, PR_auc: {macro['PR_auc']:.2f}, MCC: {macro['MCC']:.2f}")
    return out


def plot_volcano_significant(df, uniquecontrast, gene,
                             *,
                             key='uniquecontrast',
                             sig_col='significant',
                             fc_thresh=1.0, p_thresh=0.05,
                             xmin=-5.0, xmax=5.0,
                             figsize=(6, 6), dpi=100,
                             up_color='#008bfb', down_color='#ff0051',
                             ns_color='lightgrey',
                             gene_category=None, category_colors=None,
                             ax=None, title=None):
    """
    Volcano for a single experiment (``uniquecontrast``), colouring ONLY the
    targets flagged *significant* — up (logfc > 0) in ``up_color``, down
    (logfc < 0) in ``down_color`` — with every other gene left grey. ``gene``
    is ringed + annotated so you can see where the target of interest lands.

    Significance is read from the ``sig_col`` column when present (e.g.
    ``FBX_MEASURE``'s ``significant`` 0/1 flag); if that column is absent it
    falls back to ``|logfc| >= fc_thresh & pvalue <= p_thresh``. The dashed
    reference lines always reflect ``fc_thresh`` / ``p_thresh``.

    Sibling of :func:`plot_volcano`, but keyed on ``uniquecontrast`` (one
    experiment) instead of ``compound``, and gating colour on the significance
    flag rather than the thresholds.

    :param df df: long table with ``key``, ``genes``, ``logfc``, ``pvalue``
        (optionally ``sig_col``). For the FBX data this is ``FBX_MEASURE``.
    :param str uniquecontrast: the experiment id to plot (value in ``key``).
    :param str gene: gene symbol to ring/annotate; ``None`` -> no highlight
        (compound-level view); a symbol absent from the data is silently skipped.
    :param str key: column identifying the experiment (default ``'uniquecontrast'``).
    :param str sig_col: significance-flag column; threshold fallback if missing.
    :param float fc_thresh, p_thresh: thresholds for the dashed guides (and the
        significance fallback when ``sig_col`` is absent).
    :param float xmin, xmax: x-axis (logfc) limits.
    :param str up_color, down_color, ns_color: dot colours (direction mode).
    :param dict gene_category: optional ``{gene: category}`` map (e.g. from
        :func:`categorize_genes`). When given, significant points are coloured by
        functional *category* instead of red/blue up/down — direction is then
        encoded by marker shape (``^`` up, ``v`` down) and the legend lists the
        categories present (with counts). ``None`` -> classic up/down colouring.
    :param dict category_colors: ``{category: hex}`` palette (default
        ``CATEGORY_COLORS``); only used when ``gene_category`` is given.
    :param Axes ax: draw into an existing Axes; new figure if ``None``.
    :param str title: ``None`` -> default caption; ``''`` -> no title; else verbatim.
    :return df: per-gene aggregate (``genes``, ``logfc``, ``pvalue``,
        ``nlog10p``, ``significant``).
    """
    import matplotlib.pyplot as plt

    has_sig = sig_col in df.columns
    cols = ['genes', 'logfc', 'pvalue'] + ([sig_col] if has_sig else [])
    sub = df[df[key] == uniquecontrast][cols].dropna(subset=['genes', 'logfc', 'pvalue'])
    if sub.empty:
        print(f'> {uniquecontrast}: no rows for {key}')
        return None

    # collapse any duplicate gene rows (e.g. multiple protein groups): mean
    # logfc, min p-value, and significant-if-any across them.
    aggspec = {'logfc': ('logfc', 'mean'), 'pvalue': ('pvalue', 'min')}
    if has_sig:
        aggspec['significant'] = (sig_col, 'max')
    agg = sub.groupby('genes').agg(**aggspec).reset_index()
    agg['nlog10p'] = -np.log10(agg['pvalue'].clip(lower=1e-300))

    if 'significant' in agg.columns:
        sig = agg['significant'].astype(float) > 0
    else:
        sig = (agg['logfc'].abs() >= fc_thresh) & (agg['pvalue'] <= p_thresh)
        agg['significant'] = sig.astype(int)
    up   = sig & (agg['logfc'] > 0)
    down = sig & (agg['logfc'] < 0)
    ns   = ~sig

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # non-significant background (both colouring modes)
    ax.scatter(agg.loc[ns, 'logfc'], agg.loc[ns, 'nlog10p'],
               s=8, c=ns_color, edgecolor='none', alpha=0.5,
               label=f'ns ({int(ns.sum())})')

    legend_handles = None
    if gene_category is not None:
        # colour-by-function: one colour per category, ^ = up / v = down
        from matplotlib.lines import Line2D
        cmap = category_colors or CATEGORY_COLORS
        cats = agg['genes'].map(lambda g: gene_category.get(g, 'Other'))
        # categories present among significant points, most-frequent first
        present = cats[sig].value_counts()
        for category in present.index:
            col = cmap.get(category, cmap.get('Other', '#cfcfcf'))
            cm = sig & (cats == category)
            ax.scatter(agg.loc[cm & (agg['logfc'] > 0), 'logfc'],
                       agg.loc[cm & (agg['logfc'] > 0), 'nlog10p'],
                       s=22, marker='^', c=col, edgecolor='none', alpha=0.9)
            ax.scatter(agg.loc[cm & (agg['logfc'] < 0), 'logfc'],
                       agg.loc[cm & (agg['logfc'] < 0), 'nlog10p'],
                       s=22, marker='v', c=col, edgecolor='none', alpha=0.9)
        legend_handles = [Line2D([0], [0], marker='o', linestyle='', markersize=6,
                                 color=cmap.get(c, '#cfcfcf'), label=f'{c} ({n})')
                          for c, n in present.items()]
        legend_handles += [
            Line2D([0], [0], marker='^', linestyle='', markersize=6,
                   color='#444', label='▲ up-modulated'),
            Line2D([0], [0], marker='v', linestyle='', markersize=6,
                   color='#444', label='▼ down-modulated'),
        ]
    else:
        ax.scatter(agg.loc[up, 'logfc'], agg.loc[up, 'nlog10p'],
                   s=12, c=up_color, edgecolor='none', alpha=0.9,
                   label=f'sig up ({int(up.sum())})')
        ax.scatter(agg.loc[down, 'logfc'], agg.loc[down, 'nlog10p'],
                   s=12, c=down_color, edgecolor='none', alpha=0.9,
                   label=f'sig down ({int(down.sum())})')

    # threshold guides
    ax.axhline(-np.log10(p_thresh), ls='--', lw=0.7, c='#888')
    ax.axvline(+fc_thresh,          ls='--', lw=0.7, c='#888')
    ax.axvline(-fc_thresh,          ls='--', lw=0.7, c='#888')

    # highlight target gene (gene=None -> no highlight, e.g. compound-level view)
    if gene is not None:
        tg = agg[agg['genes'] == gene]
        if tg.empty:
            print(f'> {gene} not measured in {uniquecontrast}')
        else:
            ax.scatter(tg['logfc'], tg['nlog10p'],
                       s=70, facecolor='none', edgecolor='black', lw=1.5, zorder=5)
            ax.annotate(gene,
                        xy=(tg['logfc'].iat[0], tg['nlog10p'].iat[0]),
                        xytext=(8, 6), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', lw=0.7))

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('logfc')
    ax.set_ylabel('-log10(p-value)')
    if title is None:
        title = f'{uniquecontrast}  ({len(agg):,} genes, {int(sig.sum())} significant)'
    ax.set_title(title)
    if legend_handles is not None:
        ax.legend(handles=legend_handles, loc='upper left',
                  bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False,
                  title='function', title_fontsize=8)
    else:
        ax.legend(loc='best', fontsize=8, frameon=False)
    plt.tight_layout()
    return agg


def _build_gene_patents_html_map(gene_patents_df, top_n, depmap_url_template):
    """Build {gene: <html>} for the per-gene patents panel.

    Expects a DataFrame with columns ``gene``, ``Company``, ``Patent Number``
    (``Year`` optional, used only for sort). Returns an empty dict if the
    input is None or missing the required columns. Caller serialises the
    dict to JSON and injects it as a global ``window.__GENE_PATENTS__``.
    """
    if gene_patents_df is None or gene_patents_df.empty:
        return {}
    required = {'gene', 'Company', 'Patent Number'}
    if not required.issubset(gene_patents_df.columns):
        return {}

    out = {}
    has_year = 'Year' in gene_patents_df.columns
    sort_cols = ['gene', 'Year'] if has_year else ['gene']
    asc       = [True, False]    if has_year else [True]
    df = (gene_patents_df.dropna(subset=['gene'])
                          .sort_values(sort_cols, ascending=asc, na_position='last'))
    for gene, grp in df.groupby('gene', sort=False):
        rows_html = []
        for _, r in grp.head(top_n).iterrows():
            comp  = str(r.get('Company', '')) or '—'
            patno = str(r.get('Patent Number', '')) or '—'
            yr    = ''
            if has_year and pd.notna(r.get('Year')):
                try:
                    yr = f' <span style="color:#999;">({int(r["Year"])})</span>'
                except Exception:
                    yr = ''
            rows_html.append(
                f'<tr><td style="padding-right:8px;font-weight:600;">{comp}</td>'
                f'<td style="font-family:ui-monospace,monospace;color:#333;">{patno}{yr}</td></tr>'
            )
        if not rows_html:
            continue
        depmap = depmap_url_template.format(gene=gene)
        out[gene] = (
            f'<div class="pat-header">'
            f'<span class="pat-gene">{gene}</span>'
            f' <a class="pat-depmap" href="{depmap}" target="_blank" '
            f'rel="noopener" title="open in DepMap">DepMap ↗</a>'
            f'</div>'
            f'<table class="pat-table"><tbody>{"".join(rows_html)}</tbody></table>'
        )
    return out


def _volcano_svg_string(df, uniquecontrast, gene,
                        *,
                        key='uniquecontrast', sig_col='significant',
                        fc_thresh=1.0, p_thresh=0.05,
                        xmin=-8.0, xmax=8.0, size_px=350,
                        up_color='#008bfb', down_color='#ff0051', ns_color='lightgrey'):
    """
    Render the significant-only volcano (one ``uniquecontrast``) to an *interactive*
    SVG string. The dense non-significant cloud is rasterised (keeps the file small),
    while each significant point is a vector marker carrying a ``<title>`` (gene
    name) so a browser shows a native hover tooltip — like the 3D dots. The target
    ``gene`` is ringed + annotated. Returns ``''`` on empty/failure.
    """
    import io
    import xml.etree.ElementTree as ET
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_sig = sig_col in df.columns
    cols = ['genes', 'logfc', 'pvalue'] + ([sig_col] if has_sig else [])
    sub = df[df[key] == uniquecontrast][cols].dropna(subset=['genes', 'logfc', 'pvalue'])
    if sub.empty:
        return ''
    aggspec = {'logfc': ('logfc', 'mean'), 'pvalue': ('pvalue', 'min')}
    if has_sig:
        aggspec['significant'] = (sig_col, 'max')
    agg = sub.groupby('genes').agg(**aggspec).reset_index()
    agg['nlog10p'] = -np.log10(agg['pvalue'].clip(lower=1e-300))
    if 'significant' in agg.columns:
        sig = agg['significant'].astype(float) > 0
    else:
        sig = (agg['logfc'].abs() >= fc_thresh) & (agg['pvalue'] <= p_thresh)
    up = sig & (agg['logfc'] > 0)
    down = sig & (agg['logfc'] < 0)
    ns = ~sig

    fig, ax = plt.subplots(figsize=(size_px / 100, size_px / 100), dpi=100)
    gid2gene = {}
    try:
        # rasterised grey background (one image inside the SVG, not thousands of nodes)
        ax.scatter(agg.loc[ns, 'logfc'], agg.loc[ns, 'nlog10p'], s=6, c=ns_color,
                   edgecolor='none', alpha=0.5, rasterized=True, zorder=1)
        i = 0
        for mask, color in [(down, down_color), (up, up_color)]:
            for _, r in agg.loc[mask].iterrows():
                gid = f'sig{i}'
                sc = ax.scatter([r['logfc']], [r['nlog10p']], s=14, c=color,
                                edgecolor='none', zorder=3)
                sc.set_gid(gid)
                gid2gene[gid] = str(r['genes'])
                i += 1
        ax.axhline(-np.log10(p_thresh), ls='--', lw=0.7, c='#888')
        ax.axvline(+fc_thresh, ls='--', lw=0.7, c='#888')
        ax.axvline(-fc_thresh, ls='--', lw=0.7, c='#888')
        tg = agg[agg['genes'] == gene]
        if not tg.empty:
            ax.scatter(tg['logfc'], tg['nlog10p'], s=70, facecolor='none',
                       edgecolor='black', lw=1.5, zorder=5)
            ax.annotate(gene, xy=(tg['logfc'].iat[0], tg['nlog10p'].iat[0]),
                        xytext=(8, 6), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', lw=0.7))
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel('logfc')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('')   # the panel labels the volcano in HTML
        buf = io.StringIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
    except Exception:
        plt.close(fig)
        return ''
    plt.close(fig)

    # inject <title>gene</title> into each significant point's <g id="sig*">
    try:
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        root = ET.fromstring(buf.getvalue())
        ns_uri = '{http://www.w3.org/2000/svg}'
        for el in root.iter():
            gid = el.get('id')
            if gid in gid2gene:
                t = ET.SubElement(el, ns_uri + 'title')
                t.text = gid2gene[gid]
                el.insert(0, t)
        return ET.tostring(root, encoding='unicode')
    except Exception:
        return buf.getvalue()


def _volcano_render_worker(args):
    """Module-level worker used by `plot_target_3d` / `plot_3d_interface` when
    `n_jobs > 1`.

    At module level so loky/cloudpickle can serialise it by reference. Receives a
    small pre-sliced per-key DataFrame instead of the full source. Returns a
    base64 PNG, or — when ``significant`` is set — an interactive SVG string.
    """
    import io, base64
    import matplotlib
    matplotlib.use('Agg')  # headless backend in workers
    import matplotlib.pyplot as plt
    gene, compound, sub, size_px, xmin, xmax = args[:6]
    significant = args[6] if len(args) > 6 else False
    if significant:
        return _volcano_svg_string(sub, compound, gene, key='compound',
                                   sig_col='significant',
                                   xmin=xmin, xmax=xmax, size_px=size_px)
    fig, ax = plt.subplots(figsize=(size_px / 100, size_px / 100), dpi=100)
    try:
        plot_volcano(sub, compound, gene,
                     xmin=xmin, xmax=xmax, ax=ax, title='')
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG', bbox_inches='tight')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ''
    finally:
        plt.close(fig)


def _volcano_cache_fname(gene, key, xlim, size_px, ext='.svg'):
    """Canonical on-disk volcano cache filename for a (focal gene, volcano key) pair.

    Single source of truth for the name used by `plot_3d_interface`'s disk cache, so
    a separate re-render pass (e.g. `recompute_volcanoes`) writes files the interface
    later finds as cache hits. The key string is data-independent on purpose — the
    cache is keyed by identity + render params, NOT by the underlying p-values, so a
    data change (like flooring p-values) requires explicitly regenerating the file.
    """
    import hashlib
    s = f'{gene}|{key}|{xlim[0]}|{xlim[1]}|{size_px}|{ext}'
    return hashlib.md5(s.encode()).hexdigest()[:16] + ext


def recompute_volcanoes(volcano_source, pairs, volcano_dir, *,
                        volcano_key='uniquecontrast', significant=True,
                        xlim=(-8.0, 8.0), size_px=350, n_jobs=1):
    """
    Re-render a specific set of ``(focal_gene, key)`` volcanoes to ``volcano_dir``,
    overwriting the disk cache used by :func:`plot_3d_interface`. Filenames match
    :func:`_volcano_cache_fname`, so a subsequent interface build picks them up as
    cache hits (no full re-render).

    Use this to refresh volcanoes after changing the underlying data (the cache is
    NOT data-keyed, so a plain interface re-run would otherwise reuse stale images).

    :param DataFrame volcano_source: long table with ``volcano_key``, ``genes``,
        ``logfc``, ``pvalue`` (and ``significant`` when ``significant=True``).
    :param iterable pairs: iterable of ``(gene, key_value)`` to render. ``gene`` is
        the focal/ringed target; ``key_value`` selects the experiment's points.
    :param str volcano_dir: output directory (created if missing).
    :param xlim/size_px: MUST match the values passed to ``plot_3d_interface`` or the
        filenames won't line up with what the interface expects.
    :return dict: ``{'requested', 'written', 'skipped', 'dir'}``.
    """
    import os
    import base64
    pairs = [(str(g), k) for g, k in pairs]
    if not pairs:
        return {'requested': 0, 'written': 0, 'skipped': 0, 'dir': volcano_dir}
    os.makedirs(volcano_dir, exist_ok=True)
    sig = bool(significant) and ('significant' in volcano_source.columns)
    ext = '.svg' if sig else '.png'

    # Pre-slice the source per key once; rename the key column to 'compound' so the
    # module-level worker (which filters on 'compound') can be reused as-is. Drop any
    # pre-existing 'compound' first so the rename can't create a duplicate column
    # (the source often carries both 'uniquecontrast' and a separate 'compound').
    keys = sorted({k for _, k in pairs})
    cols = ['genes', 'logfc', 'pvalue'] + (['significant'] if sig else [])
    src = (volcano_source.drop(columns=['compound'], errors='ignore')
           .rename(columns={volcano_key: 'compound'})
           if volcano_key != 'compound' else volcano_source)
    filt = (src.loc[src['compound'].isin(keys), ['compound'] + cols]
            .dropna(subset=['genes', 'logfc', 'pvalue']))
    sub_cache = {c: g for c, g in filt.groupby('compound', sort=False)}
    empty = filt.iloc[0:0]

    def _args(g, k):
        return (g, k, sub_cache.get(k, empty), size_px, xlim[0], xlim[1], sig)

    if n_jobs == 1:
        contents = [_volcano_render_worker(_args(g, k))
                    for g, k in tqdm(pairs, desc='recompute volcanoes', unit='vol', mininterval=0.5)]
    else:
        import contextlib
        import joblib as _joblib
        from joblib import Parallel, delayed
        # bridge joblib's per-batch completion callback to a tqdm bar
        @contextlib.contextmanager
        def _tqdm_joblib(pbar):
            class _Cb(_joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *a, **kw):
                    pbar.update(n=self.batch_size)
                    return super().__call__(*a, **kw)
            prev = _joblib.parallel.BatchCompletionCallBack
            _joblib.parallel.BatchCompletionCallBack = _Cb
            try:
                yield pbar
            finally:
                _joblib.parallel.BatchCompletionCallBack = prev
                pbar.close()
        with _tqdm_joblib(tqdm(total=len(pairs), desc='recompute volcanoes', unit='vol', mininterval=0.5)):
            contents = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_volcano_render_worker)(_args(g, k)) for g, k in pairs)

    written = skipped = 0
    for (g, k), content in zip(pairs, contents):
        if not content:
            skipped += 1
            continue
        path = os.path.join(volcano_dir, _volcano_cache_fname(g, k, xlim, size_px, ext))
        if sig:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)
        else:
            with open(path, 'wb') as fh:
                fh.write(base64.b64decode(content))
        written += 1
    print(f'> recompute_volcanoes: {written:,} written, {skipped:,} empty/failed '
          f'(of {len(pairs):,} requested) -> {volcano_dir}')
    return {'requested': len(pairs), 'written': written, 'skipped': skipped,
            'dir': volcano_dir}


def floor_zero_pvalues_and_refresh_volcanoes(measure, volcano_dir, *,
                                             drop_plates=(), volcano_key='uniquecontrast',
                                             xlim=(-8.0, 8.0), size_px=350, n_jobs=1,
                                             floor_inplace=True):
    """
    Floor 0.0 p-values to the smallest non-zero p-value and refresh ONLY the cached
    volcanoes of experiments that had >=1 floored target.

    A p-value of 0.0 -> +inf under -log10 and is clipped at 1e-300 by the renderers
    (plots at y=300). Flooring zeros to ``pmin`` (smallest observed non-zero p) caps the
    y-axis at ``-log10(pmin)`` instead. The volcano disk cache is keyed by identity +
    render params (NOT data), so the affected images are overwritten explicitly via
    :func:`recompute_volcanoes`; :func:`plot_3d_interface` then picks them up as cache
    hits. This is a one-off cache-refresh utility: new experiments are rendered fresh
    (and already floored) by the interface, so only run it when an EXISTING cached
    experiment's p-values changed.

    Crash-safe: renders from a floored COPY first; ``measure`` is floored in place (when
    ``floor_inplace``) only AFTER the re-render succeeds, so a failed render leaves the
    zeros intact and the call stays re-runnable.

    :param DataFrame measure: per-(gene x experiment) table with ``pvalue``,
        ``significant``, ``logfc``, ``plate``, ``genes`` and ``volcano_key`` columns.
    :param str volcano_dir: cache directory whose images are overwritten — MUST match the
        ``plot_3d_interface`` ``volcano_dir``/``xlim``/``size_px`` or filenames won't align.
    :param drop_plates: noisy plates excluded before rendering (as in the interface build).
    :param bool floor_inplace: floor ``measure`` in place after a successful re-render.
    :return dict: ``{'n_floored', 'n_experiments', 'pmin', 'n_pairs', 'stats'}``;
        ``stats`` is ``None`` (and ``pmin`` is ``None``) when there were no zeros to floor.
    """
    was_zero = measure['pvalue'].eq(0.0)
    if not was_zero.any():
        return {'n_floored': 0, 'n_experiments': 0, 'pmin': None, 'n_pairs': 0, 'stats': None}
    pmin = measure.loc[measure['pvalue'] > 0, 'pvalue'].min()
    assert pd.notna(pmin), 'no non-zero p-values to floor to'
    affected = set(measure.loc[was_zero, volcano_key].unique())

    # render from a FLOORED COPY (don't touch `measure` yet); same plate-drop + significant-
    # down-hit rule the interface uses, restricted to the experiments that had a floored target.
    meas = measure[~measure['plate'].isin(list(drop_plates))].copy()
    meas.loc[meas['pvalue'].eq(0.0), 'pvalue'] = pmin
    pairs = (meas[(meas['significant'] == 1) & (meas['logfc'] < 0)
                  & meas[volcano_key].isin(affected)]
             [['genes', volcano_key]].dropna().drop_duplicates())
    stats = recompute_volcanoes(
        meas, pairs.itertuples(index=False, name=None), volcano_dir,
        volcano_key=volcano_key, significant=True, xlim=xlim, size_px=size_px, n_jobs=n_jobs)

    if floor_inplace:   # re-render succeeded -> match `measure` to the regenerated images
        measure.loc[was_zero, 'pvalue'] = pmin
    return {'n_floored': int(was_zero.sum()), 'n_experiments': len(affected),
            'pmin': float(pmin), 'n_pairs': len(pairs), 'stats': stats}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global plate-quality scan + drop validation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assess_plates_globally(
    df_raw, MF_features, genes,
    *,
    label_col='logfc_corrected',
    plate_col='MSPlate',
    rf_params=None,
    min_train=20,
    min_test=5,
    n_rf_jobs=8,
    seed=0,
    drop_frac_neg=0.5,
    drop_median_r2=0.0,
    verbose=True,
):
    """
    Per-gene leave-one-plate-out CV across many genes, aggregated to a single
    drop recommendation that should help the majority of genes.

    For every (gene, plate) pair, train RF on every compound's mean label across
    plates ≠ P and predict its plate-P measurement. The resulting (gene × plate)
    R² matrix is then aggregated per plate.

    A plate is recommended for drop when BOTH:
      * fraction of genes with LOPO R² < 0 exceeds ``drop_frac_neg`` (default 0.5)
      * median R² across genes is below ``drop_median_r2`` (default 0.0)

    :return dict: {'lopo_matrix', 'plate_scores', 'recommended_drop'}.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    if rf_params is None:
        rf_params = {'n_estimators': 100, 'max_depth': 20,
                     'max_features': 'sqrt', 'min_samples_leaf': 1}

    df = df_raw.dropna(subset=[plate_col]).copy()
    if label_col not in df.columns:
        raise ValueError(f'label_col {label_col!r} not in df_raw')

    feat_cols = [c for c in MF_features.columns if c != 'compound']
    rows = []
    for gene in tqdm(genes, desc='LOPO matrix', disable=not verbose):
        sub = df[df['genes'] == gene]
        if sub.empty:
            continue
        # collapse intra-plate replicates
        cp = sub.groupby(['compound', plate_col])[label_col].mean().reset_index()
        plates_here = cp[plate_col].unique()
        for P in plates_here:
            tr = (cp[cp[plate_col] != P]
                  .groupby('compound')[label_col].mean()
                  .reset_index().rename(columns={label_col: 'label'}))
            te = (cp[cp[plate_col] == P][['compound', label_col]]
                  .rename(columns={label_col: 'label'}))
            tr_xy = pd.merge(MF_features, tr, on='compound').dropna()
            te_xy = pd.merge(MF_features, te, on='compound').dropna()
            if len(tr_xy) < min_train or len(te_xy) < min_test:
                continue
            try:
                rf = RandomForestRegressor(**rf_params, n_jobs=n_rf_jobs,
                                            random_state=seed)
                rf.fit(tr_xy[feat_cols], tr_xy['label'])
                yhat = rf.predict(te_xy[feat_cols])
                yte  = te_xy['label'].values
                r2 = r2_score(yte, yhat) if len(yte) >= 2 else float('nan')
            except Exception:
                r2 = float('nan')
            rows.append({'gene': gene, 'plate': P, 'r2': r2,
                         'n_train': len(tr_xy), 'n_test': len(te_xy)})

    lopo_long = pd.DataFrame(rows)
    if lopo_long.empty:
        if verbose:
            print('> no (gene, plate) pairs survived the train/test minima')
        return {'lopo_matrix': pd.DataFrame(),
                'plate_scores': pd.DataFrame(),
                'recommended_drop': []}

    lopo_matrix = lopo_long.pivot(index='gene', columns='plate', values='r2')
    n_eval = lopo_matrix.notna().sum(axis=0)
    plate_scores = pd.DataFrame({
        'n_genes_evaluated':      n_eval,
        'frac_genes_negative_r2': (lopo_matrix < 0).sum(axis=0) / n_eval.replace(0, np.nan),
        'median_r2':              lopo_matrix.median(axis=0),
        'mean_r2_clipped':        lopo_matrix.clip(lower=-1, upper=1).mean(axis=0),
    }).sort_values('median_r2', ascending=True)

    drop_mask = ((plate_scores['frac_genes_negative_r2'] > drop_frac_neg) &
                 (plate_scores['median_r2'] < drop_median_r2))
    recommended_drop = plate_scores.index[drop_mask].tolist()

    if verbose:
        print(f'> evaluated {lopo_matrix.shape[0]} genes × '
              f'{lopo_matrix.shape[1]} plates  '
              f'({(~lopo_matrix.isna()).sum().sum():,} (gene, plate) cells)')
        print(f'> recommended drop ({len(recommended_drop)} plates): {recommended_drop}')

    return {'lopo_matrix':      lopo_matrix,
            'plate_scores':     plate_scores,
            'recommended_drop': recommended_drop}


def validate_plate_drop(
    df_raw, MF_features, genes, drop_plates,
    *,
    label_col='logfc_corrected',
    plate_col='MSPlate',
    rf_params=None,
    n_rf_jobs=8,
    seed=0,
    verbose=True,
    ML_Reg_module=None,
):
    """
    For each gene, compare 5-fold CV R² on the full data vs after dropping
    ``drop_plates``. Returns a per-gene table with the delta + sample sizes.

    The CV harness (``ML_Reg_module.run_K_Fold_Xval_Regression``) must be passed
    explicitly so this function isn't tied to a specific path layout.
    """
    from sklearn.ensemble import RandomForestRegressor
    if rf_params is None:
        rf_params = {'n_estimators': 100, 'max_depth': 20,
                     'max_features': 'sqrt', 'min_samples_leaf': 1}
    if ML_Reg_module is None:
        raise ValueError('pass the ML_Reg module so we use the same CV harness as the notebook')

    rows = []
    for gene in tqdm(genes, desc='validate drop', disable=not verbose):
        full = df_raw[df_raw['genes'] == gene]
        if full.empty:
            continue
        kept = full[~full[plate_col].isin(drop_plates)]
        for cond_name, src in [('keep_all', full), ('drop', kept)]:
            ml = (src.groupby('compound')[label_col].mean()
                     .reset_index().rename(columns={label_col: 'label'}))
            ml = pd.merge(MF_features, ml, on='compound').dropna()
            if len(ml) < 10:
                rows.append({'gene': gene, 'condition': cond_name,
                             'n': len(ml), 'r2': float('nan')})
                continue
            try:
                rf = RandomForestRegressor(**rf_params, n_jobs=n_rf_jobs,
                                            random_state=seed)
                _, df_pred = ML_Reg_module.run_K_Fold_Xval_Regression(
                    ml, model=rf, col_to_rm=['compound', 'label'], ID='compound',
                    get_ints=False, v=False, to_impute=None, rm_empty_cols=False,
                )
                r2 = ML_Reg_module.get_reg_metrics_from_preddf(df_pred, v=False)['r2']
            except Exception:
                r2 = float('nan')
            rows.append({'gene': gene, 'condition': cond_name,
                         'n': len(ml), 'r2': r2})

    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame()
    r2_w = long.pivot(index='gene', columns='condition', values='r2')
    n_w  = long.pivot(index='gene', columns='condition', values='n').rename(
        columns={'keep_all': 'n_keep', 'drop': 'n_drop'})
    out = r2_w.join(n_w)
    out['delta'] = out['drop'] - out['keep_all']
    out = out.sort_values('delta', ascending=True)

    if verbose:
        d = out['delta'].dropna()
        print(f'> mean   Δ R²: {d.mean():+.4f}')
        print(f'> median Δ R²: {d.median():+.4f}')
        print(f'> genes that improve (Δ > 0): {(d > 0).sum()} / {len(d)}')
        print(f'> genes that worsen  (Δ < 0): {(d < 0).sum()} / {len(d)}')

    return out


def cumulative_plate_ablation(
    df_raw, MF_features, genes, drop_order,
    *,
    label_col='logfc_corrected',
    plate_col='MSPlate',
    rf_params=None,
    n_rf_jobs=8,
    seed=0,
    verbose=True,
    ML_Reg_module=None,
):
    """
    For k = 0, 1, …, len(drop_order), drop the first ``k`` plates from
    ``drop_order`` and run 5-fold CV R² per gene. Returns a long-format
    DataFrame with one row per (k, gene): {k, gene, r2, n_compounds, delta,
    plate_dropped_at_this_k}.

    Δ is computed against each gene's k=0 baseline so it tracks the marginal
    impact of cumulatively dropping plates in the supplied order — useful for
    finding the sweet spot before R² plateaus or declines.
    """
    from sklearn.ensemble import RandomForestRegressor
    if rf_params is None:
        rf_params = {'n_estimators': 100, 'max_depth': 20,
                     'max_features': 'sqrt', 'min_samples_leaf': 1}
    if ML_Reg_module is None:
        raise ValueError('pass ML_Reg_module so we use the same CV harness as the notebook')

    def _cv_r2(sub):
        ml = (sub.groupby('compound')[label_col].mean()
                 .reset_index().rename(columns={label_col: 'label'}))
        ml = pd.merge(MF_features, ml, on='compound').dropna()
        if len(ml) < 10:
            return float('nan'), len(ml)
        try:
            rf = RandomForestRegressor(**rf_params, n_jobs=n_rf_jobs, random_state=seed)
            _, df_pred = ML_Reg_module.run_K_Fold_Xval_Regression(
                ml, model=rf, col_to_rm=['compound', 'label'], ID='compound',
                get_ints=False, v=False, to_impute=None, rm_empty_cols=False,
            )
            return ML_Reg_module.get_reg_metrics_from_preddf(df_pred, v=False)['r2'], len(ml)
        except Exception:
            return float('nan'), len(ml)

    rows = []
    for k in tqdm(range(0, len(drop_order) + 1), desc='cumulative drop k',
                  disable=not verbose):
        drop_set = set(drop_order[:k])
        for g in tqdm(genes, desc=f'k={k}', leave=False, disable=not verbose):
            sub = df_raw[(df_raw['genes'] == g) & ~df_raw[plate_col].isin(drop_set)]
            r2, n = _cv_r2(sub)
            rows.append({
                'k': k, 'gene': g, 'r2': r2, 'n_compounds': n,
                'plate_dropped_at_this_k': drop_order[k - 1] if k > 0 else None,
            })

    df = pd.DataFrame(rows)
    baseline = df.loc[df['k'] == 0].set_index('gene')['r2']
    df['delta'] = df['r2'] - df['gene'].map(baseline)
    return df


# --- DEPRECATED 2026-05-19: superseded by compute_R2_for_all_genes.compute_gene_R2 (single source of truth). Commented out pending confirmation of the new path; remove after verifying. ---
# def compute_gene_sar_r2(
#     gene, df_raw, features,
#     *,
#     label_col='logfc',
#     model_class=None,
#     model_params=None,
#     min_compounds=100,
#     n_null=0,
#     n_jobs=8,
#     seed=0,
#     ML_Reg_module=None,
#     verbose=False,
# ):
#     """
#     5-fold cross-validated SAR predictability for one gene.
#
#     Filters ``df_raw`` to the gene, aggregates ``label_col`` per compound (mean
#     across replicates), merges with ``features`` on ``compound``, and runs the
#     project's K-fold CV harness to get an R². Optionally repeats with shuffled
#     labels ``n_null`` times to estimate the mean of the null distribution.
#
#     The returned dict matches the SAR-screen CSV header verbatim, so a caller
#     can do ``writer.writerow(result)`` with no transformation. Skipped genes
#     (``n <= min_compounds``) return NaN R²/nullR² with the actual compound
#     count, so the caller's resume-set still includes them and they don't get
#     retried on the next pass.
#
#     :param str gene: gene symbol to filter ``df_raw['genes']`` on
#     :param df df_raw: must have ``genes``, ``compound``, and ``label_col``
#     :param df features: molecular features keyed by ``compound``
#     :param str label_col: which column to predict (e.g. ``'logfc'`` or ``'logfc_corrected'``)
#     :param type model_class: e.g. ``RandomForestRegressor``; instantiated fresh per call
#     :param dict model_params: kwargs for the model constructor
#     :param int min_compounds: skip if compounds-after-merge ≤ this
#     :param int n_null: label-shuffle permutations for null R²; 0 = skip
#     :param int n_jobs: passed as ``n_jobs`` to the model
#     :param int seed: passed as ``random_state`` to the model
#     :param module ML_Reg_module: project's CV harness, passed in to avoid hard imports
#     :return dict: ``{'gene', 'R2', 'nullR2', 'n'}``
#     """
#     if model_class is None:
#         raise ValueError('pass model_class (e.g. RandomForestRegressor)')
#     if ML_Reg_module is None:
#         raise ValueError('pass ML_Reg_module so we use the same CV harness as the notebook')
#     if model_params is None:
#         model_params = {}
#
#     sub = df_raw[df_raw['genes'] == gene]
#     if sub.empty:
#         return {'gene': gene, 'R2': float('nan'), 'nullR2': float('nan'), 'n': 0}
#
#     agg = (sub.groupby('compound')[label_col].mean()
#               .reset_index()
#               .dropna(subset=[label_col])
#               .rename(columns={label_col: 'label'}))
#
#     ML_data = pd.merge(features, agg, on='compound').dropna()
#     n = len(ML_data)
#
#     if n <= min_compounds:
#         if verbose:
#             print(f'  [skip] {gene}: only {n} compounds (min_compounds={min_compounds})')
#         return {'gene': gene, 'R2': float('nan'), 'nullR2': float('nan'), 'n': n}
#
#     def _new_model():
#         # fresh instance per call so RF/XGB internal state never leaks between fits
#         return model_class(**{**model_params, 'n_jobs': n_jobs, 'random_state': seed})
#
#     _, df_pred = ML_Reg_module.run_K_Fold_Xval_Regression(
#         ML_data, model=_new_model(),
#         col_to_rm=['compound', 'label'], ID='compound',
#         get_ints=False, v=False, to_impute=None, rm_empty_cols=False,
#     )
#     R2 = ML_Reg_module.get_reg_metrics_from_preddf(df_pred, v=False)['r2']
#
#     null_R2 = float('nan')
#     if n_null > 0:
#         rng = np.random.default_rng(seed)
#         nulls = []
#         for _ in range(n_null):
#             shuffled = ML_data.copy()
#             shuffled['label'] = rng.permutation(shuffled['label'].values)
#             _, df_pred_null = ML_Reg_module.run_K_Fold_Xval_Regression(
#                 shuffled, model=_new_model(),
#                 col_to_rm=['compound', 'label'], ID='compound',
#                 get_ints=False, v=False, to_impute=None, rm_empty_cols=False,
#             )
#             nulls.append(ML_Reg_module.get_reg_metrics_from_preddf(df_pred_null, v=False)['r2'])
#         null_R2 = float(np.mean(nulls))
#
#     if verbose:
#         print(f'  {gene}: R²={R2:.3f}  null={null_R2:.3f}  n={n}')
#
#     return {'gene': gene, 'R2': float(R2), 'nullR2': null_R2, 'n': n}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Autoresearch progress plot (Karpathy-style)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_autoresearch_progress(
    jsonl_path,
    *,
    metric_name=None,
    higher_is_better=True,
    title=None,
    annotate_kept=True,
    annotate_max_chars=40,
    figsize=(14, 7),
    save_path=None,
    ax=None,
):
    """
    Karpathy-style autotune-progress plot for an autoresearch run.

    X-axis = experiment index (0..N as they appear in the JSONL log).
    Y-axis = the run's optimisation metric.
    Light-grey dots = discarded experiments (didn't beat the running best).
    Green dots      = kept improvements (new champion at that index).
    Green line      = running best.
    Optional rotated text labels per kept improvement showing its ``desc``.

    :param str/Path jsonl_path: path to autoresearch.jsonl (one rec per line).
    :param str metric_name: which key to plot on Y (e.g. ``'mean_r2'``,
        ``'pr_auc'``). Defaults to each rec's ``metric_name`` field, or the
        most common ``metric_name`` across the log if absent.
    :param bool higher_is_better: True for accuracy-style metrics, False for
        losses (validation BPB, RMSE).
    :param str title: figure title; defaults to a one-liner with N + N_kept.
    :param bool annotate_kept: if True, rotate the kept point's ``desc`` next
        to it. Set False for very long runs.
    :param int annotate_max_chars: truncate long descs to this many chars.
    :return: ``(fig, ax)``.
    """
    import json
    import matplotlib.pyplot as plt
    from collections import Counter
    from pathlib import Path

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f'no autoresearch log at {jsonl_path}')

    recs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not recs:
        raise ValueError(f'autoresearch log at {jsonl_path} is empty')

    # auto-pick metric_name if not supplied
    if metric_name is None:
        names = [r.get('metric_name') for r in recs if r.get('metric_name')]
        if not names:
            raise ValueError(
                'no metric_name in any rec; pass metric_name= explicitly')
        metric_name = Counter(names).most_common(1)[0][0]

    metrics = [r.get(metric_name) for r in recs]
    # running best computed afresh — robust to missing _kept_as_best flags.
    running_best = []
    kept_idx = []
    best = -float('inf') if higher_is_better else float('inf')
    is_better = (lambda x, b: x is not None and np.isfinite(x) and x > b) \
                if higher_is_better else \
                (lambda x, b: x is not None and np.isfinite(x) and x < b)
    for i, m in enumerate(metrics):
        if is_better(m, best):
            best = m
            kept_idx.append(i)
        running_best.append(best if np.isfinite(best) else float('nan'))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xs = np.arange(len(recs))
    valid_mask = np.array([m is not None and np.isfinite(m) for m in metrics])
    discarded_mask = valid_mask.copy()
    discarded_mask[kept_idx] = False

    ax.scatter(xs[discarded_mask],
               np.asarray(metrics, dtype=float)[discarded_mask],
               s=18, color='#cccccc', alpha=0.6, edgecolor='none',
               label='Discarded', zorder=1)
    ax.scatter(xs[kept_idx],
               np.asarray(metrics, dtype=float)[kept_idx],
               s=70, color='#2ca870', edgecolor='#1d6a45', linewidth=1.0,
               label='Kept', zorder=3)
    ax.plot(xs, running_best, color='#2ca870', linewidth=1.5, alpha=0.85,
            label='Running best', zorder=2)

    if annotate_kept:
        for i in kept_idx:
            desc = (recs[i].get('desc') or recs[i].get('id') or '')
            if len(desc) > annotate_max_chars:
                desc = desc[:annotate_max_chars - 1] + '…'
            ax.annotate(
                desc, xy=(i, metrics[i]),
                xytext=(4, 4), textcoords='offset points',
                rotation=45, ha='left', va='bottom',
                fontsize=7, color='#1d6a45', alpha=0.9, zorder=4,
            )

    ax.set_xlabel('Experiment #')
    direction = '(higher is better)' if higher_is_better else '(lower is better)'
    ax.set_ylabel(f'{metric_name}  {direction}')
    if title is None:
        title = (f'Autoresearch progress: {len(recs)} experiments, '
                 f'{len(kept_idx)} kept improvements')
    ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


# ── Single high-level cell-state label per compound from its Hallmark GSEA ──
# Hallmark -> MSigDB process category: the CANONICAL 8 categories from Liberzon et al. 2015
# (Cell Systems, Table 1). Citable taxonomy; EDIT here to retune. The wiki documents the prior
# phenotype-oriented 9-theme variant (exact regroupings) to revert to if readability is preferred.
HALLMARK_THEMES = {
    # proliferation  (P53_PATHWAY is canonically HERE, not DNA damage)
    'HALLMARK_E2F_TARGETS': 'Proliferation', 'HALLMARK_G2M_CHECKPOINT': 'Proliferation',
    'HALLMARK_MYC_TARGETS_V1': 'Proliferation', 'HALLMARK_MYC_TARGETS_V2': 'Proliferation',
    'HALLMARK_MITOTIC_SPINDLE': 'Proliferation', 'HALLMARK_P53_PATHWAY': 'Proliferation',
    # DNA damage
    'HALLMARK_DNA_REPAIR': 'DNA damage', 'HALLMARK_UV_RESPONSE_UP': 'DNA damage',
    'HALLMARK_UV_RESPONSE_DN': 'DNA damage',
    # metabolic
    'HALLMARK_OXIDATIVE_PHOSPHORYLATION': 'Metabolic', 'HALLMARK_GLYCOLYSIS': 'Metabolic',
    'HALLMARK_FATTY_ACID_METABOLISM': 'Metabolic', 'HALLMARK_CHOLESTEROL_HOMEOSTASIS': 'Metabolic',
    'HALLMARK_BILE_ACID_METABOLISM': 'Metabolic', 'HALLMARK_XENOBIOTIC_METABOLISM': 'Metabolic',
    'HALLMARK_HEME_METABOLISM': 'Metabolic',
    # immune  (IL6_JAK_STAT3 is immune; IL2_STAT5 / TNFA are signaling, below)
    'HALLMARK_ALLOGRAFT_REJECTION': 'Immune', 'HALLMARK_COAGULATION': 'Immune',
    'HALLMARK_COMPLEMENT': 'Immune', 'HALLMARK_INTERFERON_ALPHA_RESPONSE': 'Immune',
    'HALLMARK_INTERFERON_GAMMA_RESPONSE': 'Immune', 'HALLMARK_IL6_JAK_STAT3_SIGNALING': 'Immune',
    'HALLMARK_INFLAMMATORY_RESPONSE': 'Immune',
    # signaling  (TNFA_SIGNALING_VIA_NFKB, IL2_STAT5_SIGNALING, MTORC1_SIGNALING are canonically HERE)
    'HALLMARK_ANDROGEN_RESPONSE': 'Signaling', 'HALLMARK_ESTROGEN_RESPONSE_EARLY': 'Signaling',
    'HALLMARK_ESTROGEN_RESPONSE_LATE': 'Signaling', 'HALLMARK_IL2_STAT5_SIGNALING': 'Signaling',
    'HALLMARK_KRAS_SIGNALING_UP': 'Signaling', 'HALLMARK_KRAS_SIGNALING_DN': 'Signaling',
    'HALLMARK_MTORC1_SIGNALING': 'Signaling', 'HALLMARK_NOTCH_SIGNALING': 'Signaling',
    'HALLMARK_PI3K_AKT_MTOR_SIGNALING': 'Signaling', 'HALLMARK_HEDGEHOG_SIGNALING': 'Signaling',
    'HALLMARK_TGF_BETA_SIGNALING': 'Signaling', 'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'Signaling',
    'HALLMARK_WNT_BETA_CATENIN_SIGNALING': 'Signaling',
    # pathway  (the catch-all: apoptosis / hypoxia / secretion / UPR / ROS)
    'HALLMARK_APOPTOSIS': 'Pathway', 'HALLMARK_HYPOXIA': 'Pathway',
    'HALLMARK_PROTEIN_SECRETION': 'Pathway', 'HALLMARK_UNFOLDED_PROTEIN_RESPONSE': 'Pathway',
    'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY': 'Pathway',
    # cellular component
    'HALLMARK_APICAL_JUNCTION': 'Cellular component', 'HALLMARK_APICAL_SURFACE': 'Cellular component',
    'HALLMARK_PEROXISOME': 'Cellular component',
    # development  (ADIPOGENESIS, ANGIOGENESIS, EMT are canonically HERE)
    'HALLMARK_ADIPOGENESIS': 'Development', 'HALLMARK_ANGIOGENESIS': 'Development',
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': 'Development', 'HALLMARK_MYOGENESIS': 'Development',
    'HALLMARK_SPERMATOGENESIS': 'Development', 'HALLMARK_PANCREAS_BETA_CELLS': 'Development',
}

# readable signed names for the phenotype-clear categories; others fall back to "Category ↑/↓".
# (The canonical "Pathway"/"Cellular component" buckets are heterogeneous, hence no special name.)
def label_signatures_hallmark(msigdb_sig, *, theme_map=HALLMARK_THEMES, min_abs_nes=1.0, top_k=3,
                              compound_col='compound', term_col='term_name',
                              nes_col='NES', collection_col='collection'):
    """Assign ONE high-level signed cell-state label per compound from its Hallmark GSEA.

    Each Hallmark set is mapped to its MSigDB process category (``theme_map`` — the canonical
    8 categories from Liberzon et al. 2015, Table 1); a compound's
    per-theme score is the signed mean NES of the theme's ``top_k`` strongest (by |NES|) sets
    — robust to diluting a coherent subset across a large heterogeneous theme, and a mixed-
    direction theme cancels out (coordination-aware, unlike a single-term argmax). The label is
    the theme with the largest ``|score|``, named purely descriptively as ``"theme ↑/↓"``
    (no interpretive mapping); themes weaker than ``min_abs_nes`` yield ``"Unclassified"``.

    :param df msigdb_sig: long GSEA table (one row per compound × term) with NES + collection.
    :param dict theme_map: Hallmark term_name -> theme. :param float min_abs_nes: label floor.
    :return df: one row per compound — ``signature_label``, ``top_theme``/``top_nes``,
        ``second_theme``/``second_nes`` (runner-up for transparency), ``proliferation_nes``
        (always shown — the key axis for anti-proliferative screens).
    """
    # theme score = signed mean of the top_k sets by |NES| (coherent sub-signal, not diluted over all
    # members); shared with hallmark_theme_matrix. stack() drops absent (compound, theme) cells.
    theme_nes = (hallmark_theme_matrix(msigdb_sig, theme_map=theme_map, top_k=top_k,
                                       compound_col=compound_col, term_col=term_col,
                                       nes_col=nes_col, collection_col=collection_col)
                 .stack().rename('nes').reset_index())
    theme_nes['rank'] = theme_nes.groupby(compound_col)['nes'].transform(
        lambda s: s.abs().rank(method='first', ascending=False))

    top    = theme_nes[theme_nes['rank'] == 1].set_index(compound_col)
    second = theme_nes[theme_nes['rank'] == 2].set_index(compound_col)
    prolif = (theme_nes[theme_nes['theme'] == 'Proliferation']
              .set_index(compound_col)['nes'])

    def _name(theme, nes):
        # purely descriptive: the dominant theme + its direction, no interpretive mapping.
        if abs(nes) < min_abs_nes:
            return 'Unclassified'
        return f'{theme} {"↑" if nes > 0 else "↓"}'

    out = pd.DataFrame(index=top.index)
    out['signature_label']  = [_name(t, n) for t, n in zip(top['theme'], top['nes'])]
    out['top_theme']        = top['theme']
    out['top_nes']          = top['nes'].round(2)
    out['second_theme']     = second['theme']
    out['second_nes']       = second['nes'].round(2)
    out['proliferation_nes'] = prolif.round(2)
    return out.reset_index().rename(columns={'index': compound_col})


def hallmark_theme_matrix(msigdb_sig, *, theme_map=HALLMARK_THEMES, top_k=3,
                          compound_col='compound', term_col='term_name',
                          nes_col='NES', collection_col='collection'):
    """Per-compound × Hallmark-theme signed-NES matrix — the signature-space features.

    Each Hallmark set maps to its MSigDB process category (``theme_map``); a compound's
    theme score is the signed mean NES of that category's ``top_k`` strongest (by |NES|)
    sets — the same coordination-aware score `label_signatures_hallmark` reduces to a single
    label, exposed here as an 8-dim vector for clustering / embedding. Absent (compound,
    theme) cells are left NaN (a compound with no sets in a category); fill with 0.0 for a
    dense feature matrix.

    :param df msigdb_sig: long GSEA table (one row per compound × term) with NES + collection.
    :return df: compound (index) × theme (columns, sorted) signed-NES matrix.
    """
    hall = msigdb_sig[msigdb_sig[collection_col] == 'Hallmark'].copy()
    hall['theme'] = hall[term_col].map(theme_map)
    hall = hall.dropna(subset=['theme'])
    mat = (hall.groupby([compound_col, 'theme'])[nes_col]
           .apply(lambda s: s.reindex(s.abs().sort_values(ascending=False).index).head(top_k).mean())
           .unstack('theme'))
    return mat.reindex(columns=sorted(set(theme_map.values())))
