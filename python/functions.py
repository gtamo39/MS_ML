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
        colors=('#d8cdbf', '#c9b79a', '#88a06a', '#d99a3a', '#b8412f'),
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
    :param colors: per-category fill colours (same length/order as ``cats``).
    :param str silent_label: inactive label (used for the activity-rate line).
    :param bool show_rate_line: overlay the non-silent activity-rate line.
    :param int dpi: figure resolution (only used when ``ax`` is None; default 150).
    :param ax: optional matplotlib axes.
    :return: ``(ax, summary)`` — axes and a per-tranche DataFrame indexed by date
        with an ``n`` column and one share column per category.
    """
    import matplotlib.pyplot as plt
    cats = list(cats); colors = list(colors)

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
                  font: 11px sans-serif; color: #333; max-height: 80vh;
                  overflow-y: auto; display: none; user-select: none; }
  #filter-panel .fp-group + .fp-group { margin-top: 8px; }
  #filter-panel .pf-head { font-weight: 700; padding-bottom: 4px;
                           border-bottom: 1px solid #eee; margin-bottom: 4px; }
  #filter-panel .pf-head span { color: #1D3557; cursor: pointer; font-weight: 400;
                                font-size: 10px; }
  #filter-panel .pf-head span:hover { text-decoration: underline; }
  #filter-panel label { display: block; padding: 1px 0; cursor: pointer; white-space: nowrap; }
  #filter-panel input { margin-right: 5px; vertical-align: middle; }
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
  <div class="fp-group" id="plate-group">
    <div class="pf-head">Plates <span id="pf-all">all</span> / <span id="pf-none">none</span></div>
    <div id="pf-boxes"></div>
  </div>
  <div class="fp-group" id="activity-group">
    <div class="pf-head">Activity <span id="af-all">all</span> / <span id="af-none">none</span></div>
    <div id="af-boxes"></div>
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
    var patents = window.__GENE_PATENTS__ || {};
    var depmapTpl = window.__DEPMAP_URL__ || "https://depmap.org/portal/gene/{gene}";
    var pageSize = window.__PAGE_SIZE__ || 5;
    var axis = window.__AXIS_LABELS__ || {x: "X", y: "Y", z: "Z"};
    var axisHelp = window.__AXIS_HELP__ || {};
    var plates = window.__PLATES__ || [];
    var ticked = {};
    plates.forEach(function(p) { ticked[p] = true; });
    var activities = window.__ACTIVITIES__ || [];
    var tickedAct = {};
    activities.forEach(function(a) { tickedAct[a] = true; });
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

    var pinned = false;
    var currentGene = "";
    var fullArr = [];
    var entries = [];      // [{t: row, idx: absolute index in fullArr}] minus __META__
    var page = 0;
    var volPinIdx = null;  // data-eidx of the compound whose volcano(s) are click-pinned
    var recolor3d = function() {};  // set by the slider block; re-applies gene colouring

    // A gene is "active" under the current Plate + Activity ticks if it has at
    // least one compound whose plate AND activity are both ticked. Used to grey
    // out genes that have no compound on the selected plates/activities.
    function geneHasVisibleCompound(gene) {
      var arr = (window.__GENE_COMPOUNDS__ || {})[gene];
      if (!arr) return false;
      for (var i = 0; i < arr.length; i++) {
        var t = arr[i];
        if (!t || t[0] === "__META__") continue;
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
        var img = t[1]
          ? '<img src="data:image/png;base64,' + t[1] + '" draggable="false"/>'
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
    function vimg(v) {
      // 'svg' -> interactive <object> (native <title> tooltips on significant
      // points); 'path' -> external PNG <img>; 'b64' -> inline PNG <img>.
      if (VMODE === "svg") {
        return '<object class="vobj" type="image/svg+xml" data="' + v + '"></object>';
      }
      var src = (VMODE === "path") ? v : ("data:image/png;base64," + v);
      return '<img loading="lazy" src="' + src + '"/>';
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
          html += '<div class="vlabel">' + currentGene + ' · ' + cmp + ' · '
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
      if (!pinned) return;
      var cell = e.target.closest(".cell");
      if (!cell) return;
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
      if (render(e.points && e.points[0])) {
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
        html += '<label><input type="checkbox" value="' + v + '" checked>' + v + '</label>';
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
      document.getElementById(allId).addEventListener("click", function() { setAll(true); });
      document.getElementById(noneId).addEventListener("click", function() { setAll(false); });
      return true;
    }
    var hasPlateG = buildGroup(plates, ticked, pfBoxes, "pf-all", "pf-none");
    var hasActG   = buildGroup(activities, tickedAct, afBoxes, "af-all", "af-none");
    if (hasPlateG || hasActG) pf.style.display = "block";

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
                   && geneHasVisibleCompound(o.text[k]);
            m.push(inr); if (inr) total++;
          }
          masks[ti] = m;
        });
        // Mutate the trace data directly, then force a full redraw. Plotly.restyle
        // of x/y/z on a gl3d (WebGL) scatter3d updates the data but does NOT
        // reliably repaint the 3D scene — Plotly.redraw() does. In-range genes
        // always keep their gene-name label.
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
        Plotly.redraw(gd);
        document.getElementById("rp-count").textContent = total + " in range";
      }
      recolor3d = applyRanges;   // let the Plate/Activity checkboxes re-colour too
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
        fig.write_html(html_path, include_plotlyjs='cdn')

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
    volcano_source=None,
    volcano_key='uniquecontrast',
    page_size=5,
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
    title='SAR predictability × disease association × MS score',
    range_sliders=False,
    range_defaults=None,
    control_genes=(),
    gene_research=None,
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
    if have_compounds:
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
                _vsrc = (volcano_source.rename(columns={volcano_key: 'compound'})
                         if volcano_key != 'compound' else volcano_source)
            cdf = compounds_df[compounds_df['gene'].isin(set(highlighted['gene']))].copy()
            if plate_aware:
                all_plates = sorted(compounds_df['plate'].dropna().astype(str).unique())
                # Optional per-experiment activity level (nr_down bucket). Ordered
                # high→silent for the checkbox panel; only levels present are kept.
                has_act = 'activity' in compounds_df.columns
                has_ng = 'n_genes' in compounds_df.columns   # genes measured in the experiment
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
                for gene, gdf in cdf.groupby('gene', sort=False):
                    entries = [['__META__', '', _meta_str(gene), '', '', '']]
                    for compound, cg in gdf.groupby('compound', sort=False):
                        lab = str(compound)
                        smi = (cg['smiles'].dropna().iloc[0]
                               if cg['smiles'].notna().any() else None)
                        ei = len(entries)
                        plate_rows = []   # [plate, logfc, volcano, activity, n_genes] per plate
                        for pi, (_, pr) in enumerate(cg.iterrows()):
                            plate_rows.append([
                                str(pr['plate']),
                                f"{pr['logfc']:.2f}" if pd.notna(pr['logfc']) else '',
                                '',   # volcano b64/path filled in the render pass below
                                (str(pr['activity']) if has_act and pd.notna(pr['activity'])
                                 else ''),
                                (str(int(pr['n_genes'])) if has_ng and pd.notna(pr['n_genes'])
                                 else ''),
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
                for gene, grp in cdf.groupby('gene', sort=False):
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

                def _vfname(g, vk):
                    key = f'{g}|{vk}|{volcano_xlim[0]}|{volcano_xlim[1]}|{volcano_size_px}|{_ext}'
                    return hashlib.md5(key.encode()).hexdigest()[:16] + _ext

                # cache hits: file already on disk -> reference it, skip render
                render = []
                for (g, vk, ei, pi) in tasks:
                    fn_ = _vfname(g, vk)
                    if os.path.exists(os.path.join(volcano_dir, fn_)):
                        _set_volcano(g, ei, pi, _rel + '/' + fn_)
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
                    _set_volcano(g, ei, pi, _rel + '/' + fn_)
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
    fig.add_trace(go.Scatter3d(
        x=plot_df[x_col], y=plot_df[y_col], z=plot_df['_zplot'],
        mode='markers',
        marker=dict(size=3, color='lightgrey', opacity=0.5, line=dict(width=0)),
        name=f'all ({len(plot_df):,})',
        text=_hover_text(plot_df), hoverinfo='text',
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
        trace_kw = dict(
            x=grp[x_col], y=grp[y_col], z=grp['_zplot'],
            mode='markers+text',
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
            'labelMax': max(70, target + 10),
        }
        # Per-axis default lower-handle overrides, e.g. {'y': 0.35} to start the
        # association handle at 0.35 (clamped to the axis range).
        for _ax, _lo in (range_defaults or {}).items():
            if _ax in ranges_cfg:
                ranges_cfg[_ax]['lo'] = float(np.clip(_lo, ranges_cfg[_ax]['min'],
                                                      ranges_cfg[_ax]['max']))
        print(f'  [range_sliders] default box ≈ {_count(_p)} genes '
              f'(target {target}); drag any handle to widen/narrow')

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
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # 5) optional standalone HTML with on-hover structure thumbnails
    if html_path:
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        fig.write_html(html_path, include_plotlyjs='cdn')
        gene_patents_map = _build_gene_patents_html_map(
            gene_patents_df, gene_patents_top_n, depmap_url_template)
        import json as _json
        inject_data = (
            '<script>window.__GENE_COMPOUNDS__ = '
            + _json.dumps(custom if have_compounds else {}) + ';\n'
            'window.__GENE_PATENTS__ = '
            + _json.dumps(gene_patents_map) + ';\n'
            'window.__DEPMAP_URL__ = ' + _json.dumps(depmap_url_template) + ';\n'
            'window.__PAGE_SIZE__ = ' + str(int(page_size)) + ';\n'
            'window.__PLATES__ = ' + _json.dumps(list(all_plates)) + ';\n'
            'window.__ACTIVITIES__ = ' + _json.dumps(list(all_activities)) + ';\n'
            'window.__RANGES__ = ' + _json.dumps(ranges_cfg) + ';\n'
            'window.__AREA_DATA__ = ' + _json.dumps(area_data) + ';\n'
            'window.__VOLCANO_MODE__ = '
            + _json.dumps('svg' if (volcano_significant and 'significant'
                                    in (volcano_source.columns if volcano_source is not None else []))
                          else ('path' if (volcano_dir and html_path) else 'b64')) + ';\n'
            'window.__AXIS_LABELS__ = '
            + _json.dumps({'x': x_label, 'y': y_label, 'z': z_label}) + ';\n'
            'window.__AXIS_HELP__ = ' + _json.dumps(_axis_help) + ';\n'
            'window.__GENE_RESEARCH__ = ' + _json.dumps(gene_research or {}) + ';</script>')
        with open(html_path) as fh:
            html = fh.read()
        with open(html_path, 'w') as fh:
            fh.write(html.replace('</body>', inject_data + _INTERFACE_INJECT + '</body>'))
        print(f'wrote {html_path}  ({os.path.getsize(html_path) / 1e6:.1f} MB)')

    if nb_display:
        fig.show()

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


def plot_volcano_significant(df, uniquecontrast, gene,
                             *,
                             key='uniquecontrast',
                             sig_col='significant',
                             fc_thresh=1.0, p_thresh=0.05,
                             xmin=-5.0, xmax=5.0,
                             figsize=(6, 6), dpi=100,
                             up_color='#008bfb', down_color='#ff0051',
                             ns_color='lightgrey',
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
    :param str gene: gene symbol to ring/annotate; silently skipped if absent.
    :param str key: column identifying the experiment (default ``'uniquecontrast'``).
    :param str sig_col: significance-flag column; threshold fallback if missing.
    :param float fc_thresh, p_thresh: thresholds for the dashed guides (and the
        significance fallback when ``sig_col`` is absent).
    :param float xmin, xmax: x-axis (logfc) limits.
    :param str up_color, down_color, ns_color: dot colours.
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
    ax.scatter(agg.loc[ns,   'logfc'], agg.loc[ns,   'nlog10p'],
               s=8,  c=ns_color,   edgecolor='none', alpha=0.5,
               label=f'ns ({int(ns.sum())})')
    ax.scatter(agg.loc[up,   'logfc'], agg.loc[up,   'nlog10p'],
               s=12, c=up_color,   edgecolor='none', alpha=0.9,
               label=f'sig up ({int(up.sum())})')
    ax.scatter(agg.loc[down, 'logfc'], agg.loc[down, 'nlog10p'],
               s=12, c=down_color, edgecolor='none', alpha=0.9,
               label=f'sig down ({int(down.sum())})')

    # threshold guides
    ax.axhline(-np.log10(p_thresh), ls='--', lw=0.7, c='#888')
    ax.axvline(+fc_thresh,          ls='--', lw=0.7, c='#888')
    ax.axvline(-fc_thresh,          ls='--', lw=0.7, c='#888')

    # highlight target gene
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
