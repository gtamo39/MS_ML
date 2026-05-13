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
    var gd   = document.querySelector(".plotly-graph-div") || document.querySelector(".js-plotly-plot");
    if (!gd) return;
    var pinned = false;
    var currentGene = "";
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
      return true;
    }
    function unpin() {
      pinned = false;
      box.classList.remove("pinned");
      box.style.display = "none";
      volBox.style.display = "none";
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
    });
    gd.on("plotly_click", function(e) {
      if (render(e.points && e.points[0])) {
        pinned = true;
        box.classList.add("pinned");
        box.style.display = "block";
      }
    });
    clo.addEventListener("click", unpin);
    document.addEventListener("keydown", function(e) {
      if (e.key === "Escape" && pinned) unpin();
    });
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
    disease_area_colors=None,
    na_area_color='#bbbbbb',
    title='SAR predictability × disease relevance × MCS fold-enrichment',
    html_path=None,
    height=900,
    width=1500,
    show=True,
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

    custom = {}
    for _, row in highlighted.iterrows():
        triples = []
        # Index 0 is a per-gene META row: ['__META__', '', '<fisher_p str>'].
        # The hover JS detects '__META__' to populate the panel header; the
        # existing compound-render loop skips it because t[1] (b64) is empty.
        fp_val = row.get('fisher_p') if 'fisher_p' in highlighted.columns else None
        if fp_val is None or pd.isna(fp_val):
            fp_str = '—'
        else:
            fp_str = '< 0.0001' if fp_val < 0.0001 else f'{fp_val:.4f}'
        triples.append(['__META__', '', f'fisher_p={fp_str}'])
        for k in range(1, top_n_hover + 1):
            c = row.get(f'top{k}_compound')
            s = row.get(f'top{k}_smiles')
            l = row.get(f'top{k}_logfc')
            triples.append([
                str(c) if pd.notna(c) else '',
                _compound_b64(c if pd.notna(c) else None,
                              s if pd.notna(s) else None),
                f'{l:.2f}' if pd.notna(l) else '',
            ])
        custom[row['gene']] = triples

    n_thumbs = _stats['png'] + _stats['rdkit']
    print(f'> built {n_thumbs:,} structure thumbnails across {len(custom)} highlighted genes '
          f'(png={_stats["png"]}, rdkit-fallback={_stats["rdkit"]}, missing={_stats["miss"]}; '
          f'png_dir={png_dir!r})')

    # 3b) optional per-(gene, compound) volcano thumbnails. Each compound row
    #     gets a 4th element: a base64 PNG of the volcano. Shown by the JS
    #     panel only when pinned AND the user hovers a compound cell.
    if df_raw is not None:
        # Build the task list once; pad missing-compound rows so JS always has t[3].
        tasks = []
        for g, triples in custom.items():
            for i in range(1, len(triples)):
                t = triples[i]
                if t[0]:
                    tasks.append((g, t[0], i))
                else:
                    triples[i] = list(t) + ['']
        n_expected = len(tasks)

        if n_expected == 0:
            pass
        elif volcano_n_jobs == 1:
            # ----- serial path (unchanged behaviour for n_jobs=1) -----
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
                custom[g][i] = list(custom[g][i]) + [b64]
                pbar.update(1)
            pbar.close()
        else:
            # ----- parallel path: pre-slice df_raw per compound, then loky -----
            # Pre-slicing keeps per-task IPC payloads tiny — sending the full
            # ~200k-row df_raw to every worker would dominate runtime.
            import contextlib
            import joblib as _joblib
            from joblib import Parallel, delayed
            unique_cmps = sorted({c for _, c, _ in tasks})
            sub_cache = {
                c: df_raw[df_raw['compound'] == c]
                       [['compound', 'genes', 'logfc', 'pvalue']].dropna()
                for c in unique_cmps
            }
            print(f'> pre-sliced df_raw for {len(unique_cmps):,} compounds; '
                  f'rendering {n_expected:,} volcanoes on {volcano_n_jobs} workers...')

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
                custom[g][i] = list(custom[g][i]) + [b64]
        print(f'> rendered {n_expected:,} volcanoes')
    else:
        # Pad compound rows so JS can always read t[3] safely.
        for triples in custom.values():
            for i in range(1, len(triples)):
                triples[i] = list(triples[i]) + ['']

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
        with open(html_path) as fh:
            html = fh.read()
        with open(html_path, 'w') as fh:
            fh.write(html.replace('</body>', _HOVER_INJECT + '</body>'))
        print(f'wrote {html_path}  ({os.path.getsize(html_path) / 1e6:.1f} MB)')

    if show:
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
    ax.set_title(title or f'{compound}  ({len(agg):,} genes)')
    ax.legend(loc='best', fontsize=8, frameon=False)
    plt.tight_layout()
    return agg


def _volcano_render_worker(args):
    """Module-level worker used by `plot_target_3d` when `n_jobs > 1`.

    Has to be at module level so loky/cloudpickle can serialize it by
    reference rather than trying to pickle a closure. Receives a small
    pre-sliced per-compound DataFrame instead of the full ``df_raw`` to
    keep the per-task IPC payload tiny.
    """
    import io, base64
    import matplotlib
    matplotlib.use('Agg')  # headless backend in workers
    import matplotlib.pyplot as plt
    gene, compound, sub, size_px, xmin, xmax = args
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


def compute_gene_sar_r2(
    gene, df_raw, features,
    *,
    label_col='logfc',
    model_class=None,
    model_params=None,
    min_compounds=100,
    n_null=0,
    n_jobs=8,
    seed=0,
    ML_Reg_module=None,
    verbose=False,
):
    """
    5-fold cross-validated SAR predictability for one gene.

    Filters ``df_raw`` to the gene, aggregates ``label_col`` per compound (mean
    across replicates), merges with ``features`` on ``compound``, and runs the
    project's K-fold CV harness to get an R². Optionally repeats with shuffled
    labels ``n_null`` times to estimate the mean of the null distribution.

    The returned dict matches the SAR-screen CSV header verbatim, so a caller
    can do ``writer.writerow(result)`` with no transformation. Skipped genes
    (``n <= min_compounds``) return NaN R²/nullR² with the actual compound
    count, so the caller's resume-set still includes them and they don't get
    retried on the next pass.

    :param str gene: gene symbol to filter ``df_raw['genes']`` on
    :param df df_raw: must have ``genes``, ``compound``, and ``label_col``
    :param df features: molecular features keyed by ``compound``
    :param str label_col: which column to predict (e.g. ``'logfc'`` or ``'logfc_corrected'``)
    :param type model_class: e.g. ``RandomForestRegressor``; instantiated fresh per call
    :param dict model_params: kwargs for the model constructor
    :param int min_compounds: skip if compounds-after-merge ≤ this
    :param int n_null: label-shuffle permutations for null R²; 0 = skip
    :param int n_jobs: passed as ``n_jobs`` to the model
    :param int seed: passed as ``random_state`` to the model
    :param module ML_Reg_module: project's CV harness, passed in to avoid hard imports
    :return dict: ``{'gene', 'R2', 'nullR2', 'n'}``
    """
    if model_class is None:
        raise ValueError('pass model_class (e.g. RandomForestRegressor)')
    if ML_Reg_module is None:
        raise ValueError('pass ML_Reg_module so we use the same CV harness as the notebook')
    if model_params is None:
        model_params = {}

    sub = df_raw[df_raw['genes'] == gene]
    if sub.empty:
        return {'gene': gene, 'R2': float('nan'), 'nullR2': float('nan'), 'n': 0}

    agg = (sub.groupby('compound')[label_col].mean()
              .reset_index()
              .dropna(subset=[label_col])
              .rename(columns={label_col: 'label'}))

    ML_data = pd.merge(features, agg, on='compound').dropna()
    n = len(ML_data)

    if n <= min_compounds:
        if verbose:
            print(f'  [skip] {gene}: only {n} compounds (min_compounds={min_compounds})')
        return {'gene': gene, 'R2': float('nan'), 'nullR2': float('nan'), 'n': n}

    def _new_model():
        # fresh instance per call so RF/XGB internal state never leaks between fits
        return model_class(**{**model_params, 'n_jobs': n_jobs, 'random_state': seed})

    _, df_pred = ML_Reg_module.run_K_Fold_Xval_Regression(
        ML_data, model=_new_model(),
        col_to_rm=['compound', 'label'], ID='compound',
        get_ints=False, v=False, to_impute=None, rm_empty_cols=False,
    )
    R2 = ML_Reg_module.get_reg_metrics_from_preddf(df_pred, v=False)['r2']

    null_R2 = float('nan')
    if n_null > 0:
        rng = np.random.default_rng(seed)
        nulls = []
        for _ in range(n_null):
            shuffled = ML_data.copy()
            shuffled['label'] = rng.permutation(shuffled['label'].values)
            _, df_pred_null = ML_Reg_module.run_K_Fold_Xval_Regression(
                shuffled, model=_new_model(),
                col_to_rm=['compound', 'label'], ID='compound',
                get_ints=False, v=False, to_impute=None, rm_empty_cols=False,
            )
            nulls.append(ML_Reg_module.get_reg_metrics_from_preddf(df_pred_null, v=False)['r2'])
        null_R2 = float(np.mean(nulls))

    if verbose:
        print(f'  {gene}: R²={R2:.3f}  null={null_R2:.3f}  n={n}')

    return {'gene': gene, 'R2': float(R2), 'nullR2': null_R2, 'n': n}


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
