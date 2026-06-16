"""Plotting helpers for the cytotoxicity analysis notebook (MS_cytotox.ipynb).

Kept here so the notebook cells stay thin and the logic is reusable / testable.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr


def plot_cytotox_vs_expression(
    mat, expr, gene='FBXO31', *,
    metric='viability',
    compounds=None,
    group_label=None,
    group_colors=None,
    n_label=6,
    min_cells=5,
    expr_unit='log2(TPM + 1)',
    point_color='#888888',
    figsize=(6.5, 5.5),
    dpi=110,
    ax=None,
):
    """
    Scatter a per-cell-line cytotox metric against a gene's expression — one dot per
    cell line — with a least-squares trend line, Spearman/Pearson stats in the title,
    and the most extreme cell lines annotated.

    The y metric is averaged across a CHOSEN set of compounds:
      * ``metric='viability'``   -> mean viability % over the selected compounds
                                     (LOWER = more killing)
      * ``metric='sensitivity'`` -> mean % killing = 100 - viability
                                     (HIGHER = more sensitive)

    :param DataFrame mat: viability matrix, compounds (rows) x cell lines (cols), %.
    :param DataFrame expr: expression matrix, cell lines (rows) x genes (cols).
    :param str gene: expression column to plot on x (default 'FBXO31').
    :param str metric: ``'viability'`` or ``'sensitivity'``.
    :param compounds: list of compound ids to average over; ``None`` = every compound
        present in ``mat`` (ids not in ``mat.index`` are ignored).
    :param group_label: optional Series (index = cell line) of categorical groups used
        to colour the dots (e.g. ``'sensitive'``/``'resistant'``); ``None`` = one colour.
    :param dict group_colors: ``{group: colour}`` for the dots; missing groups -> grey.
    :param int n_label: annotate the N most extreme cell lines (largest z-scored radial
        distance from the centre); 0/None = no labels.
    :param int min_cells: require >= this many shared, non-NaN cell lines, else raise.
    :param str expr_unit: unit appended to the x-axis label (e.g. ``'log2(TPM + 1)'``);
        pass ``None`` / ``''`` to show just ``'<gene> expression'``.
    :param str point_color: dot colour when ``group_label`` is None.
    :param tuple figsize / int dpi: figure size / resolution (ignored if ``ax`` given).
    :param ax: optional matplotlib Axes to draw on (else a new figure is created).

    :return dict: ``{'data': per-cell-line DataFrame, 'spearman': (rho, p),
        'pearson': (r, p), 'n': int, 'n_compounds': int, 'ax': Axes}``.
    """
    if metric not in ('viability', 'sensitivity'):
        raise ValueError("metric must be 'viability' or 'sensitivity'")
    if gene not in expr.columns:
        raise ValueError(f'{gene!r} not in expr columns')

    # 1) select compounds and compute the per-cell-line metric
    if compounds is None:
        sel = mat
    else:
        keep = [c for c in compounds if c in mat.index]
        if not keep:
            raise ValueError('none of the requested compounds are present in mat')
        sel = mat.loc[keep]
    viability = sel.apply(pd.to_numeric, errors='coerce').mean(axis=0)   # per cell line
    y_series = viability if metric == 'viability' else (100 - viability)

    # 2) align with the gene's expression over shared cell lines
    df = pd.DataFrame({'expr': expr[gene], metric: y_series}).dropna()
    if len(df) < min_cells:
        raise ValueError(f'only {len(df)} shared cell lines (< min_cells={min_cells})')
    x = df['expr'].values
    y = df[metric].values

    # 3) stats
    rho, p_s = spearmanr(x, y)
    r, p_p = pearsonr(x, y)

    # 4) colours
    gc = group_colors or {'sensitive': '#b8412f', 'resistant': '#5b9bd5'}
    if group_label is not None:
        groups = group_label.reindex(df.index)
        df['group'] = groups
        cols = groups.map(gc).fillna('#cccccc').values
    else:
        cols = point_color

    # 5) plot
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, c=cols, s=32, edgecolor='k', linewidth=0.3, alpha=0.9)
    b1, b0 = np.polyfit(x, y, 1)                       # least-squares trend line
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, b1 * xs + b0, color='k', lw=1.2, ls='--')

    # annotate the most extreme cell lines (largest z-scored radial distance from centre)
    if n_label:
        zx = (x - x.mean()) / (x.std() + 1e-9)
        zy = (y - y.mean()) / (y.std() + 1e-9)
        for j in np.argsort(np.hypot(zx, zy))[-int(n_label):]:
            ax.annotate(str(df.index[j]), (x[j], y[j]), fontsize=7, fontweight='bold',
                        xytext=(4, 4), textcoords='offset points')

    ylabel = ('mean viability (%)' if metric == 'viability'
              else 'sensitivity (mean % killing)')
    ax.set_xlabel(f'{gene} expression - {expr_unit}' if expr_unit else f'{gene} expression')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs {gene} expression  ({sel.shape[0]} compounds)\n'
                 f'Spearman rho={rho:.2f} (p={p_s:.3g})  |  '
                 f'Pearson r={r:.2f} (p={p_p:.3g})')

    if group_label is not None:
        handles = [Line2D([0], [0], marker='o', ls='', mec='k', mfc=c, label=g)
                   for g, c in gc.items()]
        handles.append(Line2D([0], [0], marker='o', ls='', mec='k',
                               mfc='#cccccc', label='other'))
        ax.legend(handles=handles, frameon=False, fontsize=9, loc='best')

    return {'data': df, 'spearman': (rho, p_s), 'pearson': (r, p_p),
            'n': len(df), 'n_compounds': sel.shape[0], 'ax': ax}


def cell_sensitivity(mat, compounds=None):
    """
    Per-cell-line sensitivity = mean % killing (``100 - viability``) across a chosen
    compound set.

    :param DataFrame mat: viability matrix, compounds (rows) x cell lines (cols), %.
    :param compounds: compound ids to average over; ``None`` = every compound in ``mat``
        (ids not in ``mat.index`` are ignored). Higher value = more sensitive.
    :return Series: sensitivity per cell line (index = cell line).
    """
    if compounds is None:
        sel = mat
    else:
        keep = [c for c in compounds if c in mat.index]
        if not keep:
            raise ValueError('none of the requested compounds are present in mat')
        sel = mat.loc[keep]
    return (100 - sel.apply(pd.to_numeric, errors='coerce')).mean(axis=0)


def sensitivity_labels(mat, compounds=None, *, method='tertile', frac=1 / 3):
    """
    Discretise per-cell-line sensitivity (over a chosen compound set) into
    sensitive/resistant labels — the binarisation used for the cytotox classifier.

    :param DataFrame mat: viability matrix, compounds (rows) x cell lines (cols), %.
    :param compounds: compound ids whose mean killing defines sensitivity; ``None`` = all.
    :param str method: ``'tertile'`` (a.k.a. extreme/quantile split — the top ``frac``
        is 'sensitive', the bottom ``frac`` is 'resistant', the ambiguous middle is
        DROPPED) or ``'median'`` (above-median 'sensitive', else 'resistant', nothing
        dropped).
    :param float frac: tail fraction kept at each extreme for the ``'tertile'`` split,
        in ``(0, 0.5]``. ``1/3`` = classic tertile split; ``0.25`` = top/bottom quartiles;
        smaller = more extreme contrast (fewer, cleaner cell lines).
    :return Series: 'sensitive'/'resistant' per labelled cell line (name='group').
    """
    sens = cell_sensitivity(mat, compounds)
    lab = pd.Series(index=sens.index, dtype=object, name='group')
    if method == 'tertile':
        if not 0 < frac <= 0.5:
            raise ValueError('frac must be in (0, 0.5]')
        lo, hi = sens.quantile([frac, 1 - frac])
        lab[sens >= hi] = 'sensitive'
        lab[sens <= lo] = 'resistant'
        return lab.dropna()                 # drop the ambiguous middle band
    if method == 'median':
        med = sens.median()
        lab[sens >= med] = 'sensitive'
        lab[sens < med] = 'resistant'
        return lab
    raise ValueError("method must be 'tertile' or 'median'")


def plot_gene_expression_heatmap(
    expr, genes, cells=None, *,
    group_label=None, group_colors=None,
    cmap='viridis', center=None, square=True, annot=False, standardize=False,
    expr_unit='log2(TPM + 1)', figsize=None, dpi=120, ax=None,
):
    """
    Heatmap of a few selected genes (rows) across cell lines (columns), coloured by
    expression level — a slide-ready square-cell figure.

    :param DataFrame expr: expression matrix, cell lines (rows) x genes (cols).
    :param genes: list of gene columns to show (rows of the heatmap); missing ones skipped.
    :param cells: cell lines (columns) to show; ``None`` = every line in ``expr``.
    :param group_label: optional Series (index = cell line) of sensitive/resistant groups;
        when given, columns are ordered group-by-group (sensitive first, then by mean
        expression of the selected genes), separated by a divider, and the x-tick labels
        are coloured by group.
    :param dict group_colors: ``{group: colour}`` for the x-tick labels.
    :param cmap: matplotlib colormap name or Colormap object for the expression values.
    :param center: value at which to centre the colormap (e.g. ``0`` with a diverging
        cmap when ``standardize=True``); ``None`` = no centring.
    :param bool square: draw square cells (overall shape then follows #genes x #cells).
    :param bool annot: write the value inside each cell (use only for few cell lines).
    :param bool standardize: z-score each gene across cell lines (contrast) instead of
        raw values; the colour bar label switches accordingly.
    :param str expr_unit: unit shown on the colour bar when not standardizing.
    :param figsize/dpi: figure size / resolution (auto-sized from counts if None).
    :param ax: optional Axes to draw on.
    :return: the matplotlib Axes.
    """
    import seaborn as sns
    genes = [g for g in genes if g in expr.columns]
    if not genes:
        raise ValueError('none of the requested genes are in expr.columns')
    cl = [c for c in (list(cells) if cells is not None else list(expr.index))
          if c in expr.index]
    if not cl:
        raise ValueError('no requested cell lines are in expr.index')

    gc = group_colors or {'sensitive': '#b8412f', 'resistant': '#5b9bd5'}
    seps = []
    if group_label is not None:
        rank = {'sensitive': 0, 'resistant': 1}
        meanexpr = expr.loc[cl, genes].mean(axis=1)
        g = group_label.reindex(cl)
        cl = sorted(cl, key=lambda c: (rank.get(g.get(c), 2), -meanexpr.get(c, 0.0)))
        seq = list(group_label.reindex(cl).values)
        seps = [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]   # group dividers

    H = expr.loc[cl, genes].T.astype(float)   # rows = genes, cols = cell lines
    if standardize:
        H = H.sub(H.mean(axis=1), axis=0).div(H.std(axis=1) + 1e-9, axis=0)
        cbar_label = 'expression (z-score per gene)'
    else:
        cbar_label = f'expression  {expr_unit}' if expr_unit else 'expression'

    if ax is None:
        if figsize is None:
            figsize = (max(6.0, 0.32 * len(cl)), max(2.2, 0.55 * len(genes) + 1.4))
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(H, cmap=cmap, center=center, square=square, linewidths=0.4,
                linecolor='white', annot=annot, fmt='.1f', ax=ax,
                cbar_kws={'label': cbar_label, 'shrink': 0.5})
    for s in seps:                                  # vertical group separators
        ax.axvline(s, color='k', lw=1.5)
    g = group_label.reindex(cl) if group_label is not None else None
    for tick, c in zip(ax.get_xticklabels(), cl):   # rotate + colour x labels
        tick.set_rotation(90)
        tick.set_fontsize(7)
        if g is not None:
            tick.set_color(gc.get(g.get(c), '#444444'))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel('cell line')
    ax.set_ylabel('')
    return ax
