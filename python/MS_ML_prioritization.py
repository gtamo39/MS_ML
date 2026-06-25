"""
Plotting helpers for the chemistry prioritization workflow (`MS_ML_Prioritization.ipynb`).

Each function compares a SELECTED subset (e.g. the Pareto top-K) against the FULL
predicted library across a list of objective columns. Conventions shared by all:

  * ``objectives``  — list of numeric columns to look at (per-target logfc + p_single_low).
  * ``full`` / ``selected`` — the full prediction frame and the selected subset.
  * ``maximize``    — objectives we want HIGH (default ``('p_single_low',)``); the rest LOW.
  * ``colors``      — dict of named hexes (e.g. the notebook's ``SERAC_C``); merged over a
                      built-in default so the keys ``paper`` / ``azure`` / ``ember`` always exist.

All functions return their matplotlib Figure (the correlation one returns ``(fig, corr)``)
so callers can save / further-tweak; the inline backend renders them when called in a cell.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / 'Scripts'   # ../../Scripts (Statistics_tools)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import Statistics_tools as stats_tools

DEFAULT_COLORS = {'paper': '#E3E0DA', 'azure': '#0EA5CE', 'ember': '#E65D32'}


def _colors(colors):
    """Merge a (partial) color dict over the defaults so paper/azure/ember always resolve."""
    return {**DEFAULT_COLORS, **(colors or {})}


def plot_pairwise_matrix(objectives, full, selected, *, colors=None, maximize=('p_single_low',),
                         bg_n=50000, panel=2.4, dpi=110, seed=0):
    """
    Pairwise scatter matrix: ``selected`` (azure) over a faint ``full`` background (paper).

    Off-diagonal = scatter of both layers + a dashed median CROSSHAIR per layer (v=x-median,
    h=y-median). Diagonal = KDE on a TWIN axis so the row's y stays in value units (not density),
    with the dashed median vline. An X marks each panel's DESIRABLE corner — low for objectives,
    high for those in ``maximize`` (target-vs-target -> bottom-left; target-vs-p_single_low ->
    bottom-right). Column titles = objective names. Returns the Figure.
    """
    c = _colors(colors)
    bg = full.sample(min(len(full), bg_n), random_state=seed)
    n = len(objectives)
    fig, axes = plt.subplots(n, n, figsize=(panel * n, panel * n), dpi=dpi, sharex='col', sharey='row')
    for i, yv in enumerate(objectives):
        for j, xv in enumerate(objectives):
            ax = axes[i, j]
            if i == j:
                axk = ax.twinx()                          # KDE on its own density axis; main y stays in value units
                for sub, color, a in [(bg, c['paper'], 0.6), (selected, c['azure'], 0.5)]:
                    sns.kdeplot(x=sub[xv], ax=axk, color=color, fill=True, alpha=a)
                    axk.axvline(sub[xv].median(), color=color, ls='--', lw=1.6, zorder=5)
                axk.set_ylabel(''); axk.set_yticks([])
            else:
                ax.scatter(bg[xv],       bg[yv],       s=6, color=c['paper'], alpha=0.5, edgecolor='none')
                ax.scatter(selected[xv], selected[yv], s=9, color=c['azure'], alpha=0.8, edgecolor='none')
                for sub, color in [(bg, c['paper']), (selected, c['azure'])]:   # median crosshair per layer
                    ax.axvline(sub[xv].median(), color=color, ls='--', lw=1.2, zorder=4)
                    ax.axhline(sub[yv].median(), color=color, ls='--', lw=1.2, zorder=4)
            ax.set_xlabel(xv if i == n - 1 else ''); ax.set_ylabel(yv if j == 0 else '')
            if i == 0:
                ax.set_title(xv, fontweight='bold')       # column titles = objective name
    for i, yv in enumerate(objectives):                   # second pass (limits settled): desirable-corner X
        for j, xv in enumerate(objectives):
            if i == j:
                continue
            ax = axes[i, j]
            (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
            xc = x1 - 0.06 * (x1 - x0) if xv in maximize else x0 + 0.06 * (x1 - x0)
            yc = y1 - 0.06 * (y1 - y0) if yv in maximize else y0 + 0.06 * (y1 - y0)
            ax.scatter([xc], [yc], marker='X', s=170, color=c['ember'], edgecolor='k', linewidths=0.8, zorder=6)
    fig.suptitle('Pairwise objectives — selected (azure) over full library (paper); X = desirable corner', y=1.01)
    fig.tight_layout()
    return fig


def plot_objective_kde(objectives, dfs, colors, titles, *, ncols=3, dpi=110):
    """
    Per-objective KDE grid overlaying N distributions (one panel per objective). ``dfs``, ``colors``
    and ``titles`` are PARALLEL lists — e.g. ``[final_pred, prio_all, prio_noTGF]`` /
    ``[SERAC_C['paper'], SERAC_C['azure'], SERAC_C['ember']]`` / ``['lib', 'with TGFB1', 'without
    TGFB1']``. Each panel fills every distribution + draws its dashed median line (count + median in
    the legend). Mirrors the ``plot_nice_*`` overlay style in ``stats_tools.py``. Returns the Figure.
    """
    assert len(dfs) == len(colors) == len(titles), 'dfs, colors, titles must be parallel lists'
    n = len(objectives)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.67, nrows * 3.5), dpi=dpi)
    axflat = np.atleast_1d(axes).ravel()
    for ax, col in zip(axflat, objectives):
        for sub, color, name in zip(dfs, colors, titles):
            med = sub[col].median()
            sns.kdeplot(x=sub[col], ax=ax, fill=True, color=color, alpha=0.45,
                        label=f'{name} ({len(sub):,}, med={med:.2f})')
            ax.axvline(med, color=color, ls='--', lw=1.5)
        ax.set_title(col); ax.set_xlabel(''); ax.set_ylabel(''); ax.legend(fontsize=8)
    for ax in axflat[n:]:
        ax.axis('off')                                    # hide unused panels
    fig.suptitle('Per-objective value distribution — full library vs Pareto selections', y=1.02)
    fig.tight_layout()
    return fig


def plot_objective_correlation(objectives, full, *, method='spearman', cmap='RdBu_r',
                               figsize=(6.5, 5.5), dpi=120):
    """
    Correlation heatmap of the PREDICTED objectives over ``full`` (Spearman by default), drawn with
    ``stats_tools.heatmap`` + ``annotate_heatmap``, diverging cmap centered at 0. Reveals the objective
    geometry behind the Pareto front (aligned block vs antagonistic objectives). NOTE: read maximised
    objectives (e.g. p_single_low) with the goal flipped; predictions share H236 features so part of
    the correlation is induced by the shared featurisation. Returns ``(fig, corr)``.
    """
    corr = full[objectives].corr(method=method)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im, _ = stats_tools.heatmap(corr.values, objectives, objectives, ax=ax, cmap=cmap, vmin=-1, vmax=1,
                                cbarlabel=f'{method.capitalize()} ρ (predicted)')
    stats_tools.annotate_heatmap(im, valfmt='{x:.2f}', fontsize=8)
    ax.set_title('Predicted-objective correlation (full library)', pad=28)
    fig.tight_layout()
    return fig, corr


def plot_joint_pair(xv, yv, full, selected, *, colors=None, maximize=('p_single_low',),
                    bg_n=50000, height=6, seed=0):
    """
    Jointplot of ONE objective pair: central scatter (selected azure over full paper) with marginal
    KDEs on each axis, dashed medians in the marginals, and the desirable-corner X. Returns the
    seaborn ``JointGrid``.
    """
    c = _colors(colors)
    bg = full.sample(min(len(full), bg_n), random_state=seed)
    lo, hi = full[[xv, yv]].min(), full[[xv, yv]].max()
    g = sns.JointGrid(height=height, ratio=4)
    g.ax_joint.scatter(bg[xv],       bg[yv],       s=8,  color=c['paper'], alpha=0.5, edgecolor='none', label=f'all ({len(bg):,})')
    g.ax_joint.scatter(selected[xv], selected[yv], s=14, color=c['azure'], alpha=0.85, edgecolor='none', label=f'selected ({len(selected):,})')
    for sub, color in [(bg, c['paper']), (selected, c['azure'])]:
        sns.kdeplot(x=sub[xv], ax=g.ax_marg_x, color=color, fill=True, alpha=0.5)
        sns.kdeplot(y=sub[yv], ax=g.ax_marg_y, color=color, fill=True, alpha=0.5)
        g.ax_marg_x.axvline(sub[xv].median(), color=color, ls='--', lw=1.5)
        g.ax_marg_y.axhline(sub[yv].median(), color=color, ls='--', lw=1.5)
    xc, yc = (hi[xv] if xv in maximize else lo[xv]), (hi[yv] if yv in maximize else lo[yv])
    g.ax_joint.scatter([xc], [yc], marker='X', s=220, color=c['ember'], edgecolor='k', linewidths=1, zorder=6)
    g.set_axis_labels(xv, yv); g.ax_joint.legend(fontsize=8, loc='best')
    g.figure.suptitle(f'{xv} vs {yv} — selected (azure) over full (paper); X = desirable corner', y=1.02, fontsize=11)
    return g
