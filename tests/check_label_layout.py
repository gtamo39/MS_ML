"""
Check the layout of the discovery-curve names (fn.label_points_vertical, Px_Universe cell 9092b849).

The script draws a synthetic curve shaped like the real one (6,200 compounds, about 3,860 proteins,
cum = 6000 * (1 - exp(-x / 6000))) the same way as the cell. It places 28 names in four synthetic
position spreads. It also places a real-like spread: the real ranks of the validated compounds
(VALIDATED_TARGET_FILE, ranked among the MS_PATH compounds by SRB number)
with fake names of the real lengths. It prints counts only: no compound id, no gene name.

Run it from the repo root:
    python tests/check_label_layout.py               # font 8, the cell default
    python tests/check_label_layout.py --fs 9        # compare a larger font
Renders go to output/label_layout/ (git-ignored).
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import python.functions as fn

N = 6200   # compounds on the synthetic curve, about the real primary screen
K = 28     # names in each synthetic spread, as many as the real validated compounds


def synthetic_curve(n=N):
    """
    x = 1..n and a cumulative protein count shaped like the real discovery curve.
    return tuple: (x, cum)
    """
    x = np.arange(1, n + 1)
    return x, 6000 * (1 - np.exp(-x / 6000))


def synthetic_spreads(rng, n=N, k=K):
    """
    Four position spreads of k names: uniform, one cluster in the middle, one at the right end, and the edges.
    return dict: {spread name: positions}
    """
    return {'uniform': rng.choice(np.arange(1, n + 1), k, replace=False),
            'cluster': rng.choice(np.arange(2800, 4000), k, replace=False),
            'right': rng.choice(np.arange(4800, n + 1), k, replace=False),
            'edges': np.r_[rng.choice(np.arange(1, 400), 6, replace=False),
                           rng.choice(np.arange(n - 300, n + 1), 6, replace=False),
                           rng.choice(np.arange(400, n - 300), k - 12, replace=False)]}


def real_like_spread(n=N):
    """
    The real ranks of the validated compounds, scaled to n, with fake names of the real label lengths.
    Reads config VALIDATED_TARGET_FILE and MS_PATH; prints nothing.
    return tuple: (positions, names)
    """
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'config', 'config.yaml')))
    path = {k: cfg[k] if os.path.isabs(cfg[k]) else os.path.join(ROOT, cfg[k]) for k in ('VALIDATED_TARGET_FILE', 'MS_PATH')}
    lab = (fn.read_validated_cmp(path['VALIDATED_TARGET_FILE']).drop_duplicates(['compound', 'genes'])
           .groupby('compound')['genes'].agg('/'.join))
    ms = set(pd.read_parquet(path['MS_PATH'], columns=['compound'])['compound'])
    order = pd.Index(sorted(ms | set(lab.index), key=lambda c: int(c.split('-')[1])))
    pos = np.clip((np.array([order.get_loc(c) + 1 for c in lab.index]) * n / len(order)).astype(int), 1, n)
    return pos, [re.sub(r'[^/]', 'Q', g) for g in lab]


def draw_and_place(pos, names, fs, first):
    """
    Draw the curve as cell 9092b849 does (figure, colorbar, labels, note), then place the names.
    return tuple: (fig, ax, texts, lines, note)
    """
    x, cum = synthetic_curve()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sc = ax.scatter(x, cum, c=np.gradient(cum), s=6)
    fig.colorbar(sc, ax=ax, label='new proteins per 100 compounds (last 250)')
    ax.set(xlabel='compounds screened, lowest SRB first', ylabel=' New proteins discovered (cumulative)')
    note = ax.text(0.98, 0.04, '3,865 proteins discovered\n62.3 per 100 compounds on average\n'
                               '36.4 per 100 compounds at the end (last 250)',
                   transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    texts, lines = fn.label_points_vertical(ax, pos, cum[np.asarray(pos) - 1], names, curve_x=x, curve_y=cum,
                                            avoid=[note], first=first,
                                            text_kw=dict(fontsize=fs, fontweight='bold'), line_kw=dict(lw=0.8))
    return fig, ax, texts, lines, note


def layout_counts(fig, ax, texts, lines, note):
    """
    Count the layout faults, in display pixels. texts and lines come in the same (x) order.
    return dict: overlaps (label pairs), on_curve (labels over a curve point), line_through (a line through
                 another label), on_note, outside (labels out of the axes), off_centre (labels slid on their line)
    """
    r = fig.canvas.get_renderer()
    B = [t.get_window_extent(r) for t in texts]
    S = [ax.transData.transform(s) for s in lines.get_segments()]
    C = ax.transData.transform(np.c_[synthetic_curve()])
    A, nb, n = ax.get_window_extent(r), note.get_window_extent(r), len(B)
    return {'overlaps': sum(B[i].overlaps(B[j]) for i in range(n) for j in range(i + 1, n)),
            'on_curve': sum(((C[:, 0] > b.x0) & (C[:, 0] < b.x1) & (C[:, 1] > b.y0) & (C[:, 1] < b.y1)).any() for b in B),
            'line_through': sum(o.x0 < s[0, 0] < o.x1 and o.y0 < s[:, 1].max() and o.y1 > s[:, 1].min()
                                for i, s in enumerate(S) for j, o in enumerate(B) if j != i),
            'on_note': sum(b.overlaps(nb) for b in B),
            'outside': sum(b.x0 < A.x0 or b.x1 > A.x1 or b.y0 < A.y0 or b.y1 > A.y1 for b in B),
            'off_centre': sum(abs((b.x0 + b.x1) / 2 - s[0, 0]) > 1 for b, s in zip(B, S))}


def main():
    ap = argparse.ArgumentParser(description='Layout check of fn.label_points_vertical on the discovery curve.')
    ap.add_argument('--fs', type=float, default=8, help='font size of the names (cell VAL_FS)')
    ap.add_argument('--first', type=float, default=0.08, help='shortest line, fraction of the plot height (cell VAL_DY)')
    ap.add_argument('--png', default=os.path.join(ROOT, 'output', 'label_layout'), help='folder for the renders')
    ap.add_argument('--no-real', action='store_true', help='skip the real-like spread (it reads the config files)')
    a = ap.parse_args()
    os.makedirs(a.png, exist_ok=True)
    names = [f'GENE{i:02d}' for i in range(K)]
    names[3], names[11] = 'GENE03/GENE27', 'GENE11/GENE28'
    runs = {k: (p, names) for k, p in synthetic_spreads(np.random.default_rng(4)).items()}
    if not a.no_real:
        runs['real-like'] = real_like_spread()
    for k, (pos, nm) in runs.items():
        t = time.time()
        fig, ax, texts, lines, note = draw_and_place(pos, nm, a.fs, a.first)
        c = layout_counts(fig, ax, texts, lines, note)
        fig.savefig(os.path.join(a.png, f'{k}_fs{a.fs:g}.png'), bbox_inches='tight')
        plt.close(fig)
        print(f'{k:9s} names {len(texts):2d} | ' + ' | '.join(f'{m} {v}' for m, v in c.items()) + f' | {time.time() - t:.1f} s')


if __name__ == '__main__':
    main()
