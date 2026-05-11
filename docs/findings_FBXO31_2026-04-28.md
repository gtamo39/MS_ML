# FBXO31 active/silent classifier — autoresearch findings

**Run date:** 2026-04-28
**Dataset:** 1,696 unique compounds, 70 % positive (active), curated through 2026-04-23
**Loop driver:** [autoresearch/loop_prompt.md](../autoresearch/loop_prompt.md)
**Full log:** [autoresearch/logs/autoresearch.jsonl](../autoresearch/logs/autoresearch.jsonl)

This is a snapshot of what the autoresearch loop produced on this dataset
on this date. Generic methodology lives in [autoresearch.md](autoresearch.md);
this file is run-specific.

---

## TL;DR

Autoresearch found a +0.006 K-fold roc_auc gain (champion `t2_morgan_plus_maccs`),
but multi-evaluator triangulation showed it was largely an artefact of the
K-fold metric. **The deployable model is the smaller `t2_maccs_plus_physchem`**
(173 features instead of 2,221) — equivalent K-fold, equivalent scaffold-CV,
**better temporal AUC**, simpler, more interpretable.

The bigger finding is that **all model classes hover at ~0.59 on the temporal
split** (XGB champion, smaller XGB, ChemProp D-MPNN). The chemistry / library /
assay distribution shifted between training and the 2026-04-23 batch; no model
architecture on the current dataset extrapolates well into that tranche.

---

## Three-evaluator leaderboard

|                                      | random K-fold | scaffold K-fold | temporal split (n=212) |
|---|---|---|---|
| XGB baseline (Morgan, 2054 feat)            | 0.6909 | ≈0.68  | n/a    |
| XGB champion (Morgan + MACCS, 2221 feat)    | **0.6969** | 0.6842 | 0.5883 |
| **XGB smaller (MACCS + physchem, 173 feat)** ← deployed | 0.6942 | 0.6828 | **0.6128** |
| ChemProp D-MPNN (graph)                     | (not run) | (not run) | 0.5915 |
| Naive (DummyClassifier stratified)          | ≈0.50 | ≈0.50  | ≈0.50  |

(All XGB models use the baseline params: `n_estimators=200, max_depth=3,
learning_rate=0.05, scale_pos_weight=1`, `random_state=42`.)

### Reading the three columns

- **Random ≈ Scaffold (Δ ≈ 0.012)**: scaffold leakage in K-fold is small.
  Bemis-Murcko produced 760 unique scaffolds across 1696 compounds —
  ~2.2 compounds/scaffold — so random splits don't get many "free" hits
  from same-scaffold neighbours.
- **Scaffold ≫ Temporal (Δ ≈ 0.07–0.09)**: the actual generalisation cliff
  is *time*, not scaffold. The 2026-04-23 batch is in a different region of
  chemical / assay / property space than the training data through 2026-04-15.

---

## What the loop tried

22 logged iterations across 4 tiers + 4 derived experiments. Full audit in
[autoresearch/logs/autoresearch.jsonl](../autoresearch/logs/autoresearch.jsonl);
human-readable trail in [autoresearch/candidates.md](../autoresearch/candidates.md).
Highlights:

| Direction | Outcome |
|---|---|
| **Class re-weighting** (`scale_pos_weight ∈ {1,3,6,10}`) | All `>1` regress. The dataset is 70 % positive; positives aren't the minority. |
| **Bigger fingerprints** (Morgan r=3, nBits=4096; multi-FP stacks) | All regress. With n=1696, more FP bits is more noise per signal. |
| **Different fingerprint families** (atom-pair, topological torsion, MACCS, no-chirality) | MACCS alone matches Morgan; others worse. |
| **Different model classes** (RF, ExtraTrees, LogReg, ChemProp) | All within ±0.01 of XGB on K-fold; no clear winner. |
| **Synthesised: Morgan + MACCS** (list-extender mode) | +0.006 K-fold — **first kept improvement, became champion**. |
| **Synthesised: MACCS + physchem only** (list-extender mode) | Within 0.003 of champion on K-fold; +0.024 on temporal split. **The model worth deploying.** |
| **Threshold optim / calibration** | Skipped — monotonic transforms don't move AUC. |

What the loop *didn't* try (out of scope this run): scaffold-aware metric
during the loop (would have changed the leaderboard), data augmentation,
applicability-domain filtering, ensembling.

---

## What this means for the model

### Deployable

Use `t2_maccs_plus_physchem`:
- **Features (173):** MACCS keys (167) + physicochemical descriptors (`Hba`, `Hbd`, `MW`, `TPSA`, `LogP`, `NRB`).
- **Model:** `XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, scale_pos_weight=1, random_state=42)`.
- **Cached features:** [autoresearch/logs/ML_data_morgan_plus_maccs.pkl](../autoresearch/logs/ML_data_morgan_plus_maccs.pkl) (subset to MACCS+physchem cols at deploy time, or rebuild the smaller pkl).
- **Reproduce in notebook:**
  ```python
  from autoresearch.run_one import load_ML_data, run_one
  ML_data = load_ML_data('autoresearch/logs/ML_data_morgan_plus_maccs.pkl')
  PHYSCHEM = ['Hba','Hbd','MW','TPSA','LogP','NRB']
  MACCS = [c for c in ML_data.columns if c.startswith('MK')]
  change = {'id': 't2_maccs_plus_physchem',
            'desc': 'deployed model',
            'model': {'cls': 'XGBClassifier',
                      'params': {'n_estimators': 200, 'max_depth': 3,
                                 'learning_rate': 0.05}},
            'features': PHYSCHEM + MACCS, 'cutoff': 0.5, 'folds': 5}
  rec = run_one(ML_data, change)
  ```

### Not deployable as-is

- **The K-fold champion (`t2_morgan_plus_maccs`)** carries 13× more features
  with no temporal benefit. The Morgan FPs actively *hurt* on temporal split
  (champion 0.59 vs smaller 0.61).
- **ChemProp** doesn't help. Same temporal AUC as XGB at 30× the training
  time. Could revisit with `cv` mode + scaffold-aware splits + tuned hyperparams,
  but the prior is unfavourable on this dataset size.

---

## What this means for next steps

The biggest leverage isn't another model — it's **more diverse and more
recent training data**. Concrete moves:

1. **Quantify the temporal drift directly.**
   - Property-distribution shift: `MW`, `LogP`, `TPSA` etc. between
     `date<2026-04-23` and `date>=2026-04-23`.
   - Scaffold-distribution shift: how many 2026-04-23 scaffolds appear in
     the training set?
   - Tanimoto-NN: for each held-out compound, distance to nearest training
     compound. Predict-vs-distance plot will show if performance collapses
     beyond a similarity threshold.
2. **Applicability domain filter.** Wrap the deployed model with a similarity
   gate: refuse to predict (or flag low-confidence) for compounds whose
   nearest training neighbour is below e.g. Tanimoto 0.3. This is honest
   uncertainty quantification, not a model improvement.
3. **Re-train periodically.** When the next ≥100 compounds with labels are
   available, fold them in and re-run the autoresearch loop with `metric_name`
   switched to **scaffold-grouped K-fold roc_auc** (or a composite of
   scaffold-CV and temporal AUC). The current K-fold target is mis-aligned
   with the deployment metric.
4. **Active learning** (`t4_active_learning_round`, deferred this run) once
   you're ready to order compounds — the deployed model + the enumeration
   set + uncertainty sampling pick the next batch. Won't *improve* the
   current model; will improve the *next* one.

---

## Operational notes from this run

- **22 loop iterations + 4 derived experiments** consumed roughly 1 hour of
  wall-clock and ~1 chat session of context.
- **List-extender mode produced both winning ideas** (`morgan_plus_maccs` and
  `maccs_plus_physchem`). Strict list-consumer would have shipped the
  Morgan-baseline as champion and stopped — see
  [autoresearch.md § list-consumer vs list-extender](autoresearch.md#list-consumer-vs-list-extender).
- **One bug caught in flight**: ChemProp's first temporal eval reported
  `roc_auc=1.0`. Diagnosed as ground-truth column leakage in the predictions
  CSV; fixed in [autoresearch/chemprop_eval.py](../autoresearch/chemprop_eval.py)
  with `--drop-extra-columns` + name-pattern column matching. Bogus log entry
  was excised and the corrected number (`0.5915`) re-logged.
- **Data locality respected throughout** — no SMILES, labels, or predictions
  ever left the machine; locality rule formalised in [CLAUDE.md](../CLAUDE.md)
  and persisted to agent memory.

---

## File pointers

- Champion artefacts:
  - [autoresearch/logs/best.json](../autoresearch/logs/best.json) (K-fold champion `t2_morgan_plus_maccs`)
  - [autoresearch/logs/ML_data_morgan_plus_maccs.pkl](../autoresearch/logs/ML_data_morgan_plus_maccs.pkl)
  - [autoresearch/logs/ML_data_t2_maccs_keys.pkl](../autoresearch/logs/ML_data_t2_maccs_keys.pkl) (smaller deploy candidate)
- Full audit trail:
  - [autoresearch/logs/autoresearch.jsonl](../autoresearch/logs/autoresearch.jsonl)
  - [autoresearch/candidates.md](../autoresearch/candidates.md)
- ChemProp eval script: [autoresearch/chemprop_eval.py](../autoresearch/chemprop_eval.py)
- Generic autoresearch reference: [autoresearch.md](autoresearch.md)
