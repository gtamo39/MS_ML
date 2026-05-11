# Autoresearch — setup reference

A canonical "how to set up an autoresearch loop in this repo" guide. Use it
when you want an agent to **iterate autonomously** on any problem with a
mechanical metric — ML model improvement, prompt engineering, parameter
sweeps, anything where you can score the output and prefer it.

---

## What it is

Autoresearch (Karpathy's term) is a tight loop:

```
  ┌────────────────────────────────────────────────────────────┐
  │ 1. Pick the next change from a candidate list              │
  │ 2. Apply it; run a deterministic experiment                │
  │ 3. Score against a fixed metric                            │
  │ 4. Keep if metric improved by > threshold; else revert     │
  │ 5. Append everything to a structured log                   │
  │ 6. Repeat until exhausted, time-up, or N non-improvements  │
  └────────────────────────────────────────────────────────────┘
```

The key insight is that *you don't need AGI* — you need a **goal, a metric,
and a loop that never quits**. Compounding gains do the rest.

---

## Three things you must have before starting

| Ingredient | Why it matters |
|---|---|
| **A scalar metric** the loop optimises | Without one number, the agent can't "keep wins / revert losses". Pick one (e.g. `pr_auc`). |
| **A deterministic harness** that returns it | Same input → same metric. Fix `random_state` everywhere. The agent reasons about deltas, not noise. |
| **A persistent log + champion file** | The loop must survive crashes / restarts. State lives on disk, not in chat. |

If you can't answer "what number am I trying to make go up?" cleanly, fix that
first. No amount of agent infrastructure substitutes for a shaky metric.

---

## Directory layout in this repo

Each autoresearch *project* lives in its own subfolder under
`autoresearch/`. A `template/` folder holds the canonical files you'd copy
when starting a new project:

```
autoresearch/
├── template/                       # canonical files; copy when starting a new project
│   ├── candidates.md
│   ├── loop_prompt.md
│   ├── run_one.py
│   └── logs/
│       ├── autoresearch.jsonl
│       ├── best.json
│       └── ML_data.pkl
└── <project_name>/                 # one folder per research project
    ├── config.yaml                 # objectives, knobs, DOs/DONTs + system_prompt
    ├── baseline.ipynb              # the canonical pipeline the project iterates on
    ├── candidates.md               # flat experiment queue (one row per `change`)
    ├── run_one.py                  # project-specific harness, returns metric dict
    └── logs/
        ├── autoresearch.jsonl      # one JSON line per experiment
        ├── best.json               # current champion
        ├── hypotheses.md           # hypothesis evolution graph (one section / iteration)
        └── inputs.pkl              # cached pre-processed inputs (built once)
```

Everything the agent or you produces for a project lives under that project's
subfolder. The existing repo (`python/`, `vignettes/`, `data/`) is the source
of truth for data + utilities; `autoresearch/<project>/` is purely the
experiment-tracking layer.

The split into per-project folders matters because each project has its own
`config.yaml` (objectives, hypotheses, knobs) and its own `system_prompt` —
different problems optimise different metrics under different constraints, so
sharing one `loop_prompt.md` across projects breaks down quickly.

---

## Per-project workflow (`config.yaml`-driven)

Starting a new autoresearch project is a fixed sequence: the human fills in
the *objectives* half of `config.yaml`, the agent (Claude) writes the
*system_prompt* half, the human reviews, then the harness gets built. The
goal is to keep the canonical "what are we trying to do?" decisions in
human-editable YAML and let the agent translate them into the operational
prompt that the loop will consume.

### Step 1 — Scaffold a new project

```bash
mkdir autoresearch/<project_name>
mkdir autoresearch/<project_name>/logs
```

Drop in a `baseline.ipynb` showing the canonical pipeline (data load →
features → metric). The agent reads this end-to-end before iterating, so
keep it tight and well-commented; **no autoresearch decisions belong here**
— it's just the unmodified baseline that every `change` is a delta over.

### Step 2 — Fill in `config.yaml` (human side)

The human owns every field above `system_prompt`. The intended schema:

```yaml
# --- Objectives --- #
research_objective:
 - "<one-line statement of what the loop must improve>"

propose_hypotheses:
 - "<seed hypothesis 1 — testable claim, e.g. 'plate noise hurts predictions'>"
 - "<seed hypothesis 2>"

# --- Loop knobs --- #
number_of_iterations: 5            # cycles of (read → test → propose)
novel_hypotheses_per_iter: 3       # how many fresh hypotheses each cycle spawns
optimization_metric: <name>        # e.g. 'mean_r2', 'pr_auc'
improvement_threshold: 0.005       # min Δ on optimization_metric to update champion

stop_conditions:
  max_iterations: 5                # cycles, NOT individual experiments
  consecutive_no_improve: 8        # safety net
  wall_clock_hours: 12             # safety net

hypothesis_log: autoresearch/<project>/logs/hypotheses.md
candidates_log: autoresearch/<project>/candidates.md

Instructions DOs:
 - "<positive instruction>"

Instructions DONTs:
 - "<negative instruction; especially methodological guardrails like 'no per-gene drops'>"

# --- Autoresearcher prompt --- #
system_prompt: |
  # leave empty; Claude will fill this in
```

The `propose_hypotheses` block is the human's seed list — H1, H2, ... — and
seeds the *first iteration only*. After that, hypotheses are spawned by the
loop (see step 3 of the iteration cycle below).

The `Instructions DON'Ts` block is the most important field on the human
side. Each one becomes a hard constraint in the system_prompt — they're
where you encode methodological guardrails the agent might otherwise
optimise its way around (e.g. "no per-gene plate drops" → blocks the
agent from boosting individual gene R² by gene-specific filters that
won't generalise).

### Step 3 — Ask Claude to write the `system_prompt`

> "Read autoresearch/<project>/config.yaml, baseline.ipynb, and
> autoresearch/template/. Translate the objectives + knobs + DOs/DONTs into
> the system_prompt field of config.yaml."

Claude reads the human-side fields, references the template's canonical
prompt structure, and emits a `system_prompt: |` block that:

- Restates the goal in operational terms.
- Points at `baseline.ipynb` for orientation.
- Names the harness (`run_one.py`) and its `change` schema.
- Defines the **iteration cycle** (READ active hypotheses → TEST each as a
  focused `change` → PROPOSE `novel_hypotheses_per_iter` new ones for the
  next cycle).
- References every config knob by name (so the agent reads the config, not
  hardcoded numbers).
- Encodes the DOs/DONTs as agent-readable constraints.

### Step 4 — Review the `system_prompt`

Read it. Ask Claude to revise specific sections rather than rewriting from
scratch. Common things to check:

- Is the optimisation metric named correctly?
- Are the DON'Ts encoded as *hard rules*, not soft suggestions?
- Are config knobs referenced by name (so future tweaks propagate)?
- Are stop conditions reconciled (no contradictory numbers from the prompt
  vs the config)?
- Is the iteration cycle correctly cycle-based (one iteration tests N
  hypotheses) and not the legacy "1 experiment per iteration"?

### Step 5 — Build the harness (`run_one.py`)

Adapt `template/run_one.py` to your project's metric. Public API stays the
same (`run_one`, `append_log`, `load_best`, `update_best_if_improved`,
`cache_inputs` / `load_inputs`); only `_evaluate` and the metric-name
default change. For regression problems, add `mean_r2` / `median_r2` /
`per_gene_r2` to the rec dict; for classification, the FBXO31 set
(`roc_auc`, `pr_auc`, `f1`, `mcc`) is reusable.

### Step 6 — Smoke-test on the **real** cached inputs

This step is non-negotiable. A single 3-gene sandbox test (the kind we use
during harness development) catches schema bugs but **does not** exercise
the memory-heavy paths that show up only when `run_one` runs against the
real cached pickle. Skipping this step has cost a multi-hour OOM crash on
this very project — see the gotchas section below.

The smoke test must do all of the following on the **actual** `inputs.pkl`
(not a sandbox subset):

1. **`load_inputs()` round-trip** — confirms the cache is well-formed and
   times the load (slow loads slow every iteration; > 60 s is a flag).
2. **Cache size sanity check** — pickle > 500 MB doubles in RAM under any
   plate-drop / row-filter `.copy()` path. Common cause: forgetting to
   pre-filter `df_raw` to the calibration genes (a 19 M-row proteomics
   dump pickled whole is ~3.6 GB; filtering to 111 genes brings it to
   ~60 MB — see the gotchas below).
3. **Two cheap end-to-end runs** with small RF (`n_estimators=20,
   max_depth=5`) on a 5-gene subset of the calibration set:
     - `__smoke_no_drop` — no plate filter (baseline path)
     - `__smoke_with_drop` — plate-drop filter ON (the
       `.isin().copy()` path that's the usual OOM offender)
4. **Schema check** — every rec must have `mean_r2`, `median_r2`,
   `n_genes_evaluated`, `metric`, `metric_name`. `n_genes_evaluated` must
   be > 0 (otherwise the per-gene CV loop is silently skipping every gene
   due to a min_compounds / label-col / feature-merge bug).
5. **Peak RSS measurement** via `resource.getrusage(RUSAGE_SELF).ru_maxrss`
   — flag if > 8 GB on the *cheap* RF (real RF with `n_estimators=100`,
   `n_jobs=8` will exceed this).
6. **Time budget extrapolation** — multiply cheap-RF runtime by ~5×
   (full-RF cost) × 3 (candidates per iteration) × max_iterations (5) to
   get the projected wall-clock. If it exceeds `stop_conditions.wall_clock_hours`,
   warn before launching.

The autoresearch loop is **not safe to start** until the smoke test
prints "SMOKE TEST OK" with no warnings. The bundled implementation in
`autoresearch/optimizeMS_genes_R2/run_one.py:smoke_test_loop_prereqs()`
encodes all six checks; copy it when scaffolding a new project.

Smoke-test results are intentionally **not** logged to
`autoresearch.jsonl` and **do not** update `best.json` — they're
pre-flight checks, not real experiments.

### Step 7 — Cache inputs

From inside `baseline.ipynb`, after the data-loading cells have run:

```python
from autoresearch.<project>.run_one import cache_inputs
cache_inputs({
    'df_raw_pristine': df_raw,        # un-filtered; drops happen inside run_one
    'MF_features':     MF_features,
    'genes':           genes,
})
```

This writes `logs/inputs.pkl`. Every iteration of the loop reads this pickle
instead of rebuilding features. Re-run this cell whenever upstream data
changes (new proteomics dump, new chemlib, different gene list).

### Step 8 — Bootstrap the loop's state files

Three small files have to exist before the agent can iterate. Claude can
write all three for you on request — the inputs are `config.yaml`'s
`propose_hypotheses` block plus the metric/threshold knobs.

**`logs/best.json`** — the champion file, seeded empty:
```json
{
  "metric_name":           "mean_r2",
  "metric":                null,
  "improvement_threshold": 0.005
}
```

`metric_name` and `improvement_threshold` come straight from
`config.yaml`'s `optimization_metric` and `improvement_threshold`. `metric`
is `null` so the first run unconditionally promotes.

**`candidates.md`** — the flat experiment queue, seeded with iteration-1
entries derived from each seed hypothesis. One-line tier-organised checklist:

```markdown
## Tier 1 — H1 (plate noise)
- [ ] **t1_drop_lopo_majority** — drop plates with LOPO R² < 0 across ≥ 50% of
       calibration genes. Cache: inputs_drop_lopo.pkl.

## Tier 3 — H2 (model swap)
- [ ] **t3_xgb_baseline** — XGBRegressor with sensible defaults on the same
       features and label as the RF baseline.
```

**`logs/hypotheses.md`** — the hypothesis evolution log, seeded with the
two starting hypotheses from `propose_hypotheses` and an empty iteration-1
section ready for the agent to fill in:

```markdown
# Hypotheses log

## Iteration 1 — seeded from config.yaml `propose_hypotheses`

### H1 — plate noise hurts cross-target predictions
Test: drop_plates derived from a global rule (no per-gene drops).
Verdict: pending.
Spawned: pending.

### H2 — current model is mis-calibrated; another estimator may do better
Test: swap RandomForestRegressor for XGBRegressor / LGBMRegressor on the
same features + label.
Verdict: pending.
Spawned: pending.
```

The agent writes the verdict + the 3 spawned hypotheses at the end of
iteration 1, then iteration 2 starts from those.

### Step 9 — Paste the prompt and walk away

Copy `config.yaml`'s `system_prompt` block (everything between `system_prompt: |`
and the next top-level key) into the VS Code Claude chat as the `/loop`
input. The loop self-paces and stops on whichever `stop_conditions` fires
first.

---

## File-by-file

### `autoresearch/run_one.py` — the harness

The single source of truth for "how do I run one experiment?". Public API:

| Function | Purpose |
|---|---|
| `run_one(ML_data, change)` | Apply `change` to `ML_data`, run K-fold CV, return a metric dict. Pure function. |
| `cache_ML_data(ML_data)` / `load_ML_data()` | Pickle round-trip so the agent doesn't recompute features every iteration. |
| `append_log(rec)` | Append one JSON line to `logs/autoresearch.jsonl`. |
| `load_best()` | Read `logs/best.json`. |
| `update_best_if_improved(rec)` | Promote `rec` to champion iff `rec[metric] - best[metric] > improvement_threshold`. |

`change` schema:

```python
{
    'id':       'unique_change_id',           # required, used for log dedup
    'desc':     'one-line human description',
    'model':    {'cls': 'XGBClassifier',
                 'params': {...}},            # supported cls list in run_one.py
    'features': 'all' | [col1, col2, ...],    # which feature columns to keep
    'cutoff':   0.5,                          # decision threshold for f1/pred_y
    'folds':    5,
    'notes':    'optional free text'
}
```

`rec` schema (also one line in `autoresearch.jsonl`):

```python
{
    'id', 'desc', 'ts', 'duration_s',
    'n', 'n_pos', 'n_features', 'folds', 'cutoff',
    'metric', 'metric_name',                  # the optimisation target
    'roc_auc', 'pr_auc', 'f1', 'mcc',         # full metric panel for analysis
    'model', 'features',                      # echoed for reproducibility
    'notes'                                   # if present in change
}
```

Two ways to invoke:

```python
# Python API (preferred — what /loop will use)
from autoresearch.run_one import run_one, append_log, update_best_if_improved
rec = run_one(ML_data, change)
append_log(rec)
update_best_if_improved(rec)
```

```bash
# CLI (handy for shell-driven loops, cron, etc.)
echo '<change.json>' | python autoresearch/run_one.py
```

### `autoresearch/candidates.md`

A markdown checklist, tiered cheapest → most expensive. The agent picks the
first unchecked entry, runs it, marks it `[x] → logged as <change_id>`. Order
matters — early wins seed the champion before tokens are spent on long-shots.

When extending the candidate list, write entries that translate directly to a
`change` dict:
> `[ ] **t1_spw_3** — XGB scale_pos_weight=3, default everything else.`

becomes
```python
{'id': 't1_spw_3', 'desc': 'XGB scale_pos_weight=3...', 'model': {...}}
```

### `autoresearch/logs/best.json`

The current champion. Schema:

```json
{
  "metric_name": "pr_auc",
  "improvement_threshold": 0.005,
  "metric": 0.412,
  "change_id": "t1_spw_6",
  "desc": "XGB scale_pos_weight=6",
  "model": { "cls": "XGBClassifier", "params": { "..." : "..." } },
  "features": "all",
  "cutoff": 0.5,
  "folds": 5,
  "updated_at": "2026-04-28T22:31:04"
}
```

`improvement_threshold` is the noise band — set it just above the run-to-run
variance you observe when re-running an identical change. Anything smaller
will declare every random fluctuation a new champion and you'll chase noise.

### `autoresearch/logs/autoresearch.jsonl`

Append-only. Every iteration writes one line. This is your post-hoc analysis
substrate: `jq` over it to find regressions, plot metric vs time, audit which
candidate was actually run, etc.

### `autoresearch/loop_prompt.md`

The verbatim `/loop` prompt to paste into the VS Code Claude chat. Includes:

1. The goal / harness / log / stop-condition spec.
2. A bootstrap snippet to cache `ML_data` once.
3. A "verify the harness on a known-good baseline" snippet.
4. Monitoring one-liners (`tail -f`, `watch + jq`).

---

## End-to-end procedure

1. **Build `ML_data` once** in your notebook (the expensive feature build).
2. **Cache it** so the loop doesn't redo this every iteration:
   ```python
   from autoresearch.run_one import cache_ML_data
   cache_ML_data(ML_data)
   ```
3. **Sanity-check the harness** with a known-good baseline (the snippet at the
   bottom of [autoresearch/loop_prompt.md](../autoresearch/loop_prompt.md)).
   Confirm `rec` looks right and `best.json` got populated.
4. **Paste the `/loop` prompt** from `loop_prompt.md` into the VS Code Claude
   chat. The agent self-paces.
5. **Open a side terminal** and tail the log:
   ```bash
   tail -f autoresearch/logs/autoresearch.jsonl | jq '{id, metric, duration_s}'
   ```
6. **Walk away.** The loop stops on its own (8 consecutive non-improvements,
   12 hours, or candidates exhausted).

---

## Operational basics — starting, monitoring, stopping

### How to know whether a loop is currently running

Without an explicit signal it's easy to lose track. Three ways to check:

| Signal | Where to look |
|---|---|
| **Right after `/loop`** | The chat prints `Next wakeup scheduled for HH:MM:SS (in Ns)` when the loop reschedules itself. **No such line = loop will not fire again.** |
| **Future log activity** | A live loop appends a new JSON line to `autoresearch/logs/autoresearch.jsonl` per iteration. If the file's mtime keeps creeping forward without you touching anything, the loop is active. |
| **File mtime check** | `ls -la autoresearch/logs/autoresearch.jsonl` — last write time tells you when the most recent iteration completed. |

A live tail in a side terminal is the cheapest continuous monitor:
```bash
tail -f autoresearch/logs/autoresearch.jsonl | jq '{id, metric, duration_s}'
```

### How to stop a running loop

| Option | When to use |
|---|---|
| **Type `/loop stop`** | Cleanest. Explicit signal that the next firing should not reschedule. |
| **Reply with `stop the loop` (or any message)** | Works because each iteration includes a turn — your reply pre-empts the next `ScheduleWakeup`. |
| **Close the VS Code Claude chat** | Loops in the extension die with the conversation. Hard kill. |

### How to know the agent is actually pacing itself responsibly

The `/loop` skill in this repo's setup uses `ScheduleWakeup`, not a Monitor.
Each turn ends with one wakeup queued. Multiple wakeups stacking up is a bug —
the agent should always announce the schedule (`Next wakeup scheduled for...`)
so you can spot accidental doubles in the chat scrollback. If you don't see
the announcement, no wakeup is pending.

### Smart per-iteration wake-ups (event + fallback)

When iterations run >5 min — common once you stack heavier candidates
(MAE-split RF, n_estimators=300, ChemProp) — fixed `ScheduleWakeup`
delays become a bad trade-off: too short and you wake up before the bg
python finishes (race condition on log files), too long and you waste
compute idling. The fix is the standard "event + fallback heartbeat"
pattern from the /loop spec:

**Three-process pattern per iteration:**

1. **Run the experiments in the background** (`Bash run_in_background=True`)
   via the project's `run_iter_with_progress(ML_inputs, changes)` helper.
   This wraps the change list in a `tqdm` bar, calls `run_one + append_log
   + update_best_if_improved` per change, and emits one summary line to
   stdout per experiment. The bg output file is human-monitorable in
   real time.
2. **Arm a watcher** via a second `Bash run_in_background=True` doing
   `until grep -q '"id": "<last_change_id>"' $LOG_FILE; do sleep 5; done`.
   When the LAST experiment's change_id appears in the JSONL the loop
   exits → you get a `task-notification` within seconds of compute
   completion. (Per the Monitor docs, single completion notifications
   are Bash-with-until's job, not Monitor's; Monitor is for
   per-occurrence streaming.)
3. **Schedule a fallback** `ScheduleWakeup(delaySeconds = predict_iter_runtime(n))`
   using `predict_iter_runtime` from `run_one.py` — predicts conservatively
   from the median `duration_s` of prior runs × `n_experiments` × 2× slow
   factor + 60 s grace. The fallback only fires if the bg python crashes
   silently before writing the last change_id.

**Why three processes (not just one):** if the experiment runner crashes
mid-iteration, the watcher's `until grep` will spin forever (no last
change_id ever lands) and only the fallback wake-up rescues you. If the
experiment runner finishes happily, the watcher exits within ~5 s of the
last log line. Both paths converge on "agent woken up at the right time".

### In-turn iteration vs scheduled iteration

There's a tension worth flagging: the `/loop` spec says "schedule a wakeup at
the end of each turn", but if your iterations are 5 s and you're actively
watching, **in-turn iteration (no wakeups, just keep running)** is much more
efficient — every wakeup forces a prompt-cache miss and re-loads the
conversation context. For autoresearch specifically:

- **Use scheduled wakeups** when iterations are slow (>5 min) or when you
  want the loop to run unattended overnight.
- **Use in-turn iteration** ("power through") when iterations are fast and
  you're watching live. Tell the agent explicitly: *"power through; don't
  reschedule"* — otherwise it'll default to the spec.

In this repo the typical iteration is 2–5 s, so in-turn is usually the
right call when you're driving interactively.

---

## Triangulating with multiple evaluators

A single metric (e.g. `5-fold roc_auc`) can mislead. The autoresearch loop
optimises against `best.metric` faithfully, but if that metric is leaky, the
loop happily climbs an artefact.

The minimum I'd recommend for any classification task with chemical or
temporal structure: **three complementary evaluators**, run after the loop
has produced a candidate champion.

| Evaluator | What it isolates | When to trust |
|---|---|---|
| **Random K-fold** | Total predictive power, accepting structural leakage. | Comparing model variants on equal footing. |
| **Scaffold K-fold** (Bemis-Murcko + `GroupKFold`) | Generalisation across novel scaffolds. | Reporting "novel chemistry" performance to stakeholders. |
| **Temporal split** (train `date < dt`, test `date >= dt`) | Robustness to data drift over time. | The honest "predict next batch" number. |

The gap between these tells you something specific:

- **Random ≫ Scaffold**: scaffold leakage in K-fold is real. Move to scaffold-CV for the loop's metric.
- **Scaffold ≫ Temporal**: temporal drift dominates — the chemistry, assay, or library composition has shifted. No model architecture can rescue this; you need newer training data or an applicability-domain check.
- **All three close**: the model's K-fold metric is honest. Trust it.

This isn't a `/loop` candidate per se — these evaluators are *audits*, run
once on the champion, **logged as `EVAL ONLY` in `autoresearch.jsonl` so they
never update `best.json`** (which would corrupt the leaderboard with
non-comparable metrics).

In this repo the FBXO31 run produced:
```
                        random K-fold   scaffold K-fold   temporal split
champion (2221 feat)      0.6969          0.6842          0.5883
smaller  (173  feat)      0.6942          0.6828          0.6128 ← deployed
```
The temporal column is what changed the deployment decision; the loop's
K-fold optimisation alone would have shipped the larger, more brittle model.

---

## List-consumer vs list-extender vs hypothesis-cycle

There are now three valid loop modes in this repo, with increasing ceilings
and decreasing predictability:

| Mode | Behaviour | Reliability | Ceiling |
|---|---|---|---|
| **List-consumer** | Pick next `[ ]` entry, run, mark `[x]`, stop. | Predictable, easy to audit. | Capped at the quality of your hand-curated `candidates.md`. |
| **List-extender** | After each result, if it's promising, propose 1–2 derived experiments and insert them into `candidates.md`. | Less predictable, more token-hungry. | Can find combinations a human didn't pre-enumerate. |
| **Hypothesis-cycle** | Each iteration is a 3-step cycle: READ active hypotheses → TEST each as a focused `change` → PROPOSE N novel hypotheses for the next cycle. State lives in `hypotheses.md`. | Least predictable; most token-hungry. | Can shift the *framing* of the problem mid-loop, not just the parameters. |

The default `loop_prompt.md` in this repo is **list-extender** — step 5 of
the per-iteration block explicitly invites the agent to synthesise derived
candidates when it sees a pattern. This is the same shape as Karpathy's
original autoresearch (the agent maintains a running idea queue based on
what it's learned), and it's what produced the +0.0060 win on the FBXO31
task — `t2_morgan_plus_maccs` was a derived experiment, not a pre-listed
one.

**If you want strict reproducibility**, delete step 5 from your prompt and
the loop becomes a list-consumer. The result for the same `candidates.md`
will be identical run-to-run.

**If you want compounding gains**, keep step 5 and accept the unpredictability.
The trade-offs to be aware of:

- The agent will burn tokens proposing experiments that may not pan out.
- The same `candidates.md` can produce different end states depending on
  what the agent observed and synthesised. Keep `autoresearch.jsonl` for
  audit (it captures everything that *was* run).
- Cap synthesis explicitly in the prompt (we cap at ≤ `novel_hypotheses_per_iter`
  derived candidates per iteration and ≤4 source representations per stack)
  to prevent combinatorial blow-up and noisy mega-stacks.

**If you want hypothesis-driven exploration**, switch to the cycle mode
(used by `optimizeMS_genes_R2/`). Each iteration emits a verdict
(VALIDATED / INCONCLUSIVE / REFUTED) per active hypothesis and spawns
exactly `novel_hypotheses_per_iter` (e.g. 3) new ones for the next cycle.
The agent must:

- Maintain `hypotheses.md` as the iteration-by-iteration evolution graph.
- Use WebSearch / WebFetch (no project data in queries) to ground novel
  hypotheses in literature, not just permute the seeds.
- Write a final summary section at iteration `max_iterations` with the
  champion config, top validated/refuted hypotheses, and a one-paragraph
  recommendation.

The cycle mode is the right choice when the **framing** of the problem is
itself uncertain (you don't yet know whether plate noise, model choice,
label transform, or data composition is the dominant lever). List-extender
shines when the framing is clear and you want compounding tactical wins
inside a known design space.

The "How to make this happen reliably" note in the prompt:

```
After running each candidate, if its result is in the top 3 so far OR
clears the improvement_threshold, propose 1-2 derived experiments
combining it with previous winners (feature stacks, model swaps, simple
param tweaks). Insert them as new [ ] entries near the top of the
appropriate tier in candidates.md before continuing. State the
hypothesis in one short line.
```

This converts the loop from "list-consumer" to "list-extender" and is what
produces the kind of result you saw in this repo (a champion that wasn't
on the original candidate list).

---

## Adapting this template to a different problem

This is now driven by the per-project `config.yaml` workflow described in
the [Per-project workflow](#per-project-workflow-configyaml-driven) section
above. The condensed checklist:

1. **Scaffold a project folder** — `autoresearch/<project>/` with
   `logs/` and a `baseline.ipynb` showing the canonical pipeline.
2. **Fill in the human side of `config.yaml`** — `research_objective`,
   `propose_hypotheses` (seed list), the loop knobs (`number_of_iterations`,
   `novel_hypotheses_per_iter`, `optimization_metric`, `improvement_threshold`,
   `stop_conditions`), and the DOs/DONTs.
3. **Pick `improvement_threshold` empirically** — run `run_one` 3–5 times
   with the same `change` once the harness exists. The spread is your noise
   floor; set the threshold ~3× that. Anything smaller chases noise.
4. **Ask Claude to write the `system_prompt`** — it reads the human-side
   fields + `baseline.ipynb` + `template/` and emits the operational
   prompt. Iterate on it like any other artifact.
5. **Adapt `template/run_one.py`** to your metric. Keep the public API
   (`run_one`, `append_log`, `load_best`, `update_best_if_improved`,
   `cache_inputs` / `load_inputs`) unchanged — only `_evaluate` and the
   metric-name default change.
6. **Decide the loop mode** — list-consumer / list-extender /
   hypothesis-cycle (see [List-consumer vs list-extender vs
   hypothesis-cycle](#list-consumer-vs-list-extender-vs-hypothesis-cycle)).
   Cycle mode is the new default for projects with uncertain framing; the
   `system_prompt` Claude writes will reflect whichever you've picked.
7. **Smoke-test once**, cache inputs, paste `config.yaml`'s `system_prompt`
   into the `/loop` chat.

A non-ML example: optimising a prompt template against a held-out test set.
`run_one` calls the model, scores the output with an LLM-judge or regex, and
returns `{'pass_rate': ...}`. `metric_name = 'pass_rate'`. Same project
folder layout, same config.yaml schema, same hand-off to Claude for the
system_prompt.

### Why config.yaml as the source of truth?

The split — human writes objectives, agent writes prompt — keeps the
*decisions* in human-editable YAML and the *operationalisation* in
agent-editable prose. When you tweak `improvement_threshold: 0.005 → 0.01`
six weeks later, you don't have to re-read 200 lines of prompt to find the
number; it lives at the top of the config. Conversely, when the prompt
needs to be sharpened (clearer DON'T, better translation of a constraint),
you ask Claude to revise just that section without touching the canonical
objectives.

---

## Choosing harness vs running surface

| Situation | Where to run the loop |
|---|---|
| Watching one or two iterations | VS Code Claude extension |
| Running overnight, single machine | Terminal in `tmux`, same `/loop` prompt |
| Running on a remote server / no laptop dependency | SSH + `tmux`, or `cron` of the CLI mode |
| One-off "run experiment X tomorrow at 2am" | `/schedule` instead of `/loop` |

Don't run a long loop in the VS Code extension on a laptop you'll close —
the conversation dies with the IDE process. For >1 hour runs, port the same
prompt to a terminal `tmux` session.

---

## Gotchas / lessons learned setting this up

- **Cache the dataset.** Without `ML_data.pkl`, every iteration re-runs the
  feature build (15 s+ for our case). Iterations balloon and the agent loses
  iterations to overhead instead of experiments.
- **Pre-filter the cache to what the loop actually uses.** The
  `optimizeMS_genes_R2` loop OOM-killed itself on iteration 1's first
  plate-drop experiment because the cache was the full 19 M-row proteomics
  dump (3.6 GB pickle); the harness's `df_raw[~df.MSPlate.isin(...)].copy()`
  doubled it in RAM and then RF's `n_jobs=8` forks pushed past the WSL2 RAM
  cap. Fix: filter `df_raw` to the calibration genes BEFORE pickling
  (~60 MB in our case, 100× smaller). Run the **Step 6 smoke test on the
  real cache** to surface this before launching a multi-hour loop —
  sandbox tests on a 3-gene subset can't see this class of bug.
- **`pyarrow` may not be installed.** We use `pickle` for the cache to avoid a
  hard dep. Don't switch to `parquet` unless you've confirmed the env has it.
- **Out-of-order Jupyter execution.** If you re-run a cell that builds a
  list-of-dataframes, the variable shadows the concatenated DataFrame. Keep
  the build + concat in a single cell to avoid this class of error.
- **Don't sleep the laptop.** The Claude extension's conversation lives in
  the IDE process; sleep = loop death. Use `caffeinate` (macOS) /
  `systemd-inhibit` (Linux) / PowerToys Awake (Windows).
- **Token cost is linear.** Self-paced cadence + 5 s iterations ≈ 50–100
  model calls/hour. Set explicit stop conditions; don't rely on willpower.
- **Determinism is everything.** A non-deterministic harness will report
  "improvements" that are pure variance. Fix every seed: `random_state`,
  `numpy.random`, `xgboost`, K-fold splits.
- **Human-curated candidates beat blind search.** The first ~10 entries in
  `candidates.md` should be things you'd manually try anyway. Save Bayesian
  optimisation for after the agent has eaten the obvious wins.
- **In list-extender mode, the agent generates new candidates from observed
  patterns.** Cap synthesis (≤2 derived per iteration, ≤4 source reps per
  stack) to prevent combinatorial blow-up. If you don't cap, expect a stack
  of 8 fingerprints by hour two and a noisy regression.
- **Don't mix loop modes mid-run.** Switching from list-consumer to list-extender
  partway through a run makes the leaderboard hard to interpret — restart with
  one mode chosen up front.
- **Improvement threshold > 0.** Without it, every random fluctuation
  becomes a "new champion" and the loop chases noise forever.
- **Sanity-check too-good-to-be-true metrics.** A single 1.0 AUC, MCC = +1.0,
  or perfect F1 is almost always a bug, not a breakthrough. We hit this
  with ChemProp: the predictions CSV included the input ground-truth `label`
  column unchanged, and our column-picker grabbed it instead of the actual
  prediction column → "perfect" AUC was just `label` vs `label`. Fixes:
  (a) pass `--drop-extra-columns` to the prediction tool when the API
  supports it, and (b) match prediction columns by *name pattern*
  (`pred_*` / `*_pred` / `*_prob`) rather than by position. **Implemented
  in [autoresearch/chemprop_eval.py](../autoresearch/chemprop_eval.py)** —
  use that as a template if you wrap any other CLI-based predictor.
- **Look at *all* metrics in `_evaluate`, not just the optimisation target.**
  Patterns across roc_auc / pr_auc / f1 / mcc are diagnostic. If `roc_auc`
  improves but `mcc` flips sign, you're optimising the wrong thing or the
  threshold is mis-tuned.

---

## File pointers

**Per-project workflow (current default):**
- Project config: `autoresearch/<project>/config.yaml`
  (e.g. [autoresearch/optimizeMS_genes_R2/config.yaml](../autoresearch/optimizeMS_genes_R2/config.yaml))
- Project harness: `autoresearch/<project>/run_one.py`
- Hypothesis log: `autoresearch/<project>/logs/hypotheses.md`
- Candidate queue: `autoresearch/<project>/candidates.md`
- Run log: `autoresearch/<project>/logs/autoresearch.jsonl`
- Champion: `autoresearch/<project>/logs/best.json`

**Canonical templates** to copy when scaffolding a new project:
- Harness template: [autoresearch/template/run_one.py](../autoresearch/template/run_one.py)
- Candidates template: [autoresearch/template/candidates.md](../autoresearch/template/candidates.md)
- Loop-prompt template: [autoresearch/template/loop_prompt.md](../autoresearch/template/loop_prompt.md)

**Concept references:**
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- [Claude Code port](https://github.com/drivelineresearch/autoresearch-claude-code)
