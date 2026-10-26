# How to use a project wiki with Claude Code

This guide tells you how to set up a project wiki for Claude Code. The wiki gives Claude a
memory of your repository. Claude reads the wiki at the start of each session. Then Claude
continues the work from the last session, and you do not have to explain the project again.

We use this system in `MS_ML`. Our wiki is [`wiki/wiki.md`](../wiki/wiki.md), and our
instructions are in [`CLAUDE.md`](../CLAUDE.md).

---

## 1. The problem

Claude Code does not keep a memory between sessions. In a long session, Claude also
compacts (summarizes) the old part of the conversation, and the details go away.

Without a wiki, you must tell Claude the state of the project in each session. Claude must also
read the code and the notebooks again to find the findings and the decisions.

The wiki is one Markdown file in the repository: `wiki/wiki.md`. It holds the settled state of
the project: the findings, the decisions, and the locations of things. Claude reads it first, and
Claude updates it while it works.

---

## 2. How it works

Three parts make the system:

| Part | File | Function |
|---|---|---|
| The wiki | `wiki/wiki.md` | Holds the project state. Use one wiki for each repository. |
| The instruction | `CLAUDE.md` in the repository root | Tells Claude to read the wiki and to keep it current. Claude Code loads this file automatically at the start of each session. |
| Git | — | Keeps the history of the wiki and shares it with your team. |

The cycle in each session:

1. Claude Code loads `CLAUDE.md`.
2. `CLAUDE.md` tells Claude to read `wiki/wiki.md`.
3. Claude reads the wiki and knows the current state of the project.
4. You work with Claude as usual.
5. When a result or a decision becomes final, Claude updates the wiki.
6. Claude tells you about the update in one line, for example: *"Wiki updated: added the
   temporal-split result to Findings."*
7. At the end of the session, Claude does a reconcile pass on the full wiki.

---

## 3. Set up the wiki

Do these steps one time for each repository.

1. Make the folder `wiki/` in the repository root.
2. Copy the template from [section 7](#7-template-for-wikiwikimd) into `wiki/wiki.md`.
3. Copy the text from [section 6](#6-text-to-copy-into-claudemd) into `CLAUDE.md` in the
   repository root. If `CLAUDE.md` does not exist, make it.
4. Change the "aggregate only" line in that text to the sensitive data of your project.
5. Commit the two files to git.
6. Start a new Claude Code session in the repository.
7. Tell Claude: *"Read the repository and fill in the wiki."*
8. Read the first version of the wiki and correct the errors.

---

## 4. The structure of the wiki

Our wiki has six sections. Each section has a different function.

| Section | What it holds | Example entry |
|---|---|---|
| **Where we are now** | The current focus and the open threads: work in progress, work that Claude offered but did not do, and decisions that wait for you. | "Open: temporal-split audit of the classifier. Not started." |
| **Findings** | The conclusions of analyses. Each finding has numbers, a caveat, and a source. | "Model X: ROC AUC 0.80 on random 5-fold CV, 0.59 on a temporal split. *[cell c7b6e7c4; logs/summary.md]*" |
| **Decisions & conventions** | Parameters, definitions, approaches, and names. Each decision has its reason and, if necessary, how to revert it. | "Storage format = parquet, not CSV: 5–10x smaller, exact dtypes. *[config DFRAW_PATH]*" |
| **Where things live** | The paths to scripts, notebooks, config, data, outputs, and trained models. | "Config: `config/config.yaml`." |
| **Reusable functions** | The helpers that already exist. Claude uses them and does not write a new copy. | "`run_K_Fold_Xval_Classification(...)`: random K-fold CV." |
| **Log** | One dated line for each important session. The newest line is at the top. | "2026-08-11: moved the data loading into `python/MS_build_ML.py`." |

When an open thread is complete, mark it ~~with strikethrough~~ and **DONE (date)**. Then move
the result to Findings or Decisions.

The "Reusable functions" section prevents a frequent problem. Without it, Claude often writes a
new function that does the same work as a function in your shared library.

---

## 5. The rules for an entry

Claude obeys these rules when it writes to the wiki. Obey them also when you edit the wiki.

- **Record settled state only.** Write a finding when an analysis gives a conclusion. Do not
  record each message, each edit, or each intermediate run.
- **Reconcile. Do not only append.** When a new result contradicts an old entry, correct the
  old entry. Mark a replaced plan as superseded and point to the new entry.
- **Add the date and the source.** Put the source in square brackets at the end of the entry:
  the cell id, the output file, or the config key. Example: `_[cell 9b42a6ce; config SIG_K]_`.
- **Write aggregate data only.** Write counts, metrics, gene names, parameter values, and code
  names. Do not write SMILES, compound IDs, or values for one compound.
- **Give a one-line notice.** After each update, Claude tells you in one line what it changed.
- **Do not ask permission.** Claude updates the wiki without a question. You read the notice,
  and you correct the entry if it is wrong.

### When Claude updates the wiki

| Claude updates the wiki when... | Claude does not update the wiki for... |
|---|---|
| A finding becomes final. | Each message or each code edit. |
| A decision or a definition changes: a parameter, a cohort, a name, or a change of approach. | Each intermediate run. |
| A reusable artifact becomes stable: a new helper function, a new config key, or notebook code that moved to a `.py` file. | An exploratory question. |
| The session ends (reconcile pass). | |
| You say *"update the wiki"*. | |

Small updates during the session are important. A compaction can occur at any time, and an
update that waits for the end of the session can go away.

---

## 6. Text to copy into CLAUDE.md

This is the section from our `CLAUDE.md`. Copy it to the top of your `CLAUDE.md`.

```markdown
## Project Wiki — read this first

**At the start of every session, read `wiki/wiki.md`** — the durable, aggregate memory of this
repo (where we are, what we've concluded, decisions/conventions, where things live). It survives
context compaction and new sessions; use it to pick up exactly where we left off instead of
re-deriving state from the notebook. **One wiki per repository:** `wiki/wiki.md`.

**Claude updates it autonomously (then notes it in one line), incrementally, so it's always
current** — that, not a last-second flush, is what survives compaction. **Never ask permission
to update the wiki — just do it and give the one-line notice.** Triggers:

- A **finding settles** — an analysis reaches a conclusion worth remembering next session.
- A **decision/definition is made or changed** — a parameter, cohort definition, approach, naming, pivot.
- A **reusable artifact stabilizes** — a new `functions.py` helper, a config knob, a notebook section moved to `.py`.
- On **session wind-down**, do a reconcile pass; update on **explicit request** ("update the wiki").

Do **not** update on every message / edit / intermediate run / exploratory question — record
*settled* state, not the play-by-play.

When writing: **reconcile, don't blindly append** (fix anything new results contradicted); add the
**date + provenance** (cell id / output file / config key); **aggregate only** — findings, counts,
gene names, gene-sets, metrics — **never** SMILES / compound IDs / per-compound values; then give a
**one-line notice**.
```

---

## 7. Template for wiki/wiki.md

Copy this template into `wiki/wiki.md`. Replace the text in angle brackets.

```markdown
# <PROJECT> — project wiki
_Updated: YYYY-MM-DD. The memory of this repository. Claude reads it at the start of each session. Aggregate data only._

## Where we are now
- **Focus:** <one line: the current work>
- **Open threads:**
  - <work in progress>
  - <work that Claude offered but did not do>
  - <decisions that wait for an answer>

## Findings
- **<The conclusion, in bold>.** <numbers + caveat>. _[cell id / output file / config key]_

## Decisions & conventions
- **<The decision>** (YYYY-MM-DD): <what + why + how to revert>. _[config KEY]_

## Where things live
- **Config:** `config/config.yaml`
- **Helpers:** `python/functions.py`
- **Notebooks:** `vignettes/<name>.ipynb`
- **Key outputs:** `output/<...>`

## Reusable functions (use these first — do not write a new copy)
- `module.function(args)` — <what it does, in one line>

## Log
- YYYY-MM-DD — <one line: what changed in this session>
```

---

## 8. How to work with the wiki each day

- Start each session as usual. `CLAUDE.md` tells Claude to read the wiki, so you do not have to.
- After a pause, ask Claude *"Where are we?"*. Claude answers from the wiki.
- Read the one-line notices. If an entry is wrong, tell Claude, and Claude corrects it.
- Before you stop, tell Claude: *"We stop now. Reconcile the wiki."*
- Commit `wiki/wiki.md` together with the code that it describes. Then the git history shows
  when each decision changed.
- Read the full wiki at regular intervals. Tell Claude to remove the entries that are no longer
  true.

---

## 9. What we learned

- **Keep the wiki short.** Our wiki has 224 lines, and it is larger than one Claude `Read` call
  (25,000 tokens). Claude must then read it in parts, and each session uses more tokens. Keep
  each entry to a few lines. Put long explanations in `docs/` and link to them from the wiki.
- **Put the source on each entry.** Without the cell id or the file name, nobody can check the
  number later.
- **Write the reason for each decision.** Also write how to revert it. Then Claude does not
  suggest an approach that you rejected before.
- **Write the caveats with the numbers.** For example: *"random 5-fold CV, optimistic"*. A number
  without its caveat can give a wrong conclusion in the next session.
- **The wiki is not Claude's auto-memory.** Claude Code also has a personal memory in
  `~/.claude/projects/<repo>/memory/`. That memory holds your personal preferences and stays on
  your computer. The wiki holds the project state, it is in git, and your team shares it.

---

## 10. Data privacy

CAUTION: Claude Code sends each file that it reads to the Anthropic API. This includes the wiki
in each session. Git also copies the wiki to each clone of the repository. Do not write
confidential data in the wiki.

- Write aggregate results only: counts, metrics, gene names, parameter values.
- Do not write SMILES, compound IDs, structures, or values for one compound.
- Paths, file names, cell ids, and function names are safe.

---

## 11. Related parts of our setup

These parts are not part of the wiki, but they use the same `CLAUDE.md`. Copy them if you want
the full system.

- **Code style rules.** `CLAUDE.md` tells Claude to read `/home/gtamo/Simplify/rules.md` before it
  writes code.
- **Notebook read block.** A `PreToolUse` hook in [`.claude/settings.json`](../.claude/settings.json)
  stops Claude from reading `.ipynb` files. Notebook outputs can hold compound data.
- **Safe notebook edits.** [`python/nb_edit.py`](../python/nb_edit.py) lets Claude edit notebook
  cells without reading or clearing the outputs of the live notebook.
