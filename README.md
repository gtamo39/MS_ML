Machine Learning on Proteomics Data to guide library expansion
----------------------------------------------------------------

let's go through the Jupyter Notebook located in vignettes/main.py

### To go through the notebook report:
---------------------------------------

* install your virtual env with python: `python3 -m venv ml` then `source ml/bin/activate`
* OR with conda (recommended) `conda create -n ML python=3.12.3` then `conda activate ML`
  * **Use Python 3.12, not 3.13** — on 3.13 `pip install` tries to build `pyarrow` from source (no cp313 wheel for the pinned version) and fails with `ModuleNotFoundError: No module named 'pkg_resources'`.
* install dependencies: `pip install -r requirements.txt` (needs `git` on PATH — `descriptastorus` is pinned to a git commit, see below)
* install Vina and OpenBabel: `conda install -c conda-forge vina openbabel`
  * Conda-only — neither is on PyPI, so neither is in `requirements.txt`. Install them into the `ML` env, after `conda activate ML`.
  * Installed in `ML`: `vina` 1.2.7, `openbabel` 3.2.1 (`py312` builds — they must match the env's Python 3.12).
* `python -m ipykernel install --user --name ML --display-name "Python (ML)"`

### Chemistry features (`FEATURES_TYPE` in `config/config.yaml`)
---------------------------------------------------------------

The feature cell in `MS_TargetML` / `MS_ActivityClass` switches on `FEATURES_TYPE`:

| value | features |
|---|---|
| `prevalence` | H236 with only the Morgan bits present in >2 % of the library (the deployed models' set) |
| `H236` | the full H236 universe — 4,269 cols: Morgan `F0..F2047` + 6 physchem + `MACCS_0..166` + `AP_0..AP2047` |
| `H237` | H236 **plus** 200 descriptastorus RDKit2D descriptors as `DS_*` (4,469 cols) |
| `autoresearch` | features unpickled from an autoresearch run |
| anything else | Morgan bits + physchem properties |

`H237` = `rdkit_tools.compute_H237_features(df)`, which reuses `compute_H236_features` and appends the
descriptor block. Descriptors are **CDF-normalised to [0, 1]** by default (`normalized=False` for raw
values), and are prefixed `DS_` because descriptastorus emits names that would otherwise collide with
H236's own `TPSA` / `LogP` columns. `descriptastorus` is in `requirements.txt` but is **not on PyPI**, so
it is pinned as a git URL; it runs fully offline with no telemetry and no downloaded weights.

### Building the models (`python/MS_build_ML.py`)
-------------------------------------------------

Same classes drive the notebook and the CLI (mirrors `../Px_interface/python/Px_interface.py`):
`PARAMS` (YAML → attributes, Dropbox readers, MLTrail handle) → `DATA` (library, proteomics,
features) → `OUTPUT` (screens + deployable models). The module self-locates the repo root, so it
runs from anywhere. In a notebook:

```python
from python.MS_build_ML import PARAMS, DATA, OUTPUT
params = PARAMS('config/config.yaml'); params.load_params(); params.setup_dropbox(); params.load_registry()
data = DATA(); data.load_chemical_lib_df(params); data.get_contaminants_and_controls(params)
data.load_proteomics_data(params); data.compute_features(params)
output = OUTPUT()
```

**Register the deployable activity classifiers in MLTrail** — all three are `RandomForestClassifier(
n_estimators=200, class_weight='balanced')` on `MF_features`, differing only in the label: 5-fold CV
for the `roc_auc` / `pr_auc` metrics, refit on the whole library, then the `{model, feature_cols}`
bundle plus the `compound/smiles/label` training set go into the vault.

| flag | label | config key |
|---|---|---|
| `--save_non_silent` | active: `ndown > 0` | `NON_SILENT_MODEL_NAME` |
| `--save_single_low` | selective: `1 <= ndown <= 12` | `SINGLELOW_MODEL_NAME` |
| `--save_high` | high: `ndown > 25` | `HIGH_MODEL_NAME` |

```bash
python python/MS_build_ML.py --config config/config.yaml --save_non_silent --save_single_low --save_high
python python/MS_build_ML.py --config config/config.yaml --save_non_silent --comment 'H237, cohort minus 2026-08-13'
```

`--comment` stores a free-text note **on the version being written**, next to its metrics, so each
version can record what changed. Omit it and MLTrail leaves the field out.

```python
output.save_non_silent_model(data, params)                      # same thing from the notebook
output.save_single_low_model(data, params)
output.save_high_model(data, params)
output.save_single_low_model(data, params, model=my_clf)        # try another estimator
output.save_single_low_model(data, params, overwrite=True)      # CORRECT the latest version in place
```

The flags share one process on purpose — **never run two saves concurrently**: `Registry` reads the
whole vault JSON at construction and writes it back whole, so two handles racing on it collide on
`next_id` and the last writer wins.

The registry names come from the `Saved MLTrail models` block of `config.yaml` (`name=...` overrides
per call). Re-running **appends a new version to the same id** — matched on that name — so the trail
keeps the performance history; pass `--overwrite` only to replace a bad version. **Renaming a key
therefore starts a fresh entry** rather than versioning the old one, so keep it stable unless that is
what you want. `features_type` is taken from `FEATURES_TYPE`, and MLTrail re-derives the features
from SMILES at predict time — so the featurizer must exist there for that type (`MF_2048`, `H236`,
`H237`).

**Check what landed in the vault** — `mltrail` is a console script installed with the package, so
these work from any directory (add `--config` to hit a non-default vault):

```bash
mltrail --list                                    # id, date, experiment_name, measure
mltrail --search --experiment_name Px_activity_1_12_rf_H237         # find the id by name
mltrail --details --id <id>                       # every attribute of the latest version
mltrail --trail --metrics roc_auc --id <id>       # the metric across versions
```

**Per-gene logfc screen** (label = per-compound mean of `logfc` clipped at 0, 5-fold CV RF), written
to `GENE_SAR_OUT` one gene at a time and **resumable** — genes already in the file are skipped, so
delete it to force a full redo. ⚠ That skip is why a **changed cohort needs a new `GENE_SAR_OUT`
filename**: re-running into a file written under different settings keeps its rows and silently
mixes two datasets.

**Every config key this module reads.** The first two define the cohort and apply at import, so they
affect the screen and the classifiers alike; the last three only name the registry entries:

| key | effect |
|---|---|
| `EXCLUDE_DATES` | screen dates dropped from `df_raw` **and** `MS` (`DATA.drop_excluded_dates`) — in memory, the parquets keep them, so emptying the key restores them without a rebuild. The gene screen and the classifiers share one cohort |
| `CM2RM_PARTS` | which blacklists build `cm2rm`: any subset of `contaminants` / `control_compounds` / `fbx_independent`. `--cm2rm a,b` overrides it |
| `DROP_CM2RM` | `true` drops the `cm2rm` compounds from `df_raw` **and** `MS` at load (`DATA.drop_cm2rm`, after `drop_excluded_dates`) — in memory, the parquets keep them. Every downstream step inherits it, including `--save_non_silent` / `--save_single_low` / `--save_high`. `false` keeps them in the frames |
| `DERIVE_MS` | `true` recomputes `MS` `ndown` / `activity` / `ms_score` from `df_raw` (`DATA.derive_ms`, after `drop_cm2rm`), so the two frames always describe the same rows. The delivered values stay as `ndown_src` / `activity_src`. In memory, the parquets keep the delivered schema. `false` keeps the old `MS` |
| `PX_SOURCE` | where the FBX side comes from: `rds` reads the Px database (`RDS_*`) plus `PX_ARCHIVE_DIR`, dated by `PX_DATES` (and by CDD for the experiments it does not date, see `PX_DATES_FROM_CDD`); `dropbox` reads the tranche folders under `FBX_DIR`. Both give one entry per export date, so nothing downstream changes |
| `PX_DATES_FROM_CDD` | `true` (default): with `PX_SOURCE: rds`, a database experiment that `PX_DATES` does not date takes the CDD `Date` readout of its (batch, plate) pair, matched as for the source: the exact pair, then the `PwNN` plate in the name (`fn.fbx_dates`, `fn.cdd_pair_value`). The database has no date, and `PX_DATES` (Px_interface, 2026-10-21) dates only the experiments of the last Dropbox build, so the build skipped every newer experiment: on 2026-11-19, 823 got a CDD day, which brings 223 library compounds into `MS`. A `PX_DATES` day is the Dropbox export day of the plate. The CDD `Date` is a batch date: the CDD upload files give one value to a group of plates (2026-05-19 covers the plates of the tranches 2026-06-16 to 2026-07-14), so it is neither the screen day nor the export day. The Px database has no date at all, but it inserted the 2026-08-18 plates after the 2026-08-14 plates and before the 2026-10-06 plates, so that day sorts them in the right place. Each new day becomes a tranche (2026-11-19: `20260818` with 818 experiments, `20261007` with 4, `20260724` with 1). `false` skips these experiments (in Px_interface, a missing key also means false). A change needs `DFRAW_OVERWRITE: true` |
| `DROP_PLATE_SUBSTR` | plate-name substrings dropped from the FBX side, case-insensitive. Default `['MLN', 'KO', 'Eval']`; add `VM` / `BIND` to drop the validation conditions too |
| `EXCLUDE_SOURCES` | proteomics sources dropped from `df_raw` **and** `MS` on a rebuild, default `['BMS']`. A source is the CDD `MSData - Proteomics activities: Source`, upper-cased (`SERAC` / `BMS` / `EVOTEC`). The old tranches read it from their own metadata (`fn.load_proteomics_data`); the FBX side reads it from CDD by batch and plate (`fn.tag_fbx_sources`). It acts before the per-compound collapse, so a compound keeps its latest screen from a source that stays. It replaces the old `Source == 'SERAC'` rule, so `EVOTEC` stays. Both frames keep a `source` column (NaN: the FBX experiment has no CDD row). A change needs `DFRAW_OVERWRITE: true`; a cached load prints a note when the cache does not match |
| `CDD_PX_PROTOCOL` | the CDD protocol `MSData - Proteomics activities` (121861), read by `fn.cdd_px_sources` (GET only). Its `Source` readout gives the FBX `source` tags on a rebuild, and the `Px_Source` column of `CHEMLIB_PATH` when `CHEMLIB_COLUMNS` lists it. Its `Date` readout dates the database experiments that `PX_DATES` does not date (`PX_DATES_FROM_CDD`) |
| `CHEMLIB_COLLECTIONS` | `serac_df` keeps the cached vault molecules of these CDD collections, default `['AJ', 'AK']`, or of `CHEMLIB_SUBSERIES`. The filter acts when `DATA.load_chemical_lib_df` builds `serac_df`, so a change needs no new pull. A rebuild keeps only the `df_raw` / `MS` compounds of `serac_df`, so a change also changes that cohort |
| `CHEMLIB_SUBSERIES` | CDD `Subseries` values whose molecules join `serac_df` as well, e.g. `['AC']` (226 FBXO31 glues in no collection). `[]` = collections only |
| `CHEMLIB_COLUMNS` | the CDD Vault fields that `DATA.load_chemical_lib_df` pulls into `CHEMLIB_PATH` when `CHEMLIB_OVERWRITE` is `true`, in this order (`name` becomes `compound`). The pull reads **every vault molecule** in one walk (`fn.cdd_vault_df`, ~20K molecules, ~2.5 min), so the csv holds the whole vault, and `Collections` (the molecule's collection names joined by `;`, as in the CDD web export) and `Subseries` are always added for the filter. One name is not a CDD field: `Px_Source` is every proteomics source of the compound in `CDD_PX_PROTOCOL`, over all its batches, joined by `+` (e.g. `BMS+SERAC`; NaN = never screened). Remove it from the list to skip that pull. The library drops nothing. A cache without `Collections` (pulled before 2026-11-19) stays whole, with a note. The csv keeps one row per batch; `serac_df` keeps one row per compound, its last batch (highest `molecule_batch_identifier`, e.g. `-002` over `-001`). The cached csv keeps the fields of the last pull, so after you add or remove a name, run once with `CHEMLIB_OVERWRITE: true`. The code needs `name`, `molecule_batch_identifier`, `smiles`, the three `Px_*(yes/no)` fields and `Px_Target_interest`. A name that matches no CDD field gives an empty column and a `WARN` line from `fn.cdd_vault_df` |
| `VALIDATED_TARGET_FILE` | optional path to the curated glue list of Px_interface: a csv with the columns `genes` and `Batch Molecule-Batch ID`, one row for each validated gene and compound batch. When it is set, `fn.read_validated_cmp` reads it. Its genes replace the CDD-derived `validated_targets` (first token of `Px_Target_interest` for ligase-dependent compounds), as in Px_interface. Its rows become `validated_cmp`, with `compound` = the molecule id of the batch. In `Px_Universe`, `validated_cmp` gives the discovery-curve labels (`9092b849`), the t-SNE stars (`4d934984`) and the flavour table (`a9f41229`). Empty keeps the CDD list, and `validated_cmp` is then `None` |
| `NON_SILENT_MODEL_NAME` | MLTrail `experiment_name` for the `ndown > 0` classifier (`--save_non_silent`) |
| `SINGLELOW_MODEL_NAME` | MLTrail `experiment_name` for the `1 <= ndown <= 12` classifier (`--save_single_low`) |
| `HIGH_MODEL_NAME` | MLTrail `experiment_name` for the `ndown > 25` classifier (`--save_high`) |

⚠ A model name is also its version key: the same name appends a version to that id, a new name
starts a new id. There is no CLI flag for it — edit the key, pass a different `--config`, or use
`name=...` from the notebook.

```bash
python python/MS_build_ML.py --genes KDM1B,CIT           # a list, a file (one gene/line), or top:200 / all
python python/MS_build_ML.py --genes all --min_compounds 1000 --n_processes 8 --n_jobs 8
```

For the whole genome (~12k genes, ~17 h) run it detached:

```bash
screen -dmS gene_screen bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && cd /home/gtamo/MS_ML
  python python/MS_build_ML.py --genes all --min_compounds 1000 \
      --n_processes 8 --n_jobs 8 2>&1 | tee output/MS/gene_screen.log'
```

**`--n_processes 8 --n_jobs 8` is the measured optimum on this box, not a guess** — throughput
saturates on the memory subsystem, not on cores, so filling all 256 threads is *slower* (32×8 =
6.95 s/gene vs 8×8 = 5.81). Two gotchas the code guards against: sklearn's default loky backend
silently degrades to `n_jobs=1` inside a `multiprocessing` worker (hence the `parallel_config(
backend='threading')` wrapper), and `RandomForestRegressor` with `n_jobs>1` is not bit-reproducible
(~2e-16 from parallel `predict` accumulation), so re-runs won't reproduce the last digits of R².

A third one is **cosmetic and already silenced** in `_init_screen_worker`: `` `sklearn.utils.parallel.delayed`
should be used with `sklearn.utils.parallel.Parallel` ``. sklearn emits it when the `warnings.filters`
snapshot it captured was empty — and `_FuncWrapper` clears that *global* list on every task, so the
threading backend's 8 threads race into an empty snapshot constantly (123,720 lines / 39 MB in one
log). Unlike the loky warning it costs nothing: the config still propagates and the R² is identical.

### Pulling data files from Dropbox
-----------------------------------

The `config/config.yaml` path keys (`PATENTS_RAW`, `PX_*`, `DROPBOX_*`, `FBX_DIR`, `ENAMINE_*`, `STOCK_INVENTORY`) are WSL paths where the **laptop's Dropbox desktop client** keeps files synced (`/mnt/c/.../Serac Biosciences Dropbox/...`). From a machine without the client (e.g. the cluster), two backends can read them — pick one with **`DROPBOX_BACKEND`** in `config.yaml`. Both stream into memory; nothing is written to the local disk. The notebooks' params cell binds `_dbx` / `_fbx_open` / `_fbx_glob` to whichever backend is selected, so call sites are identical either way.

#### `rclone` (default — no laptop, no tunnel)

`functions.open_rclone` / `pull_rclone` / `glob_rclone` / `push_rclone` go straight to Dropbox through a pre-authenticated `rclone` remote, so they work with the laptop off. Paths stay exactly as in `config.yaml`: `_rclone_target` strips `DROPBOX_LOCAL_ROOT` and prefixes `DROPBOX_REMOTE`.

```bash
rclone config                       # one-time: create a `dropbox:` remote
rclone config file                  # then add root_namespace to that remote (see below)
```

**Gotcha — team namespace:** this is a Dropbox *Business* account and `Serac_team/` lives in the **team** namespace, not your personal one. Without it, `dropbox:` shows only your own space and every `Serac_team/...` path 404s. Add `root_namespace = <team id>` under `[dropbox]` in `~/.config/rclone/rclone.conf` (find the id via `users/get_current_account` → `root_info.root_namespace_id`). Edit the file **directly** — `rclone config update` re-runs the OAuth flow and hangs without a browser.

#### `ssh` (fallback — reverse tunnel to the laptop's mirror)

`functions.pull_from_dropbox` / `open_dropbox` / `push_to_dropbox` stream the paths **verbatim over passwordless SSH**. Set `DROPBOX_BACKEND: ssh` to use it.

**Topology / why a reverse tunnel:** you sit at the WSL laptop and SSH *into* the cluster, but the pull needs the cluster to reach *back* to the laptop. WSL2 is NAT'd behind Windows, so instead of exposing WSL's sshd to the LAN, ride a **reverse tunnel** on your connection.

* **Laptop (WSL), one-time — run an SSH server:**
  ```bash
  sudo apt install -y openssh-server
  sudo ssh-keygen -A
  sudo service ssh start          # use WSL's sshd (not Windows OpenSSH) so /mnt/c/... paths + cat work
  ```
* **Key auth cluster → laptop, one-time:** append the cluster's `~/.ssh/id_ed25519.pub` to the laptop-WSL `~/.ssh/authorized_keys` (`chmod 600`).
* **Each session — open the reverse tunnel from a WSL terminal on the laptop** (keep it open while pulling):
  ```bash
  ssh -R 2222:localhost:22 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes gtamo@<cluster>
  # the keepalive flags matter: without them a dead link leaves a zombie listener (see below)
  # or add `RemoteForward 2222 localhost:22` to ~/.ssh/config; or a dedicated side-tunnel:
  #   ssh -R 2222:localhost:22 -N gtamo@<cluster>   (if VS Code uses the Windows ssh client)
  ```
* **Tell the functions the host** (they read `$DROPBOX_SSH_HOST`, not the YAML; config default is `gtamo@localhost:2222`):
  ```bash
  export DROPBOX_SSH_HOST=gtamo@localhost DROPBOX_SSH_PORT=2222
  ```
* **In a notebook — pass the config path verbatim:**
  ```python
  import functions as fn
  df   = fn.pull_from_dropbox(DROPBOX_VALIDATION)            # .csv/.tsv/.xlsx/.parquet -> DataFrame
  df   = fn.pull_from_dropbox(PX_20260529_DB, usecols=[...]) # trim big files (this one ~479 MB)
  buf  = fn.open_dropbox(ENAMINE_20260513)                  # other types -> io.BytesIO (feed .sdf to RDKit)
  fn.push_to_dropbox(results_df, DROPBOX_ML + '/out.csv')   # DataFrame/bytes/file -> laptop, Dropbox syncs it up
  ```

The tunnel must be up during pulls (close the session → port 2222 disappears → connection error).

**Troubleshooting `Connection timed out during banner exchange`:** the laptop-side ssh died but left the cluster's port `LISTEN`ing with sockets in `CLOSE-WAIT` — a zombie tunnel. The listener belongs to the root sshd and **cannot be killed from the cluster**, so fix it from the **laptop**: `pkill -f '2222:localhost:22'`, then reconnect. If that reports `remote port forwarding failed for listen port 2222`, the stale listener has not been reaped yet — use `2223` and set `DROPBOX_SSH_PORT` to match. The keepalive flags above prevent this; or just use the `rclone` backend, which has no tunnel to wedge.

`pull_from_dropbox` / `pull_rclone` load the whole file into RAM; `push_to_dropbox` / `push_rclone` write into the company Dropbox — mind the data-privacy rules before pushing derived data out.