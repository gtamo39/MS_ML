Machine Learning on Proteomics Data to guide library expansion
----------------------------------------------------------------

let's go through the Jupyter Notebook located in vignettes/main.py

### To go through the notebook report:
---------------------------------------

* install your virtual env with python: `python3 -m venv ml` then `source ml/bin/activate`
* OR with conda (recommended) `conda create -n ML python=3.12.3` then `conda activate ML`
  * **Use Python 3.12, not 3.13** — on 3.13 `pip install` tries to build `pyarrow` from source (no cp313 wheel for the pinned version) and fails with `ModuleNotFoundError: No module named 'pkg_resources'`.
* install dependencies: `pip install -r requirements.txt` (needs `git` on PATH — `descriptastorus` is pinned to a git commit, see below)
* install Vina: `conda install conda-forge::vina`
* install OpenBabel: `conda install -c conda-forge openbabel`
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

**Register the deployable activity classifiers in MLTrail** — both are `RandomForestClassifier(
n_estimators=200, class_weight='balanced')` on `MF_features`, differing only in the label: 5-fold CV
for the `roc_auc` / `pr_auc` metrics, refit on the whole library, then the `{model, feature_cols}`
bundle plus the `compound/smiles/label` training set go into the vault.

| flag | label | config key |
|---|---|---|
| `--save_non_silent` | active: `ndown > 0` | `NON_SILENT_MODEL_NAME` |
| `--save_single_low` | selective: `1 <= ndown <= 12` | `SINGLELOW_MODEL_NAME` |

```bash
python python/MS_build_ML.py --config config/config.yaml --save_non_silent --save_single_low
python python/MS_build_ML.py --config config/config.yaml --save_non_silent --comment 'H237, cohort minus 2026-08-13'
```

`--comment` stores a free-text note **on the version being written**, next to its metrics, so each
version can record what changed. Omit it and MLTrail leaves the field out.

```python
output.save_non_silent_model(data, params)                      # same thing from the notebook
output.save_single_low_model(data, params)
output.save_single_low_model(data, params, model=my_clf)        # try another estimator
output.save_single_low_model(data, params, overwrite=True)      # CORRECT the latest version in place
```

Both flags share one process on purpose — **never run two saves concurrently**: `Registry` reads the
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
affect the screen and the classifiers alike; the last two only name the registry entries:

| key | effect |
|---|---|
| `EXCLUDE_DATES` | screen dates dropped from `df_raw` **and** `MS` (`DATA.drop_excluded_dates`) — in memory, the parquets keep them, so emptying the key restores them without a rebuild. The gene screen and the classifiers share one cohort |
| `CM2RM_PARTS` | which blacklists build `cm2rm`: any subset of `contaminants` / `control_compounds` / `fbx_independent`. `--cm2rm a,b` overrides it |
| `NON_SILENT_MODEL_NAME` | MLTrail `experiment_name` for the `ndown > 0` classifier (`--save_non_silent`) |
| `SINGLELOW_MODEL_NAME` | MLTrail `experiment_name` for the `1 <= ndown <= 12` classifier (`--save_single_low`) |

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