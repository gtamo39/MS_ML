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