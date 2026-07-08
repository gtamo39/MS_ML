Machine Learning on Proteomics Data to guide library expansion
----------------------------------------------------------------

let's go through the Jupyter Notebook located in vignettes/main.py

### To go through the notebook report:
---------------------------------------

* install your virtual env with python: `python3 -m venv ml` then `source ml/bin/activate`
* OR with conda (recommended) `conda create -n ML python=3.12.3` then `conda activate ML`
  * **Use Python 3.12, not 3.13** — on 3.13 `pip install` tries to build `pyarrow` from source (no cp313 wheel for the pinned version) and fails with `ModuleNotFoundError: No module named 'pkg_resources'`.
* install dependencies: `pip install -r requirements.txt`
* install Vina: `conda install conda-forge::vina`
* install OpenBabel: `conda install -c conda-forge openbabel`
* `python -m ipykernel install --user --name ML --display-name "Python (ML)"`

### Pulling data files from Dropbox (over SSH)
-----------------------------------------------

The `config/config.yaml` path keys (`PATENTS_RAW`, `PX_*`, `DROPBOX_*`, `FBX_DIR`, `ENAMINE_*`, `STOCK_INVENTORY`) are WSL paths where the **laptop's Dropbox desktop client** keeps files synced (`/mnt/c/.../Serac Biosciences Dropbox/...`). From a machine without the client (e.g. the cluster), `functions.pull_from_dropbox` / `open_dropbox` / `push_to_dropbox` stream those paths **verbatim over passwordless SSH straight into memory** — nothing is written to the local disk.

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
  ssh -R 2222:localhost:22 gtamo@<cluster>
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

The tunnel must be up during pulls (close the session → port 2222 disappears → connection error). `pull_from_dropbox` loads the whole file into RAM; `push_to_dropbox` writes into the company Dropbox — mind the data-privacy rules before pushing derived data out.