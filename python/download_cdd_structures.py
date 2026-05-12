#!/usr/bin/env python3
"""
Download compound structure PNGs from CDD Vault via the REST API.

Runs entirely on your machine. Your API token stays local; the PNGs are
written straight to disk. No chemistry data leaves your computer.

------------------------------------------------------------------
ONE-TIME SETUP
------------------------------------------------------------------
1. Get your CDD Vault API token:
   Log into CDD Vault -> click your name top-right -> "My Account"
   -> "API Tokens" tab -> "Generate New Token". Copy the token.

2. Save it somewhere local. Easiest: a plain text file like
       C:\\Users\\gtamo\\.cdd_token     (Windows)
       /home/gtamo/.cdd_token           (Linux/WSL)
   containing just the token string on one line.

3. Install the one dependency:
       pip install requests

------------------------------------------------------------------
RUN
------------------------------------------------------------------
First verify the script can see your data (no files written):

    python download_cdd_structures.py \\
        --vault 7108 \\
        --search 23196193 \\
        --token-file ~/.cdd_token \\
        --output ./png_out \\
        --discover

Then run for real (writes 6308 PNGs to ./png_out/):

    python download_cdd_structures.py \\
        --vault 7108 \\
        --search 23196193 \\
        --token-file ~/.cdd_token \\
        --output ./png_out

Resume support: rerunning skips files that already exist, so if you
stop mid-way you can pick up where you left off.

------------------------------------------------------------------
FLAGS
------------------------------------------------------------------
    --vault N          Vault ID (required)
    --search ID        Saved-search ID (numeric portion of the URL)
                       e.g. for .../searches/23196193-gdca... use 23196193
    --token-file PATH  File containing your API token (one line)
    --token STRING     Token inline (less safe — appears in shell history)
    --output PATH      Directory to write PNGs (created if absent)
    --size N           PNG width/height in px, default 600
    --workers N        Parallel downloads, default 8 (be gentle with the API)
    --delay SEC        Sleep between batches per worker, default 0.0
    --limit N          Stop after N molecules (handy for testing)
    --name-prefix STR  Stripped from molecule name to form filename, default "SRB-"
    --discover         Probe endpoints + dump 3 sample molecules, write nothing
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: please install the 'requests' package first:  pip install requests")
    sys.exit(1)


API_BASE = "https://app.collaborativedrug.com/api/v1"


# ---------- token loading ----------
def load_token(args) -> str:
    if args.token:
        return args.token.strip()
    if args.token_file:
        p = Path(args.token_file).expanduser()
        if not p.exists():
            sys.exit(f"ERROR: token file not found: {p}")
        return p.read_text().strip()
    env = os.environ.get("CDD_TOKEN")
    if env:
        return env.strip()
    sys.exit("ERROR: no API token. Use --token, --token-file, or set CDD_TOKEN env var.")


def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers["X-CDD-Token"] = token
    s.headers["Accept"] = "application/json"
    return s


# ---------- listing molecules ----------
def list_molecules_in_search(s: requests.Session, vault: int, search: str,
                             page_size: int = 1000,
                             limit: int | None = None) -> list[dict]:
    """Paginate through all molecules in a saved search and return their objects.

    Pass `limit` to short-circuit after that many results (useful for --limit).
    Prints progress to stdout so a long listing doesn't look stuck.
    """
    out: list[dict] = []
    offset = 0
    while True:
        size = page_size
        if limit is not None:
            size = min(page_size, max(1, limit - len(out)))
        url = (f"{API_BASE}/vaults/{vault}/searches/{search}/molecules"
               f"?offset={offset}&page_size={size}")
        r = s.get(url, timeout=120)
        r.raise_for_status()
        data = r.json()
        objs = data.get("objects") or []
        if not objs:
            break
        out.extend(objs)
        total = data.get("count")
        print(f"  listing... {len(out)}/{total if total is not None else '?'}", flush=True)
        if limit is not None and len(out) >= limit:
            out = out[:limit]
            break
        offset += len(objs)
        if total is not None and offset >= total:
            break
        if len(objs) < size:
            break
    return out


def list_molecule_ids_in_search(s: requests.Session, vault: int, search: str,
                                limit: int | None = None) -> list[int]:
    """Back-compat wrapper returning just IDs."""
    return [m.get("id") for m in list_molecules_in_search(s, vault, search, limit=limit) if m.get("id")]


def list_molecule_ids_in_vault(s: requests.Session, vault: int, limit: int | None = None) -> list[int]:
    """Fallback: list all molecules in the vault."""
    ids: list[int] = []
    offset = 0
    page = 1000
    while True:
        url = f"{API_BASE}/vaults/{vault}/molecules?async=False&offset={offset}&page_size={page}"
        r = s.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        objs = data.get("objects") or data.get("molecules") or []
        if not objs:
            break
        for obj in objs:
            mid = obj.get("id")
            if mid:
                ids.append(mid)
                if limit and len(ids) >= limit:
                    return ids
        if len(objs) < page:
            break
        offset += len(objs)
    return ids


# ---------- fetching one structure (CDD async job flow) ----------
# Flow:
#   1) GET /vaults/{v}/molecules/{m}/image?... -> JSON {id: JOB_ID, status: 'new', ...}
#   2) Poll GET /vaults/{v}/exports/{JOB_ID} until content-type is image/*
ENDPOINT_TEMPLATES = [
    "{API}/vaults/{v}/molecules/{m}/image?width={w}&height={w}&format=png",
]


def fetch_one(s: requests.Session, vault: int, molecule_id: int, size: int,
              endpoint_template: str | None = None,
              max_wait: float = 120.0,
              initial_poll: float = 0.3,
              max_poll: float = 2.0) -> tuple[bytes | None, str]:
    """Submit an image job and poll the export endpoint for the PNG bytes.

    Returns (png_bytes, reason). On success reason is "ok".
    On failure png_bytes is None and reason is a short label describing why.
    """
    # 1) Submit job
    submit_url = (f"{API_BASE}/vaults/{vault}/molecules/{molecule_id}/image"
                  f"?width={size}&height={size}&format=png")
    try:
        r = s.get(submit_url, timeout=30)
    except requests.RequestException as e:
        return None, f"submit_exc:{type(e).__name__}"
    if r.status_code == 429:
        return None, "submit_429_rate_limited"
    if r.status_code in (502, 503, 504):
        return None, f"submit_{r.status_code}_server"
    if r.status_code != 200:
        return None, f"submit_http_{r.status_code}"
    try:
        job = r.json()
    except Exception:
        return None, "submit_not_json"
    job_id = job.get("id")
    if not job_id:
        return None, f"submit_no_job_id:keys={','.join(list(job.keys())[:5])}"

    # 2) Poll the export endpoint until it serves an image
    poll_url = f"{API_BASE}/vaults/{vault}/exports/{job_id}"
    deadline = time.time() + max_wait
    wait = initial_poll
    last_status = None
    poll_count = 0
    while time.time() < deadline:
        poll_count += 1
        try:
            r2 = s.get(poll_url, timeout=30)
        except requests.RequestException as e:
            time.sleep(wait)
            wait = min(wait * 1.5, max_poll)
            last_status = f"poll_exc:{type(e).__name__}"
            continue
        ct = r2.headers.get("content-type", "")
        if r2.status_code == 200 and ct.startswith("image"):
            return r2.content, "ok"
        if r2.status_code == 200 and "json" in ct:
            # Still pending; capture status field if present
            try:
                j = r2.json()
                last_status = f"pending:{j.get('status', '?')}"
            except Exception:
                last_status = "pending:?"
        elif r2.status_code == 404:
            last_status = "poll_404"
        elif r2.status_code == 429:
            last_status = "poll_429"
        else:
            last_status = f"poll_http_{r2.status_code}"
        time.sleep(wait)
        wait = min(wait * 1.5, max_poll)
    return None, f"timeout:{last_status}:{poll_count}polls"


def fetch_molecule_meta(s: requests.Session, vault: int, mid: int) -> dict:
    """Get name / numeric ID for one molecule."""
    r = s.get(f"{API_BASE}/vaults/{vault}/molecules/{mid}", timeout=30)
    r.raise_for_status()
    return r.json()


# ---------- discover mode ----------
def _short(text: str, n: int = 300) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n] + "..."


def _try(s: requests.Session, url: str, label: str = ""):
    """Hit a URL and print a one-line status summary."""
    try:
        r = s.get(url, timeout=30)
    except Exception as e:
        print(f"  ERR  {label or url}  ({e})")
        return None
    ct = r.headers.get("content-type", "?")
    short_url = url.replace(API_BASE, "")
    print(f"  {r.status_code:>3}  {ct[:40]:<40}  {short_url}")
    return r


def discover(s: requests.Session, vault: int, search: str | None, size: int):
    print("=" * 70)
    print("STEP 1: Verify token by hitting a basic vault endpoint")
    print("=" * 70)
    for url in [
        f"{API_BASE}/vaults/{vault}",
        f"{API_BASE}/vaults",
    ]:
        r = _try(s, url)
        if r is not None and r.status_code == 200:
            try:
                data = r.json()
                if isinstance(data, list):
                    print(f"     -> list with {len(data)} entries; first keys: {list(data[0].keys())[:8] if data else '(empty)'}")
                else:
                    print(f"     -> keys: {list(data.keys())[:8]}")
            except Exception:
                print(f"     -> body (first 200 chars): {_short(r.text, 200)}")
            break
        if r is not None and r.status_code in (401, 403):
            print("\n  !! Token rejected. Double-check the token file content and ")
            print("     that the token has access to this vault. Stopping.")
            return

    print()
    print("=" * 70)
    print("STEP 2: Probe search endpoints (find which shape returns molecule IDs)")
    print("=" * 70)
    if not search:
        print("  (no --search provided; skipping)")
    else:
        print(f"  search arg passed: {search!r}")
        candidates = [
            f"{API_BASE}/vaults/{vault}/searches/{search}",
            f"{API_BASE}/vaults/{vault}/searches/{search}/results",
            f"{API_BASE}/vaults/{vault}/searches/{search}/molecules",
            f"{API_BASE}/vaults/{vault}/molecules?search={search}",
            f"{API_BASE}/vaults/{vault}/molecules?search_id={search}",
            f"{API_BASE}/vaults/{vault}/molecules?async=False&search={search}&page_size=3",
            f"{API_BASE}/vaults/{vault}/searches",
        ]
        for url in candidates:
            r = _try(s, url)
            if r is None or r.status_code != 200:
                continue
            try:
                data = r.json()
            except Exception:
                print(f"     -> non-JSON body: {_short(r.text, 200)}")
                continue
            if isinstance(data, list):
                print(f"     -> list with {len(data)} entries"
                      + (f"; first keys: {list(data[0].keys())[:8]}" if data else ""))
            elif isinstance(data, dict):
                keys = list(data.keys())
                print(f"     -> dict keys: {keys[:12]}")
                # try to spot molecule-id-bearing fields
                for k in ("objects", "molecules", "results", "molecule_ids", "ids"):
                    if k in data and isinstance(data[k], list):
                        print(f"        '{k}': list of {len(data[k])}"
                              + (f"  first item keys: {list(data[k][0].keys())[:8]}"
                                 if data[k] and isinstance(data[k][0], dict) else
                                 f"  first item: {data[k][0]!r}" if data[k] else ""))
            else:
                print(f"     -> body (first 200 chars): {_short(r.text, 200)}")

    print()
    print("=" * 70)
    print("STEP 3: Pull a sample molecule from the search and probe image endpoints")
    print("=" * 70)
    mid = None
    if search:
        r = _try(s, f"{API_BASE}/vaults/{vault}/searches/{search}/molecules?page_size=1")
        if r is not None and r.status_code == 200:
            try:
                data = r.json()
                objs = data.get("objects") or []
                if objs and isinstance(objs[0], dict):
                    mid = objs[0].get("id")
                    print(f"     -> sample molecule id: {mid}")
                    print(f"     -> available fields: {list(objs[0].keys())[:14]}")
                    print(f"     -> total in search ('count'): {data.get('count')}")
            except Exception as e:
                print(f"     -> couldn't parse: {e}")

    if mid:
        print()
        print("  Image endpoint returns an async job — walking the full lifecycle:")
        print()
        # 0) First: try synchronous mode with ?async=false  (cheapest if it works)
        print("  Try-0: synchronous mode (?async=false)")
        sync_url = f"{API_BASE}/vaults/{vault}/molecules/{mid}/image?async=false&width={size}&height={size}&format=png"
        r0 = _try(s, sync_url)
        if r0 is not None and r0.status_code == 200:
            body = r0.content
            ct = r0.headers.get("content-type", "")
            png_magic = body[:8] == b"\x89PNG\r\n\x1a\n"
            print(f"     -> body len={len(body)}  pngMagic={png_magic}  ct={ct}")
            if "json" in ct:
                try:
                    j = r0.json()
                    print(f"     -> json keys: {list(j.keys())[:15]}")
                except Exception:
                    pass

        print()
        # 1) Start the job
        start_url = f"{API_BASE}/vaults/{vault}/molecules/{mid}/image?width={size}&height={size}&format=png"
        r = _try(s, start_url)
        if r is None or r.status_code != 200:
            print("     (could not start image job)")
        else:
            try:
                job = r.json()
            except Exception:
                job = {}
            job_id = job.get("id")
            print(f"     -> job id={job_id}, status={job.get('status')!r}, queued_pos={job.get('queued_job_position')}")
            if job_id:
                # 2) Poll candidate job-status endpoints — wider net
                poll_candidates = [
                    f"{API_BASE}/vaults/{vault}/jobs/{job_id}/status",
                    f"{API_BASE}/vaults/{vault}/async_jobs/{job_id}",
                    f"{API_BASE}/vaults/{vault}/jobs/{job_id}",
                    f"{API_BASE}/vaults/{vault}/molecule_image_jobs/{job_id}",
                    f"{API_BASE}/vaults/{vault}/image_jobs/{job_id}",
                    f"{API_BASE}/vaults/{vault}/image_progress?id={job_id}",
                    f"{API_BASE}/vaults/{vault}/image_progress?poller_request=true",
                    f"{API_BASE}/vaults/{vault}/job_progress?id={job_id}",
                    f"{API_BASE}/vaults/{vault}/jobs?id={job_id}",
                    f"{API_BASE}/vaults/{vault}/molecules/{mid}/image_status",
                    f"{API_BASE}/vaults/{vault}/molecules/{mid}/image/jobs/{job_id}",
                    f"{API_BASE}/vaults/{vault}/molecules/{mid}/image_jobs/{job_id}",
                    f"{API_BASE}/vaults/{vault}/exports/{job_id}",
                    f"{API_BASE}/async_jobs/{job_id}",
                    f"{API_BASE}/jobs/{job_id}",
                ]
                print()
                print("     Probing poll endpoints (first that returns 200 wins):")
                working_poll = None
                for pu in poll_candidates:
                    r2 = _try(s, pu)
                    if r2 is not None and r2.status_code == 200:
                        try:
                            jd = r2.json()
                            print(f"        keys: {list(jd.keys())[:15]}")
                            if "status" in jd:
                                working_poll = pu
                                break
                        except Exception:
                            pass
                if not working_poll:
                    print("     !! No working poll endpoint found.")
                else:
                    # 3) Poll until completed
                    print()
                    print(f"     Polling {working_poll[len(API_BASE):]} until done...")
                    final = None
                    for i in range(30):
                        time.sleep(1)
                        try:
                            r3 = s.get(working_poll, timeout=30)
                        except Exception as e:
                            print(f"     poll err: {e}")
                            break
                        try:
                            jd = r3.json()
                        except Exception:
                            jd = {}
                        st = jd.get("status")
                        print(f"       t={i+1}s  status={st!r}  queued_pos={jd.get('queued_job_position')!r}  keys={list(jd.keys())[:12]}")
                        if st in ("completed", "complete", "done", "success", "finished"):
                            final = jd
                            break
                        if st in ("failed", "error", "cancelled"):
                            print(f"     !! Job ended with status={st!r}")
                            break
                    if final:
                        # 4) Look for image bytes / URL in the completed job body
                        for k, v in final.items():
                            if isinstance(v, str) and len(v) > 50:
                                if v.startswith("data:"):
                                    kind = "data-URI"
                                elif v.startswith("http"):
                                    kind = "URL"
                                elif v.startswith("/"):
                                    kind = "rel-URL"
                                elif all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n" for c in v[:120]):
                                    kind = "base64"
                                else:
                                    kind = "opaque"
                                print(f"       result field '{k}': length={len(v)}, kind={kind}")
                        # 5) Also try re-hitting the original image endpoint (maybe now serves cached)
                        print()
                        print("     Re-hitting the original image URL after completion:")
                        r4 = _try(s, start_url)
                        if r4 is not None and r4.status_code == 200:
                            body = r4.content
                            print(f"       body len={len(body)}  pngMagic={body[:8] == b'\\x89PNG\\r\\n\\x1a\\n'}")
                            ct4 = r4.headers.get("content-type", "")
                            if "application/json" in ct4:
                                try:
                                    j4 = r4.json()
                                    print(f"       json keys: {list(j4.keys())[:15]}")
                                except Exception:
                                    pass
        # Also fetch the molecule itself and inspect what keys it returns —
        # the structure or an image URL might be embedded in the response.
        print()
        print(f"  Molecule detail body keys (no chemistry values printed):")
        try:
            meta = fetch_molecule_meta(s, vault, mid)
            print(f"     -> {list(meta.keys())[:30]}")
            # Identify any field that looks like an image URL or structure ref.
            # IMPORTANT: print only metadata (name, length, kind), never the
            # actual value — so this output is safe to paste back to the assistant.
            for k, v in meta.items():
                if isinstance(v, str) and any(tok in k.lower() for tok in ("image", "structure", "depiction", "png", "svg", "url")):
                    if v.startswith("http"):
                        kind = "URL"
                    elif v.startswith("data:"):
                        kind = "data-URI"
                    elif v[:5].upper() == "<?XML" or v.startswith("<svg"):
                        kind = "SVG-inline"
                    else:
                        kind = "opaque-string"
                    print(f"     -> field '{k}': length={len(v)}, kind={kind}")
        except Exception as e:
            print(f"     -> couldn't fetch meta: {e}")

    print()
    print("=" * 70)
    print("DONE.  Paste this whole output back so we can adjust the script.")
    print("=" * 70)


# ---------- main download loop ----------
# Characters that aren't safe in filenames on Windows / macOS / Linux.
_BAD_FNAME_CHARS = r'<>:"/\\|?*'
_BAD_FNAME_TABLE = str.maketrans({c: "_" for c in _BAD_FNAME_CHARS})


def _sanitize(name: str, max_len: int = 200) -> str:
    """Make a string safe to use as a filename across operating systems.

    Replaces filesystem-reserved characters with underscores, strips control
    chars, trims whitespace, collapses internal whitespace, and length-caps
    the result. Returns "" if nothing usable remains.
    """
    # remove control chars
    name = "".join(ch for ch in name if ord(ch) >= 32)
    # collapse whitespace
    name = " ".join(name.split())
    # replace reserved chars
    name = name.translate(_BAD_FNAME_TABLE)
    # trim trailing dots/spaces (Windows hates these)
    name = name.rstrip(". ")
    if len(name) > max_len:
        name = name[:max_len].rstrip(". ")
    return name


def filename_for(name: str | None, mid: int, prefix: str, strip_prefix: bool) -> str:
    """Build the on-disk filename for a molecule.

    If `strip_prefix` is True and `name` begins with `prefix`, the prefix is
    removed from the filename (e.g. "SRB-0008912" -> "0008912.png").
    Otherwise the full name is used (e.g. "SRB-0008912.png"). Compound names
    are sanitized for cross-platform filesystem safety. Falls back to the
    numeric molecule id if no usable name is present.
    """
    if name:
        clean = _sanitize(name)
        if clean:
            if strip_prefix and clean.startswith(prefix):
                clean = clean[len(prefix):]
            if clean:
                return clean + ".png"
    return f"{mid}.png"


def download_all(s: requests.Session, vault: int, mols: list[dict], out: Path,
                 size: int, workers: int, delay: float, prefix: str,
                 strip_prefix: bool):
    """Download images for each molecule in `mols` (list of {id, name, ...} dicts)."""
    out.mkdir(parents=True, exist_ok=True)
    total = len(mols)
    n_ok, n_skip, n_err, n_done = 0, 0, 0, 0
    reasons: dict[str, int] = {}
    start = time.time()

    def worker(mol: dict):
        mid = mol.get("id")
        if not mid:
            return ("err", None, None, "no_id_in_listing")
        # Use the name from the listing response — no extra API call.
        # Fall back to fetching meta only if name is missing.
        name = mol.get("name")
        if not name:
            try:
                meta = fetch_molecule_meta(s, vault, mid)
                name = meta.get("name")
            except Exception as e:
                # Continue with numeric-id fallback rather than failing the whole compound.
                pass
        fn = filename_for(name, mid, prefix, strip_prefix)
        path = out / fn
        if path.exists() and path.stat().st_size > 0:
            return ("skip", mid, fn, None)
        png, reason = fetch_one(s, vault, mid, size)
        if not png:
            return ("err", mid, fn, reason)
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            return ("err", mid, fn, "not_png_magic")
        path.write_bytes(png)
        if delay:
            time.sleep(delay)
        return ("ok", mid, fn, None)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for status, mid, fn, reason in ex.map(worker, mols):
            n_done += 1
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
            if n_done % 25 == 0 or n_done == total:
                elapsed = time.time() - start
                rate = n_done / max(elapsed, 0.001)
                eta = (total - n_done) / max(rate, 0.001)
                top_reasons = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
                tail = "  " + " ".join(f"{k}={v}" for k, v in top_reasons) if top_reasons else ""
                print(f"  {n_done:>5}/{total}  ok={n_ok} skip={n_skip} err={n_err}  "
                      f"{rate:5.1f}/s  ETA {eta/60:5.1f} min{tail}", flush=True)
    # Final breakdown
    if reasons:
        print()
        print("Error breakdown:")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>6}  {k}")
    return n_ok, n_skip, n_err


# ---------- entry ----------
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--vault", type=int, required=True)
    p.add_argument("--search", type=str)
    p.add_argument("--token")
    p.add_argument("--token-file")
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=600)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--name-prefix", default="SRB-",
                   help="If --strip-prefix is given, this leading string is removed from filenames.")
    p.add_argument("--strip-prefix", action="store_true",
                   help="Strip --name-prefix from filenames (so SRB-0008912 becomes 0008912.png). "
                        "By default filenames preserve the full compound name (e.g. SRB-0008912.png).")
    p.add_argument("--discover", action="store_true")
    args = p.parse_args()

    token = load_token(args)
    s = make_session(token)

    if args.discover:
        discover(s, args.vault, args.search, args.size)
        return

    if args.search:
        print(f"[main] resolving search {args.search} in vault {args.vault}...", flush=True)
        mols = list_molecules_in_search(s, args.vault, args.search, limit=args.limit)
    else:
        print(f"[main] listing all molecules in vault {args.vault}...", flush=True)
        mol_ids = list_molecule_ids_in_vault(s, args.vault, limit=args.limit)
        mols = [{"id": mid} for mid in mol_ids]
    if args.limit:
        mols = mols[: args.limit]
    print(f"[main] {len(mols)} molecules to fetch", flush=True)
    if not mols:
        sys.exit("[main] no molecules found. Try --discover.")

    out = Path(args.output).expanduser()
    n_ok, n_skip, n_err = download_all(
        s, args.vault, mols, out,
        size=args.size, workers=args.workers,
        delay=args.delay, prefix=args.name_prefix,
        strip_prefix=args.strip_prefix,
    )
    print(f"\nDone.  ok={n_ok}  skipped(already-existed)={n_skip}  errors={n_err}")
    print(f"Files in: {out.resolve()}")


if __name__ == "__main__":
    main()