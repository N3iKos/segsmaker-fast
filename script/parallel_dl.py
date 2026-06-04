"""
parallel_dl.py — Parallel Download Manager for segsmaker-fast
Repository: https://github.com/N3iKos/segsmaker-fast

Provides:
    - parallel_download(url_list, dest, max_workers)  — core async downloader
    - IPython cell magic %%parallel_download           — notebook convenience wrapper

Backend: aria2c (recommended) with fallback to wget.
Each worker downloads one file at a time; multiple workers run concurrently.

Log format per worker:
    [Worker-N color] <filename>  ▶  aria2c -x16 -s16 -k1M
                  size/total(%) CN:16 DL:speed ETA:time
"""

from __future__ import annotations
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

# ── ANSI color palette (one per worker slot, cycles if >8 workers) ────────
_WORKER_COLORS = [
    "\033[91m",  # bright red
    "\033[93m",  # bright yellow
    "\033[92m",  # bright green
    "\033[96m",  # bright cyan
    "\033[94m",  # bright blue
    "\033[95m",  # bright magenta
    "\033[97m",  # bright white
    "\033[90m",  # dark gray
]
_RST   = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"

# ── Progress line regex for aria2c output ────────────────────────────────
# Matches lines like: [#abc123 1.2GiB/3.8GiB(32%) CN:16 DL:34MiB ETA:1m30s]
_ARIA2_PROGRESS_RE = re.compile(
    r"\[#\w+\s+(?P<size>[^\s/]+)/(?P<total>[^\s(]+)\((?P<pct>\d+)%\)"
    r"(?:\s+CN:(?P<cn>\d+))?"
    r"(?:\s+DL:(?P<dl>[^\s\]]+))?"
    r"(?:\s+ETA:(?P<eta>[^\s\]]+))?\]"
)

_print_lock = threading.Lock()


def _color(idx: int) -> str:
    return _WORKER_COLORS[idx % len(_WORKER_COLORS)]


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _parse_filename_from_url(url: str) -> str:
    """Extract a human-readable filename from a URL."""
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    return name or f"<unnamed>"


def _check_aria2c() -> bool:
    """Return True if aria2c is available in PATH."""
    try:
        r = subprocess.run(
            ["aria2c", "--version"],
            capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _install_aria2c():
    """Attempt to install aria2c via apt (Colab/Kaggle environment)."""
    _safe_print(f"{_DIM}[parallel_dl] aria2c not found, installing via apt...{_RST}")
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "aria2"],
        capture_output=True
    )


def _download_one_aria2c(
    idx: int,
    url: str,
    dest: Path,
    filename: Optional[str] = None,
) -> bool:
    """
    Download a single file using aria2c with 16 parallel connections.
    Streams progress lines to stdout with per-worker color coding.

    Returns True on success, False on failure.
    """
    color = _color(idx)
    name = filename or _parse_filename_from_url(url)

    _safe_print(f"\n{color}{_BOLD}[Worker-{idx+1}]{_RST} {_DIM}Starting:{_RST} {name}")

    cmd = [
        "aria2c",
        "-x", "16",          # 16 connections per server
        "-s", "16",          # 16 segments
        "-k", "1M",          # minimum piece size 1 MiB
        "--summary-interval=1",
        "--console-log-level=notice",
        "--download-result=hide",
        "-d", str(dest),
    ]
    if filename:
        cmd += ["-o", filename]
    cmd.append(url)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        last_progress = ""
        for line in proc.stdout:
            line = line.rstrip()
            m = _ARIA2_PROGRESS_RE.search(line)
            if m:
                g = m.groupdict()
                progress_str = (
                    f"{color}[Worker-{idx+1}]{_RST} "
                    f"{name[:40]} "
                    f"| {g.get('size','?')}/{g.get('total','?')} ({g.get('pct','?')}%)"
                    f" CN:{g.get('cn','?')} DL:{g.get('dl','?')} ETA:{g.get('eta','?')}"
                )
                with _print_lock:
                    sys.stdout.write(f"\r{progress_str}")
                    sys.stdout.flush()
                last_progress = progress_str
            elif line.strip():
                # Print non-progress lines (errors, notices) on own line
                _safe_print(f"\n{color}[Worker-{idx+1}]{_RST} {_DIM}{line}{_RST}")

        proc.wait()

        if last_progress:
            print()  # newline after progress

        if proc.returncode == 0:
            _safe_print(f"{color}[Worker-{idx+1}]{_RST} \033[92m✔ Done:{_RST} {name}")
            return True
        else:
            _safe_print(f"{color}[Worker-{idx+1}]{_RST} \033[91m✘ Failed:{_RST} {name} (exit {proc.returncode})")
            return False

    except Exception as exc:
        _safe_print(f"{color}[Worker-{idx+1}]{_RST} \033[91m✘ Error:{_RST} {name} — {exc}")
        return False


def _download_one_wget(idx: int, url: str, dest: Path) -> bool:
    """Fallback downloader using wget (no parallel connections)."""
    color = _color(idx)
    name = _parse_filename_from_url(url)
    _safe_print(f"{color}[Worker-{idx+1}]{_RST} (wget) {name}")

    r = subprocess.run(
        ["wget", "-q", "--show-progress", "-P", str(dest), url],
        capture_output=False
    )
    ok = r.returncode == 0
    _safe_print(f"{color}[Worker-{idx+1}]{_RST} {'OK' if ok else 'FAIL'}: {name}")
    return ok


def parallel_download(
    url_list: List[str],
    dest: str | Path = ".",
    max_workers: int = 3,
    filenames: Optional[List[Optional[str]]] = None,
    use_wget_fallback: bool = True,
) -> dict:
    """
    Download multiple files in parallel using aria2c.

    Args:
        url_list:      List of URLs to download.
        dest:          Destination directory (created if absent).
        max_workers:   Max concurrent downloads (1–8 recommended).
        filenames:     Optional list of output filenames (same length as url_list).
                       Use None entries to derive names from URLs.
        use_wget_fallback: Fall back to wget if aria2c not found after install attempt.

    Returns:
        dict with keys 'success', 'failed' — lists of URLs.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not url_list:
        print("[parallel_dl] No URLs provided.")
        return {"success": [], "failed": []}

    # Ensure aria2c is available
    if not _check_aria2c():
        _install_aria2c()
        if not _check_aria2c():
            if use_wget_fallback:
                print("[parallel_dl] aria2c unavailable. Using wget fallback (no parallel connections).")
                results = {"success": [], "failed": []}
                for i, url in enumerate(url_list):
                    ok = _download_one_wget(i, url, dest)
                    (results["success"] if ok else results["failed"]).append(url)
                return results
            else:
                raise RuntimeError(
                    "aria2c not found and wget fallback is disabled. "
                    "Run: apt-get install -y aria2"
                )

    fnames = filenames or ([None] * len(url_list))
    if len(fnames) < len(url_list):
        fnames = fnames + [None] * (len(url_list) - len(fnames))

    results = {"success": [], "failed": []}
    lock = threading.Lock()

    _safe_print(
        f"\n\033[1m[parallel_dl]\033[0m Starting {len(url_list)} download(s) "
        f"with {max_workers} worker(s) → {dest}\n"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one_aria2c, idx, url, dest, fname): url
            for idx, (url, fname) in enumerate(zip(url_list, fnames))
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                ok = future.result()
            except Exception as exc:
                _safe_print(f"[parallel_dl] Unexpected error for {url}: {exc}")
                ok = False
            with lock:
                (results["success"] if ok else results["failed"]).append(url)

    total = len(url_list)
    n_ok  = len(results["success"])
    n_fail = len(results["failed"])

    print(f"\n\033[1m[parallel_dl]\033[0m Complete: \033[92m{n_ok}/{total} succeeded\033[0m", end="")
    if n_fail:
        print(f", \033[91m{n_fail} failed\033[0m")
        for u in results["failed"]:
            print(f"  ✘ {u}")
    else:
        print()

    return results


# ── IPython Cell Magic Registration ──────────────────────────────────────

def _register_cell_magic():
    """
    Register the %%parallel_download cell magic for Jupyter/Colab notebooks.

    Usage in a notebook cell:
        %%parallel_download /path/to/dest  --workers 4
        https://huggingface.co/.../model.safetensors
        https://civitai.com/api/download/models/12345

    Options:
        --workers N    Number of parallel workers (default: 3)
        --sequential   Disable parallelism, download one by one
    """
    try:
        from IPython.core.magic import register_cell_magic
        from IPython import get_ipython
    except ImportError:
        return  # Not in IPython environment

    @register_cell_magic
    def parallel_download_magic(line, cell):  # noqa: F811
        import shlex
        parts = shlex.split(line.strip()) if line.strip() else []

        dest_arg   = "."
        workers    = 3
        sequential = False

        i = 0
        while i < len(parts):
            if parts[i] == "--workers" and i + 1 < len(parts):
                workers = int(parts[i + 1])
                i += 2
            elif parts[i] == "--sequential":
                sequential = True
                i += 1
            elif not parts[i].startswith("--"):
                dest_arg = parts[i]
                i += 1
            else:
                i += 1

        url_list = [u.strip() for u in cell.splitlines() if u.strip() and not u.strip().startswith("#")]

        if not url_list:
            print("[%%parallel_download] No URLs found in cell body.")
            return

        if sequential:
            ipy = get_ipython()
            os.chdir(dest_arg)
            for url in url_list:
                ipy.run_line_magic("download", url)
        else:
            parallel_download(url_list, dest=dest_arg, max_workers=workers)

    # Register under the name used in notebooks
    try:
        ip = get_ipython()
        if ip is not None:
            ip.register_magic_function(parallel_download_magic, magic_kind="cell", magic_name="parallel_download")
    except Exception:
        pass


_register_cell_magic()
