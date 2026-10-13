#!/usr/bin/env python
"""Shared loaders and statistics for the analysis scripts.

Reads prime-rl eval outputs (per-episode traces under */monitors/file/traces/stream/*.jsonl.zst), lays the
per-episode losses out as a (windows x assets) matrix in window-start order using the exam bank's metadata,
and computes the pooled ES90 (mean of the worst 10%) with a paired moving-block bootstrap over window-start
dates (block = 5 dates, all four assets carried together per date), as pre-registered.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import zstandard as zstd


def load_eval(eval_dir: str):
    """-> dict task_name -> (loss, asset, window_index). Needs the bank's meta for dates; see bank_layout."""
    files = sorted(glob.glob(os.path.join(eval_dir, "*", "monitors", "file", "traces", "stream", "*.jsonl.zst")))
    if not files:
        return None
    out = {}
    for f in files:
        raw = zstd.ZstdDecompressor().stream_reader(open(f, "rb")).read().decode()
        for line in raw.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            d = r["task"]["data"]
            tr = r["traces"][0]
            out[d["name"]] = (-float(tr["rewards"]["hedge_score"]["score"]), int(d["info"]["asset"]), int(d["idx"]))
    return out


def bank_layout(bank_npz: str):
    """Window index per task (tasks are concatenated per asset in window order) and the start dates."""
    z = np.load(bank_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    n_windows = len(meta["window_start_dates"])
    asset = z["asset"]
    # task i belongs to asset asset[i]; within an asset, tasks are in window order
    win = np.zeros(len(asset), int)
    for a in np.unique(asset):
        idx = np.where(asset == a)[0]
        win[idx] = np.arange(len(idx))
    assert (np.bincount(asset) == n_windows).all(), "bank is not (n_windows x n_assets)"
    return win, meta["window_start_dates"]


def loss_matrix(ev: dict, win: np.ndarray, n_windows: int, n_assets: int = 4):
    """(n_windows, n_assets) loss matrix in window-start order."""
    M = np.full((n_windows, n_assets), np.nan)
    for name, (loss, asset, idx) in ev.items():
        M[win[idx], asset] = loss
    return M


def es90(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    k = max(1, int(np.ceil(0.10 * len(x))))
    return float(np.sort(x)[-k:].mean())


def block_bootstrap_diff(A: np.ndarray, B: np.ndarray, block: int, n: int, seed: int = 0):
    """Paired moving-block bootstrap over dates of ES90(A) - ES90(B). A, B: (n_dates, n_assets)."""
    rng = np.random.default_rng(seed)
    T = A.shape[0]
    n_blocks = int(np.ceil(T / block))
    starts_all = np.arange(T - block + 1)
    diffs = np.empty(n)
    for i in range(n):
        starts = rng.choice(starts_all, n_blocks, replace=True)
        rows = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        diffs[i] = es90(A[rows].ravel()) - es90(B[rows].ravel())
    point = es90(A.ravel()) - es90(B.ravel())
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
