#!/usr/bin/env python
"""Larger-model check: five independently trained adapters (Qwen3-4B-Instruct-2507, trained only on
synthetic worlds) vs the untrained model, on the real 2020 and 2022 exam windows.

Input: runs/larger_model_5seeds/{trained,untrained}/era2019/seed<k>/exam_<crisis>_overlap/
(trained: seeds 0-4; untrained: seed 0 — the untrained model is deterministic under greedy decoding, so one
scoring suffices and is reused against every seed).

Per seed: ES90(trained_k) - ES90(untrained), paired moving-block bootstrap over window-start dates
(block 5, all assets carried, 10,000 resamples). Pooled: the five trained matrices concatenated along
the asset axis vs the untrained matrix tiled five times, same bootstrap.
Usage: python scripts/trained_vs_untrained.py [--resamples 10000]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import load_eval, bank_layout, loss_matrix, es90, block_bootstrap_diff  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def mean_boot(A, B, block=5, n=10000, seed=0):
    rng = np.random.default_rng(seed); T = A.shape[0]; nb = int(np.ceil(T / block)); starts = np.arange(T - block + 1)
    d = np.empty(n)
    for i in range(n):
        s = rng.choice(starts, nb, replace=True); rows = np.concatenate([np.arange(x, x + block) for x in s])[:T]
        d[i] = np.nanmean(A[rows]) - np.nanmean(B[rows])
    return float(np.nanmean(A) - np.nanmean(B)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resamples", type=int, default=10000); a = ap.parse_args()
    out = os.path.join(ROOT, "runs", "larger_model_5seeds")
    for crisis in ("2020", "2022"):
        win, dates = bank_layout(os.path.join(ROOT, "banks", f"exam_{crisis}_overlap.npz"))
        base = loss_matrix(load_eval(os.path.join(out, "untrained", "era2019", "seed0", f"exam_{crisis}_overlap")), win, len(dates))
        flows = []
        print(f"\n==== real {crisis}: {len(dates)} windows x 4 assets ====")
        print(f"untrained       mean {np.nanmean(base):.4f}  ES90 {es90(base.ravel()):.4f}")
        for s in range(5):
            d = os.path.join(out, "trained", "era2019", f"seed{s}", f"exam_{crisis}_overlap")
            ev = load_eval(d)
            if ev is None:
                print(f"seed {s}: missing"); continue
            M = loss_matrix(ev, win, len(dates)); flows.append(M)
            pm, lo, hi = mean_boot(M, base, n=a.resamples, seed=s); pe, elo, ehi = block_bootstrap_diff(M, base, 5, a.resamples, seed=10 + s)
            print(f"trained seed {s}  mean {np.nanmean(M):.4f} ({pm:+.4f} [{lo:+.4f},{hi:+.4f}])  ES90 {es90(M.ravel()):.4f} ({pe:+.4f} [{elo:+.4f},{ehi:+.4f}])")
        A = np.concatenate(flows, axis=1); B = np.tile(base, (1, len(flows)))
        pm, lo, hi = mean_boot(A, B, n=a.resamples, seed=99); pe, elo, ehi = block_bootstrap_diff(A, B, 5, a.resamples, seed=199)
        print(f"POOLED {len(flows)} seeds vs untrained: mean {pm:+.4f} [{lo:+.4f},{hi:+.4f}] | ES90 {pe:+.4f} [{elo:+.4f},{ehi:+.4f}] -> "
              f"{'trained better, CI excl. 0' if hi < 0 else 'not significant'} (mean); {'CI excl. 0' if ehi < 0 else 'n.s.'} (ES90)")


if __name__ == "__main__":
    main()
