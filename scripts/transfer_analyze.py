#!/usr/bin/env python
"""Transfer read (PREREG-LLM-GYM amendment 2026-10-08): Stage-0 adapters scored on real 2020/2022 exam windows.

Input: <root>/eval_<adapter>_<bank>/ prime-rl eval dirs for adapters step_0 (control), step_60 (primary), step_45
(secondary) and banks exam_2020_overlap, exam_2022_overlap. Primary contrast: step_60 vs step_0 on MEAN loss,
paired by window, moving-block bootstrap over window-start dates (block 5, all assets carried, 10k resamples).
ES90 reported as secondary. Pass = CI excludes 0 (step_60 better) on BOTH crises; null = includes 0 on both.

Usage: python scripts/transfer_analyze.py <root> [--banks banks] [--resamples 10000]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import load_eval, bank_layout, loss_matrix, es90  # noqa: E402

SCRIPTED = {"exam_2020_overlap": {"no-hedge": 0.0388, "constant-0.5": 0.0203, "BS-delta": 0.0120},
            "exam_2022_overlap": {"no-hedge": 0.0293, "constant-0.5": 0.0149, "BS-delta": 0.0091}}  # mean loss, scripts/baselines.py


def block_boot(A, B, metric, block=5, n=10000, seed=0):
    rng = np.random.default_rng(seed); T = A.shape[0]; nb = int(np.ceil(T / block)); starts = np.arange(T - block + 1)
    d = np.empty(n)
    for i in range(n):
        s = rng.choice(starts, nb, replace=True); rows = np.concatenate([np.arange(x, x + block) for x in s])[:T]
        d[i] = metric(A[rows].ravel()) - metric(B[rows].ravel())
    return metric(A.ravel()) - metric(B.ravel()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--banks", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "banks"))
    ap.add_argument("--resamples", type=int, default=10000); a = ap.parse_args()
    mean = lambda x: float(np.nanmean(x))
    verdict = {}
    for bank in ("exam_2020_overlap", "exam_2022_overlap"):
        win, dates = bank_layout(os.path.join(a.banks, f"{bank}.npz"))
        M = {}
        for ad in ("step_0", "step_60", "step_45"):
            ev = load_eval(os.path.join(a.root, f"eval_{ad}_{bank}"))
            if ev is not None:
                M[ad] = loss_matrix(ev, win, len(dates))
        print(f"\n==== {bank}: {len(dates)} windows x 4 assets ====")
        print(f"{'player':14s} {'mean loss':>10s} {'ES90':>8s}")
        for k, v in SCRIPTED[bank].items():
            print(f"{k:14s} {v:10.4f} {'':>8s}")
        for ad, m in M.items():
            print(f"{ad:14s} {mean(m):10.4f} {es90(m.ravel()):8.4f}")
        if "step_0" in M and "step_60" in M:
            pm, lo, hi = block_boot(M["step_60"], M["step_0"], mean, n=a.resamples)
            pe, elo, ehi = block_boot(M["step_60"], M["step_0"], es90, n=a.resamples, seed=1)
            ok = hi < 0
            verdict[bank] = ok
            print(f"PRIMARY step_60 - step_0: mean {pm:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {'BETTER (CI excl 0)' if ok else 'not significant'} | ES90 {pe:+.4f} [{elo:+.4f}, {ehi:+.4f}]")
        if "step_0" in M and "step_45" in M:
            pm, lo, hi = block_boot(M["step_45"], M["step_0"], mean, n=a.resamples, seed=2)
            print(f"secondary step_45 - step_0: mean {pm:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    if len(verdict) == 2:
        n_ok = sum(verdict.values())
        print("\nTRANSFER READ:", "PASS (both crises)" if n_ok == 2 else "NULL (neither crisis)" if n_ok == 0 else "INCONCLUSIVE (one crisis only)")


if __name__ == "__main__":
    main()
