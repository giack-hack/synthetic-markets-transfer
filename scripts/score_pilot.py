#!/usr/bin/env python
"""Score the Stage-0 learning check from prime-rl eval outputs.

Reads per-episode rewards from each eval's traces/stream/*.jsonl.zst (task name ->
hedge_score), pairs trained vs untrained by task, and reports the prereg's pass bar:
  trained beats untrained on mean score, bootstrap CI (paired, 10,000 resamples) excluding 0,
  AND invalid-reply rate < 5%.
Also reports where each checkpoint sits relative to the scripted baselines.

Usage: python score_pilot.py <pilot_out_dir>
"""
import glob, json, os, sys
import numpy as np
import zstandard as zstd

BASELINES = {"no-hedge": -0.0248, "constant-0.5": -0.0131, "BS-delta": -0.0060}  # scripts/baselines.py era2019_flow_heldout


def load_eval(eval_dir):
    # accepts both the raw prime-rl layout (<eval>/<run>/monitors/file/traces/stream/) and the flattened copy
    # committed in this repo (<eval>/traces/stream/); same files either way
    files = sorted(glob.glob(os.path.join(eval_dir, "*", "monitors", "file", "traces", "stream", "*.jsonl.zst"))
                   or glob.glob(os.path.join(eval_dir, "traces", "stream", "*.jsonl.zst")))
    if not files:
        return None
    scores, invalid = {}, {}
    for f in files:
        raw = zstd.ZstdDecompressor().stream_reader(open(f, "rb")).read().decode()
        for line in raw.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            name = r["task"]["data"]["name"]
            tr = r["traces"][0]
            scores[name] = tr["rewards"]["hedge_score"]["score"]
            invalid[name] = float((tr.get("metrics") or {}).get("invalid_replies", 0.0))  # hedge_gym metric, per episode
    n = len(scores)
    tot = sum(invalid.values())
    return {"scores": scores, "n": n, "invalid_per_episode": tot / n, "invalid_per_turn": tot / (21 * n),
            "episodes_with_invalid": sum(1 for v in invalid.values() if v > 0) / n}


def paired_ci(a, b, n_boot=10000, seed=0):
    """CI for mean(a - b) by paired bootstrap over tasks."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main(out):
    base = load_eval(os.path.join(out, "eval_base"))
    if base is None:
        sys.exit("no eval_base traces found")
    def inv_str(ev):
        return f"{ev['invalid_per_episode']:.3f}/ep ({ev['episodes_with_invalid']*100:.1f}% eps, {ev['invalid_per_turn']*100:.2f}% turns)"

    print(f"untrained (eval_base): n={base['n']} mean={np.mean(list(base['scores'].values())):+.4f} invalid {inv_str(base)}")
    print("scripted baselines on the same 400 worlds:", BASELINES)
    print()
    rows = []
    for d in sorted(glob.glob(os.path.join(out, "eval_step_*")), key=lambda p: int(p.rsplit("_", 1)[-1])):
        ev = load_eval(d)
        if ev is None:
            print(f"{os.path.basename(d)}: no traces"); continue
        common = sorted(set(ev["scores"]) & set(base["scores"]))
        a = [ev["scores"][k] for k in common]; b = [base["scores"][k] for k in common]
        diff, lo, hi = paired_ci(a, b)
        mean = float(np.mean(list(ev["scores"].values())))
        ci_pass = (diff > 0) and (lo > 0)
        beats = [k for k, v in BASELINES.items() if mean > v]
        rows.append((os.path.basename(d), ev["n"], mean, diff, lo, hi, inv_str(ev), ci_pass, beats))
    print(f"{'checkpoint':14s} {'n':>4s} {'mean':>9s} {'vs base':>9s} {'95% CI (paired)':>22s}  CI>0?  invalid replies                          beats")
    for name, n, mean, diff, lo, hi, inv, ci_pass, beats in rows:
        print(f"{name:14s} {n:4d} {mean:+9.4f} {diff:+9.4f}   [{lo:+.4f}, {hi:+.4f}]  {'YES' if ci_pass else 'no ':4s}  {inv:40s} {', '.join(beats) or '-'}")
    print()
    print("Pass bar (PREREG-LLM-GYM Stage 0): mean > untrained with bootstrap CI excluding 0, AND invalid replies < 5%.")
    print("The prereg does not say whether '5%' is per episode, per turn, or share of episodes; all three are shown. Truncation % is in each eval's SUCCESS line.")


if __name__ == "__main__":
    main(sys.argv[1])
