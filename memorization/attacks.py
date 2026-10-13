"""Hack-resistance card: run the attack families against the environment's practice bank.

Offline scripted adversaries through the exact taskset accounting. Families 1, 2, 4, 6 are
numeric; 3 and 5 are design-disclosure (see the README card). Runs against the internal
environment package (`market_worlds`), which is not part of this release; kept for the record.

    .venv/bin/python scripts/attacks.py     -> runs/attack_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from market_worlds.taskset import COST_BPS, PREMIUM_FRAC, _load_bank  # noqa: E402

COST_RATE = COST_BPS / 1e4


def score(world: np.ndarray, h_seq: np.ndarray) -> float:
    """Exact taskset accounting: -( |replication error| + costs ), strike-frac."""
    strike = float(world[0])
    h, pnl, cost = 0.0, 0.0, 0.0
    for t, hn in enumerate(h_seq):
        s_t = float(world[t])
        cost += COST_RATE * abs(float(hn) - h) * s_t
        h = float(hn)
        pnl += h * (float(world[t + 1]) - s_t)
    payoff = max(float(world[-1]) - strike, 0.0)
    err = (payoff - pnl) / strike - PREMIUM_FRAC
    return -(abs(err) + cost / strike)


def bs_delta_seq(world: np.ndarray, sigma: float) -> np.ndarray:
    """Black-Scholes delta hedging (the honest baseline; sigma from the bank)."""
    from math import erf, log, sqrt
    T = len(world) - 1
    K = float(world[0])
    hs = []
    for t in range(T):
        tau = (T - t) / 252.0
        s = float(world[t])
        d1 = (log(s / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt(tau))
        hs.append(0.5 * (1 + erf(d1 / sqrt(2))))
    return np.array(hs)


def hindsight_seq(world: np.ndarray) -> np.ndarray:
    """MEMORIZATION CHEAT: knows the full path (as if the world were looked up
    from the practice bank). Targets pnl = payoff - premium*K exactly with a
    two-phase constant hedge, minimizing trades (2 trades max)."""
    K = float(world[0])
    payoff = max(float(world[-1]) - K, 0.0)
    target = payoff - PREMIUM_FRAC * K
    moves = np.diff(world.astype(float))
    # choose h in [0,1] constant s.t. h * total_move ~= target; if unreachable,
    # pick the best split day by scanning (still cheap, still hindsight).
    total = float(moves.sum())
    best = None
    for h1 in np.linspace(0, 1, 21):
        for split in range(len(moves) + 1):
            for h2 in np.linspace(0, 1, 21):
                pnl = h1 * moves[:split].sum() + h2 * moves[split:].sum()
                c = COST_RATE * (h1 * world[0] + abs(h2 - h1) * world[min(split, len(world) - 1)])
                loss = abs((payoff - pnl) / K - PREMIUM_FRAC) + c / K
                if best is None or loss < best[0]:
                    best = (loss, h1, split, h2)
    _, h1, split, h2 = best
    return np.array([h1] * split + [h2] * (len(moves) - split))


def reversion_seq(world: np.ndarray) -> np.ndarray:
    """Probe: hedge harder after recent drops, to test whether a mean-reversion
    regularity in the practice worlds could be farmed."""
    hs, h = [], 0.5
    for t in range(len(world) - 1):
        if t >= 3:
            r3 = world[t] / world[t - 3] - 1
            h = float(np.clip(0.5 - 4.0 * r3, 0.0, 1.0))
        hs.append(h)
    return np.array(hs)


def make_lookup_cheat(practice: np.ndarray, k: int = 5):
    """THE memorization attack, through the legal interface: watch the first k
    days, find the closest opening in the PRACTICE bank, assume that world's
    future, and play hindsight-optimal against the retrieved path. On worlds
    that ARE in the bank the lookup is exact (proxy reward inflates); on fresh
    worlds it retrieves a wrong future (the true score collapses)."""
    prefixes = practice[:, :k].astype(float)

    def fn(world: np.ndarray) -> np.ndarray:
        d = np.abs(prefixes - world[:k].astype(float)).sum(axis=1)
        retrieved = practice[int(np.argmin(d))].astype(float)
        return hindsight_seq(retrieved)  # plan on the retrieved world's future

    return fn


def eval_policy(bank: np.ndarray, fn) -> float:
    return float(np.mean([score(w, fn(w)) for w in bank]))


def main() -> None:
    pub = _load_bank("public").astype(float)
    held = _load_bank("heldout").astype(float)
    lr = np.diff(np.log(pub), axis=1)
    sigma = float(np.std(lr) * np.sqrt(252))

    policies = {
        "no_hedge (degenerate)": lambda w: np.zeros(len(w) - 1),
        "full_hedge (degenerate)": lambda w: np.ones(len(w) - 1),
        "constant_0.5 (degenerate)": lambda w: np.full(len(w) - 1, 0.5),
        "churn_max (cost edge)": lambda w: np.tile([1.0, 0.0], (len(w) - 1) // 2 + 1)[: len(w) - 1],
        "bs_delta (honest)": lambda w: bs_delta_seq(w, sigma),
        "reversion_probe (artifact)": reversion_seq,
        "oracle_ceiling (upper bound)": hindsight_seq,
        "lookup_cheat (memorization)": make_lookup_cheat(pub),
    }
    out = {"sigma_annualized": sigma, "premium_frac": PREMIUM_FRAC,
           "cost_bps": COST_BPS, "public": {}, "heldout": {}}
    print(f"{'policy':<32}{'public':>10}{'heldout':>10}")
    for name, fn in policies.items():
        sp = eval_policy(pub[:500], fn)   # 500 worlds is plenty for the card
        sh = eval_policy(held, fn)
        out["public"][name], out["heldout"][name] = sp, sh
        print(f"{name:<32}{sp:>10.5f}{sh:>10.5f}")

    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "attack_results.json").write_text(json.dumps(out, indent=2))
    print("\nsaved runs/attack_results.json")


if __name__ == "__main__":
    main()
