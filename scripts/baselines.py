"""Scripted reference players on any hedge-gym bank, through the env's exact
accounting. No model. Usage: python scripts/baselines.py era2019_flow_heldout [exam_2020_overlap ...]

Prints mean score, ES90 of loss (|error|+cost) per player, and a per-asset table.
BS-delta uses each asset's annualized vol from the bank's metadata (sqrt(252) x daily std)."""
import json, sys
from math import erf, log, sqrt
from pathlib import Path
import numpy as np

BANKS = Path(__file__).resolve().parent.parent / "banks"
COST = 5.0 / 1e4
EPIS = 21


def N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_delta(s, k, tau_days, sigma_ann):
    if tau_days <= 0:
        return 1.0 if s > k else 0.0
    tau = tau_days / 252.0
    d1 = (log(s / k) + 0.5 * sigma_ann**2 * tau) / (sigma_ann * sqrt(tau))
    return N(d1)


def play(prices, premium, policy):
    """policy(t, prices, h) -> hedge in [0,1]. Returns (score, |error|+cost)."""
    k = prices[0]; h = 0.0; pnl = 0.0; cost = 0.0
    for t in range(len(prices) - 1):
        a = float(np.clip(policy(t, prices, h), 0.0, 1.0))
        cost += COST * abs(a - h) * prices[t]; h = a
        pnl += h * (prices[t + 1] - prices[t])
    payoff = max(prices[-1] - k, 0.0)
    err = (payoff - pnl) / k - premium
    loss = abs(err) + cost / k
    return -loss, loss


def es(x, alpha=0.90):
    x = np.sort(np.asarray(x)); n = int(np.ceil((1 - alpha) * len(x)))
    return float(x[-n:].mean())


def main(bank_names):
    for name in bank_names:
        z = np.load(BANKS / f"{name}.npz"); meta = json.loads(str(z["meta"]))
        P, A, PR = z["prices"], z["asset"], z["premium"]
        assets = meta["assets"]; tg = meta["targets"]
        sig = {i: sqrt(252.0) * tg[a]["daily"] for i, a in enumerate(assets)}
        players = {
            "no_hedge": lambda t, p, h, ai: 0.0,
            "const_0.5": lambda t, p, h, ai: 0.5,
            "bs_delta": lambda t, p, h, ai: bs_delta(p[t], p[0], EPIS - t, sig[ai]),
        }
        print(f"\n== {name}: {len(P)} tasks, era {meta.get('era')} arm {meta.get('arm')}")
        print(f"{'player':10s} {'mean score':>11s} {'ES90 loss':>10s}  " + "  ".join(f"{a:>7s}" for a in assets))
        for pname, pol in players.items():
            scores, losses, per = [], [], {i: [] for i in range(len(assets))}
            for i in range(len(P)):
                ai = int(A[i])
                s, l = play(P[i], float(PR[i]), lambda t, p, h: pol(t, p, h, ai))
                scores.append(s); losses.append(l); per[ai].append(s)
            print(f"{pname:10s} {np.mean(scores):>11.4f} {es(losses):>10.4f}  " +
                  "  ".join(f"{np.mean(per[i]):>7.4f}" for i in range(len(assets))))


if __name__ == "__main__":
    main(sys.argv[1:] or ["era2019_flow_heldout"])
