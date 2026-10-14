"""hedge-gym — option-hedging environment (verifiers v1).

Short-call hedging game with a per-task premium (per asset, the mean payoff over
the era's real history), banks stored normalized to 100, and a prompt that never
says whether the path is synthetic or real. The bank selects the world set.

Same game, accounting and prompt as `hedge_gym/` in
https://github.com/giack-hack/synthetic-markets-transfer; this copy bundles the banks.

Banks shipped in this package (hedge_gym/data/):
  era2019_flow_heldout   400 synthetic worlds (100 per asset), never trained on
  exam_2020_overlap      188 real 21-day windows of 2020 (47 per asset, 5-day grid)
  exam_2022_overlap      184 real 21-day windows of 2022 (46 per asset, 5-day grid)
Set HEDGE_GYM_BANKS=<dir> to load other banks (the training bank is available on request).
"""
import json
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import Field

import verifiers.v1 as vf

COST_BPS = 5.0
HEDGE_RE = re.compile(r"\[HEDGE\s+(1(?:\.0+)?|0(?:\.\d+)?)\]", re.IGNORECASE)

RULES = """You are a derivatives trader hedging a short option position, day by day.

The game: you sold one call option, strike 100 (= day-0 price), expiring on day 21.
You were paid a premium of {premium:.2f} for it. Each day you choose how many shares
of the underlying to hold long, as a hedge ratio between 0 and 1. Changing your hedge
costs {cost_bps:.0f} basis points of the traded value.

Objective: minimize |option payoff - premium - your hedging P&L| + trading costs.
A perfect hedge replicates the option; doing nothing leaves you naked. Prices are
shown rebased to 100 on day 0; you are not told what the asset or the period is.

Each day you will see the price so far and your current hedge. Reply with at most
two short sentences of reasoning, then EXACTLY ONE action in square brackets, e.g.
[HEDGE 0.55]. The number must be between 0.00 and 1.00. Only the bracketed
action counts. /no_think"""


def _bank_file(name: str):
    """Bank `.npz`: from HEDGE_GYM_BANKS if set, else the copy bundled in this package."""
    override = os.environ.get("HEDGE_GYM_BANKS")
    if override:
        return Path(override) / f"{name}.npz"
    return files("hedge_gym").joinpath("data", f"{name}.npz")


def parse_hedge(reply: str) -> Optional[float]:
    found = HEDGE_RE.findall(reply)
    return float(found[0]) if len(found) == 1 else None


class TaskData(vf.TaskData):
    info: dict
    prices: list[float]
    premium: float


class HedgeGymTask(vf.Task[TaskData]):
    pass


class HedgeGymConfig(vf.TasksetConfig):
    bank: str = "era2019_flow_heldout"
    """World set: era2019_flow_heldout (synthetic), exam_2020_overlap or exam_2022_overlap (real)."""
    num_tasks: int = Field(0, ge=0)
    """0 = all tasks in the bank."""
    shuffle_seed: int = 0


class HedgeGymEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, max_turns=48)
    invalid_retries: int = Field(1, ge=0)


class HedgeGymEnv(vf.Env[HedgeGymEnvConfig]):
    async def run(self, task, agents):
        prices = task.data.prices
        premium = task.data.premium
        strike = prices[0]
        cost_rate = COST_BPS / 1e4
        h, hedge_pnl, total_cost, invalid = 0.0, 0.0, 0.0, 0

        async def ask(interaction, t: int):
            recent = ", ".join(f"{p:.2f}" for p in prices[max(0, t - 9): t + 1])
            prompt = (
                f"Day {t + 1} of {len(prices) - 1}. Prices so far (last 10): {recent}. "
                f"Current price: {prices[t]:.2f}. Your current hedge: {h:.2f}. "
                f"Choose today's hedge: reply with [HEDGE 0.00-1.00]."
            )
            for _ in range(self.config.invalid_retries + 1):
                segment = await interaction.turn(prompt)
                if segment.terminated:
                    return None
                action = parse_hedge(segment.last_reply)
                if action is not None:
                    return action
                prompt = ("That was not a valid action. Reply with exactly one "
                          "bracketed hedge between 0 and 1, e.g. [HEDGE 0.40].")
            return -1.0

        async with agents.agent.interaction(task) as interaction:
            for t in range(len(prices) - 1):
                action = await ask(interaction, t)
                if action is None:
                    break
                if action < 0:
                    invalid += 1
                else:
                    total_cost += cost_rate * abs(action - h) * prices[t]
                    h = action
                hedge_pnl += h * (prices[t + 1] - prices[t])

        payoff = max(prices[-1] - strike, 0.0)
        error_frac = (payoff - hedge_pnl) / strike - premium
        cost_frac = total_cost / strike
        trace = interaction.trace
        trace.record_reward("hedge_score", -(abs(error_frac) + cost_frac))
        trace.record_metric("replication_error", abs(error_frac))
        trace.record_metric("total_cost_frac", cost_frac)
        trace.record_metric("invalid_replies", float(invalid))
        trace.info["hedge_gym"] = dict(task.data.info, final_hedge=h, replication_error_signed=error_frac)


class HedgeGymTaskset(vf.Taskset[HedgeGymTask, HedgeGymConfig]):
    def load(self) -> list[HedgeGymTask]:
        with _bank_file(self.config.bank).open("rb") as f:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            P, A, PR = z["prices"], z["asset"], z["premium"]
        order = np.arange(len(P))
        if self.config.shuffle_seed:
            np.random.default_rng(self.config.shuffle_seed).shuffle(order)
        if self.config.num_tasks:
            order = order[: self.config.num_tasks]
        tasks = []
        for i in order:
            prem = float(PR[i])
            tasks.append(HedgeGymTask(
                TaskData(
                    idx=int(i),
                    name=f"{self.config.bank}#{int(i):05d}",
                    prompt=None,
                    system_prompt=RULES.format(premium=prem * 100, cost_bps=COST_BPS),
                    prices=[round(float(p), 4) for p in P[i]],
                    premium=prem,
                    info={"bank": self.config.bank, "task_id": int(i), "asset": int(A[i]),
                          "era": meta.get("era"), "arm": meta.get("arm")},
                ),
                self.config.task,
            ))
        return tasks


__all__ = ["HedgeGymTaskset", "HedgeGymEnv"]
