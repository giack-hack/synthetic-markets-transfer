# Pre-registration (excerpts, verbatim)

These are the sections of the pre-registered plan that govern the results in this repo. The full plan, with every dated amendment, lives in the private experiments repo; commit hashes there: 9d3ca54 (2026-10-05, plan + Stage 0 bar), 056a7ec (2026-10-08, transfer read written before any exam score). Nothing inside the quoted text was edited after the runs; passages left out (parts of the plan about other work, and internal file paths) are marked […].

## The game (unchanged from market-worlds)

Short one ATM call, 21 trading days, daily reply `[HEDGE 0.00–1.00]`, 5 bps
costs, reward = −(|payoff − premium − hedge P&L| / strike + costs). […]
Single asset per episode; assets IWM, QQQ, SPY, TLT.

**Anti-recognition rule (LLMs have read about 2008):** the model sees only
prices normalized to 100 at day 0 (banks store raw levels; the harness divides
by day-0 price). No dates, no tickers, no asset names, no news. The prompt
never states whether a path is synthetic or real (market-worlds' "this is a
synthetic path" sentence is removed — it would be false […] for the exam).
Same prompt text for all arms and the exam.

**Premium leak (accepted, documented):** the premium shown in the prompt is
per asset (TLT ≈1.5%, IWM ≈2.6%), which hints at the asset class. Path
volatility leaks it too. We accept this: the model still cannot know the
*date*, which is what matters for recognizing a crisis. Alternative (hide the
premium entirely, reward only on error+cost) is rejected because the premium
is part of the deep-hedging accounting.

## The exam

Real 2008, 2020, 2022 calendar-year windows per asset […]. Stride 5 overlapping = 47/47/46
windows per asset (188/188/184 pooled); stride 21 non-overlapping = 11/12/11
per asset for the check. Never shown in training. Model decodes greedily
(temperature 0) at exam time. Invalid replies: one retry, then keep the
previous hedge (market-worlds rule), counted and reported.

**Primary metric:** pooled ES₉₀ (average of the worst 10%) of per-episode
|error| + cost […]. Moving-block bootstrap (block = 5 dates,
carrying all assets per date) + non-overlapping-window check. Mean error and
invalid-reply rate reported alongside.

## Stage 0 — Pilot […]

One arm (L_flow), one seed, small model (~1.5–4B), one GPU. **The pilot never
touches the crisis exam** — it trains on the 2020/2022 L_flow bank (walled at
2019, vol-matched, per-asset premium […]) and is scored on 400 FLOW worlds held back from
that same bank. No real price ever enters the pilot.

Pilot passes if: trained model beats L_base on heldout synthetic worlds
(mean score, bootstrap CI excluding zero) AND invalid replies < 5%.
- Pass → […]
- Fail → try once with a ~7B model. Fail again → stop; verdict "LLMs do not learn
  this game at this scale" (inconclusive for FLOW, reported as such).

## Amendment 2026-10-08 (secondary transfer read; written and committed BEFORE looking at any exam score)

Stage 0 scored the trained model only on held-out FLOW worlds, by design. The environment thesis needs a third
fact alongside "the agent learns" and "the worlds cannot be memorised": that skill learned only in synthetic worlds
shows up on real markets. This read supplies it […].

- **What is scored:** the Stage-0 adapters […]:
  step_60 (primary), step_0 (the zero adapter = untrained model through the same LoRA serving path, the control),
  step_45 (secondary). Qwen3-1.7B, greedy, the existing `pilot_eval.toml` pointed at `exam_2020_overlap` and
  `exam_2022_overlap` (188 windows each, stride 5). Same prompt, same accounting.
- **Primary contrast:** step_60 vs step_0, per crisis, paired by window, moving-block bootstrap over window-start
  dates (block 5, all assets carried, 10,000 resamples). Mean loss and ES90 both reported; the pass bar is on the
  mean (the Stage-0 metric), ES90 is the secondary read.
- **Pass:** step_60 better than step_0 with CI excluding zero on BOTH 2020 and 2022.
  **Null:** CI includes zero on both. **One crisis only:** reported as inconclusive, not a pass.
- **Context, not part of the bar:** the scripted players (no-hedge, constant-0.5, BS-delta) on the same windows.
- **What a pass does not mean:** it does not show the synthetic classroom beat a free one […], and it
  is one seed of a 1.7B model.
- […]
