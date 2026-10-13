#!/bin/bash
# Reproduces every number in README.md from the committed traces (CPU, ~1 min).
set -e; cd "$(dirname "$0")/.."
python scripts/transfer_analyze.py runs/transfer --banks banks
python scripts/score_pilot.py runs/stage0_pilot
python scripts/trained_vs_untrained.py
python scripts/baselines.py exam_2020_overlap exam_2022_overlap
