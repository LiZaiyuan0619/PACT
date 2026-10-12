#!/usr/bin/env bash
# Table 1 end to end: decode every cell, grade, and print the table.
#
#   bash scripts/reproduce_table1.sh                       # all 160 runs
#   bash scripts/reproduce_table1.sh --models llada-8b     # one model (40 runs)
#
# Extra arguments go to scripts/run_config.py (--models, --benchmarks, --runs).
# Decoding runs on one GPU (every run of the paper used one NVIDIA H200);
# grading runs on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/run_config.py configs/table1.yaml "$@"
python scripts/evaluate.py results/table1
python scripts/summarize.py results/table1
