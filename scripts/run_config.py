#!/usr/bin/env python3
"""Run every (model, benchmark, setting) of a config.

    python scripts/run_config.py configs/table1.yaml --dry_run
    python scripts/run_config.py configs/table1.yaml --models llada-8b --benchmarks gsm8k
    python scripts/run_config.py configs/sweeps.yaml --runs pact-tau0.3 fast_dllm-threshold0.9

Results go to ``<results>/<config name>/<model>/<benchmark>/<run>/``.  Each
model is loaded once.  Finished problems are skipped, so the command can be
re-run after an interruption.  The full Table 1 is 160 runs; select subsets
with --models, --benchmarks and --runs to split the work across GPUs.
"""

from __future__ import annotations

import argparse
import sys
from itertools import groupby
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from decode import write_run_info  # noqa: E402
from pact.config import expand, load_config  # noqa: E402
from pact.eval.benchmarks import load_problems  # noqa: E402
from pact.models import load_model  # noqa: E402
from pact.rules import make_rule  # noqa: E402
from pact.runner import run_rule  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--benchmarks", nargs="*")
    parser.add_argument("--runs", nargs="*", help="run names (see --dry_run)")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true", help="list the runs and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    runs = expand(config, models=args.models, benchmarks=args.benchmarks, names=args.runs)
    if not runs:
        raise SystemExit("no run matches the filters")
    for run in runs:
        print(f"{run.model:14s} {run.benchmark:10s} {run.name:28s} {run.rule} {run.params} "
              f"block={run.decoding['block_length']} T={run.decoding['temperature']} "
              f"n={run.decoding['num_samples']} shards={run.num_shards}")
    print(f"{len(runs)} runs")
    if args.dry_run:
        return 0

    for model, model_runs in groupby(runs, key=lambda r: r.model):
        lm = load_model(model, device=args.device)
        for run in model_runs:
            out = args.results / config["name"] / run.model / run.benchmark / run.name
            rule = make_rule(run.rule, **run.params)
            problems = load_problems(run.benchmark, args.data_dir, run.num_problems)
            write_run_info(out, model=run.model, benchmark=run.benchmark, rule=rule,
                           decoding=run.decoding, num_shards=run.num_shards,
                           num_problems=run.num_problems)
            print(f"[run] {out}", flush=True)
            run_rule(lm, rule, problems, out / "gen.jsonl", num_shards=run.num_shards,
                     **run.decoding)
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
