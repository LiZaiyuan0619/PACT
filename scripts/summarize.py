#!/usr/bin/env python3
"""Collect graded runs into the layout of Table 1.

    python scripts/summarize.py results/table1
    python scripts/summarize.py results/sweeps --metric pass@8 --format tsv

One block per model, one row per run, and for each benchmark the accuracy
(pass@1 in %) and the speed (tokens written per forward pass).  Run
scripts/evaluate.py first.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

BENCHMARK_ORDER = ["humaneval", "mbpp", "gsm8k", "math500", "ifeval"]
BENCHMARK_NAMES = {"humaneval": "HumanEval", "mbpp": "MBPP", "gsm8k": "GSM8K",
                   "math500": "MATH500", "ifeval": "IFEval"}
MODEL_ORDER = ["llada-8b", "llada-moe", "llada2.1-mini", "dream-7b"]
RUN_ORDER = ["one_token_per_step", "fast_dllm", "fast_dllm_pp", "eb_sampler", "klass",
             "tacg", "fixed_k", "pact"]


def order_key(order: list[str]):
    return lambda name: (order.index(name) if name in order else len(order), name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--metric", default="pass@1", choices=["pass@1", "pass@8"])
    parser.add_argument("--format", default="markdown", choices=["markdown", "tsv"])
    args = parser.parse_args()

    cells = defaultdict(dict)                     # (model, run) -> benchmark -> summary
    for path in sorted(args.results.rglob("summary.json")):
        summary = json.loads(path.read_text())
        run = path.parent.name
        cells[(summary["model"], run)][summary["benchmark"]] = summary
    if not cells:
        raise SystemExit("no summary.json found; run scripts/evaluate.py first")
    benchmarks = sorted({b for row in cells.values() for b in row},
                        key=order_key(BENCHMARK_ORDER))
    models = sorted({m for m, _ in cells}, key=order_key(MODEL_ORDER))

    header = ["Model", "Run"] + [f"{BENCHMARK_NAMES.get(b, b)} {col}"
                                 for b in benchmarks for col in ("Acc.", "Speed")]
    lines = []
    for model in models:
        runs = sorted({r for m, r in cells if m == model}, key=order_key(RUN_ORDER))
        for run in runs:
            row = [model, run]
            for benchmark in benchmarks:
                summary = cells[(model, run)].get(benchmark)
                value = None if summary is None else summary[args.metric]
                row.append("-" if value is None else f"{100 * value:.1f}")
                row.append("-" if summary is None else f"{summary['speed']:.1f}x")
            lines.append(row)
    if args.format == "tsv":
        print("\t".join(header))
        for row in lines:
            print("\t".join(row))
    else:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in lines:
            print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
