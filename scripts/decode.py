#!/usr/bin/env python3
"""Decode the problems of one benchmark with one commit rule.

    python scripts/decode.py --model llada-8b --benchmark gsm8k \
        --rule pact --param tau=0.5 --param gap=8 \
        --out results/demo/llada-8b/gsm8k/pact

Writes ``run.json`` (the settings) and ``gen.jsonl`` (one line per problem:
answers, forward passes, seconds) to ``--out``; grade them with
scripts/evaluate.py.  An interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pact.config import DECODING_DEFAULTS  # noqa: E402
from pact.eval.benchmarks import BENCHMARKS, load_problems  # noqa: E402
from pact.models import load_model, resolve_model  # noqa: E402
from pact.rules import RULES, make_rule  # noqa: E402
from pact.runner import run_rule  # noqa: E402


def parse_params(items: list[str]) -> dict:
    params = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--param expects KEY=VALUE, got {item!r}")
        params[key] = yaml.safe_load(value)
    return params


def write_run_info(out: Path, *, model: str, benchmark: str, rule, decoding: dict,
                   num_shards: int, num_problems: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    info = {"model": model, "checkpoint": resolve_model(model), "benchmark": benchmark,
            **rule.describe(), "decoding": decoding, "num_shards": num_shards,
            "num_problems": num_problems}
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True,
                        help="llada-8b, llada-moe, llada2.1-mini, dream-7b, or a checkpoint path")
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--rule", required=True, choices=sorted(RULES))
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="rule parameter, e.g. --param tau=0.5 (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num_problems", type=int, default=0, help="first N problems (0: all)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="decode in shards; the paper used 2 for MBPP and 1 elsewhere")
    for key, value in DECODING_DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rule = make_rule(args.rule, **parse_params(args.param))
    decoding = {key: getattr(args, key) for key in DECODING_DEFAULTS}
    problems = load_problems(args.benchmark, args.data_dir, args.num_problems)
    write_run_info(args.out, model=args.model, benchmark=args.benchmark, rule=rule,
                   decoding=decoding, num_shards=args.num_shards,
                   num_problems=args.num_problems)
    print(f"[decode] {args.model} {args.benchmark} {rule.describe()} "
          f"on {len(problems)} problems", flush=True)
    lm = load_model(args.model, device=args.device)
    run_rule(lm, rule, problems, args.out / "gen.jsonl", num_shards=args.num_shards, **decoding)
    print(f"[decode] wrote {args.out / 'gen.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
