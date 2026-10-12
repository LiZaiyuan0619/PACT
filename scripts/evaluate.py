#!/usr/bin/env python3
"""Grade decoded answers and report pass@1, pass@8, speed and time.

    python scripts/evaluate.py results/demo/llada-8b/gsm8k/pact
    python scripts/evaluate.py results/table1          # every run below the directory

For every run directory (one with ``run.json`` and ``gen.jsonl``) this writes
``scores.jsonl`` (correctness of each answer) and ``summary.json``:

* ``pass@1``, ``pass@8``: unbiased estimates over the answers of each problem;
* ``forward_passes``: mean forward passes per problem;
* ``speed``: tokens written per forward pass, ``gen_length / forward_passes``,
  equal to the speed-up over one token per step;
* ``seconds_per_problem``, ``seconds_per_answer``: decoding time.

Grading runs on CPU.  Code answers run in subprocesses (--code_workers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pact.eval import load_problems, mean_pass_at_k, score_answers, speed  # noqa: E402
from pact.runner import read_answers  # noqa: E402


def run_dirs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if (path / "gen.jsonl").is_file():
            found.append(path)
        else:
            found.extend(sorted(p.parent for p in path.rglob("gen.jsonl")))
    return found


def evaluate(run_dir: Path, *, data_dir=None, code_workers: int = 8) -> dict:
    info = json.loads((run_dir / "run.json").read_text())
    records = read_answers(run_dir / "gen.jsonl")
    problems = {p["id"]: p for p in load_problems(info["benchmark"], data_dir)}
    answers = {r["problem_id"]: r["answers"] for r in records}
    correct = score_answers(info["benchmark"], problems, answers, code_workers=code_workers)
    with (run_dir / "scores.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps({"problem_id": record["problem_id"],
                                     "correct": correct[record["problem_id"]]}) + "\n")
    passed = [correct[r["problem_id"]] for r in records]
    n = min(len(p) for p in passed)
    gen_length = info["decoding"]["gen_length"]
    seconds = [r["seconds"] for r in records]
    summary = {
        "model": info["model"], "benchmark": info["benchmark"], "rule": info["rule"],
        "params": {k: v for k, v in info.items()
                   if k not in ("model", "checkpoint", "benchmark", "rule", "decoding",
                                "num_shards", "num_problems")},
        "decoding": info["decoding"],
        "problems": len(records),
        "answers_per_problem": n,
        "pass@1": mean_pass_at_k(passed, 1),
        "pass@8": mean_pass_at_k(passed, 8) if n >= 8 else None,
        "forward_passes": float(np.mean([r["forward_passes"] for r in records])),
        "speed": speed([r["forward_passes"] for r in records], gen_length),
        "seconds_per_problem": float(np.mean(seconds)),
        "seconds_per_answer": float(np.mean(seconds)) / n,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--code_workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="re-grade runs that have a summary")
    args = parser.parse_args()
    dirs = run_dirs(args.paths)
    if not dirs:
        raise SystemExit("no gen.jsonl found")
    for run_dir in dirs:
        summary_path = run_dir / "summary.json"
        if summary_path.is_file() and not args.force \
                and summary_path.stat().st_mtime >= (run_dir / "gen.jsonl").stat().st_mtime:
            summary = json.loads(summary_path.read_text())
        else:
            summary = evaluate(run_dir, data_dir=args.data_dir, code_workers=args.code_workers)
        pass8 = summary["pass@8"]
        print(f"{run_dir}: pass@1 {100 * summary['pass@1']:.1f}  "
              f"pass@8 {'-' if pass8 is None else f'{100 * pass8:.1f}'}  "
              f"speed {summary['speed']:.2f}x  "
              f"{summary['seconds_per_answer']:.3f} s/answer  ({summary['problems']} problems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
