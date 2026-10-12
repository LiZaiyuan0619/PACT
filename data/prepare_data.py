#!/usr/bin/env python3
"""Build the five problem files under data/ from their public sources.

    python data/prepare_data.py            # writes data/<benchmark>.jsonl
    python data/prepare_data.py --check    # rebuilds in memory and compares

HumanEval, MBPP, GSM8K and MATH500 are downloaded from the Hugging Face Hub.
IFEval is read from the official input file shipped with the checker under
third_party/instruction_following_eval/data/input_data.jsonl.  The files in
this directory were produced by this script; see data/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pact.eval.benchmarks import (  # noqa: E402
    GSM8K_PROBLEMS, GSM8K_SHUFFLE_SEED, IFEVAL_PROBLEMS, gsm8k_gold, humaneval_question,
    math500_question, mbpp_entry_point, mbpp_question,
)

IFEVAL_INPUT = ROOT / "third_party/instruction_following_eval/data/input_data.jsonl"


def build_humaneval() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = {ex["task_id"]: ex for ex in ds}
    return [{"id": tid, "question": humaneval_question(rows[tid]),
             "entry_point": rows[tid]["entry_point"], "prompt": rows[tid]["prompt"],
             "test": rows[tid]["test"]}
            for tid in sorted(rows)]


def build_mbpp() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    rows = sorted(ds, key=lambda ex: int(ex["task_id"]))
    return [{"id": int(ex["task_id"]), "question": mbpp_question(ex),
             "entry_point": mbpp_entry_point(ex), "test_list": list(ex["test_list"])}
            for ex in rows]


def build_gsm8k() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.add_column("source_index", list(range(len(ds))))
    ds = ds.shuffle(seed=GSM8K_SHUFFLE_SEED)
    return [{"id": f"test:{ds[i]['source_index']}", "question": ds[i]["question"],
             "gold": gsm8k_gold(ds[i]["answer"])}
            for i in range(GSM8K_PROBLEMS)]


def build_math500() -> list[dict]:
    from datasets import load_dataset
    from pact.eval.math_parser import extract_answer
    ds = load_dataset("ankner/math-500", split="test")
    return [{"id": i, "question": math500_question(ds[i]["problem"]),
             "gold": extract_answer(ds[i]["solution"])}
            for i in range(len(ds))]


def build_ifeval() -> list[dict]:
    with IFEVAL_INPUT.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [{"id": int(ex["key"]), "question": ex["prompt"],
             "instruction_id_list": list(ex["instruction_id_list"]), "kwargs": list(ex["kwargs"])}
            for ex in rows[:IFEVAL_PROBLEMS]]


BUILDERS = {"humaneval": build_humaneval, "mbpp": build_mbpp, "gsm8k": build_gsm8k,
            "math500": build_math500, "ifeval": build_ifeval}


def dumps(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmarks", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--check", action="store_true",
                        help="compare with the existing files instead of writing")
    args = parser.parse_args()
    status = 0
    for name in args.benchmarks:
        text = dumps(BUILDERS[name]())
        path = args.out / f"{name}.jsonl"
        if args.check:
            same = path.is_file() and path.read_text(encoding="utf-8") == text
            print(f"{name}: {'identical' if same else 'DIFFERENT'}")
            status |= 0 if same else 1
        else:
            path.write_text(text, encoding="utf-8")
            print(f"{name}: {text.count(chr(10))} problems -> {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
