"""Running one commit rule over the problems of one benchmark.

The answers to the ``i``-th problem of a shard are drawn with seed
``seed + i``.  In the paper every benchmark runs as one shard except MBPP,
which runs as two shards of 250 problems (``configs/table1.yaml``); keeping the
shards reproduces the random draws of the paper's runs.

Output: one JSON line per problem with the answers, the forward passes spent
on them, and the decoding time.  An interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
from pathlib import Path

from .decoding import decode_question
from .eval.benchmarks import shard
from .rules.base import CommitRule


def completed_problems(path: Path) -> set:
    """Ids of the problems already written to ``path``."""
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["problem_id"] for line in handle if line.strip()}


def run_rule(lm, rule: CommitRule, problems: list[dict], out_path: str | Path, *,
             num_samples: int = 8, gen_length: int = 256, block_length: int = 256,
             temperature: float = 0.6, seed: int = 113, num_shards: int = 1,
             log_every: int = 25) -> Path:
    """Decode every problem with ``rule`` and append the answers to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_problems(out_path)
    with out_path.open("a", encoding="utf-8") as handle:
        for shard_id in range(num_shards):
            for index, problem in enumerate(shard(problems, shard_id, num_shards)):
                if problem["id"] in done:
                    continue
                answers, result = decode_question(
                    lm, problem["question"], rule, num_samples=num_samples,
                    gen_length=gen_length, block_length=block_length,
                    temperature=temperature, seed=seed + index)
                handle.write(json.dumps({
                    "problem_id": problem["id"], "shard": shard_id, "answers": answers,
                    "forward_passes": result.forward_passes,
                    "seconds": round(result.seconds, 4),
                }, ensure_ascii=False) + "\n")
                handle.flush()
                if log_every and (index + 1) % log_every == 0:
                    print(f"  [{rule.name}] shard {shard_id}: {index + 1} problems, "
                          f"last {result.forward_passes} forward passes", flush=True)
    return out_path


def read_answers(paths) -> list[dict]:
    """Records written by :func:`run_rule`, from one or more files."""
    records = []
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        with Path(path).open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records
