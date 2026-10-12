"""Smoke test of the whole pipeline on CPU: decode -> grade -> table, with a fake model."""

import json
import subprocess
import sys
from pathlib import Path

import torch
from test_decoding import MASK, TinyModel

from pact.eval import load_problems
from pact.rules import make_rule
from pact.runner import read_answers, run_rule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from decode import write_run_info  # noqa: E402
from evaluate import evaluate  # noqa: E402


class FakeLM:
    """The interface of pact.models.DiffusionLM, backed by a tiny random model."""

    mask_id = MASK
    valid_vocab = None

    def __init__(self):
        self.logits = TinyModel()

    def encode_prompt(self, question: str) -> torch.Tensor:
        return torch.tensor([len(question) % 9, 1, 2])

    def decode_answers(self, answer_ids: torch.Tensor) -> list[str]:
        return ["The answer is " + "".join(str(int(t)) for t in row[:2]) for row in answer_ids]


def test_decode_grade_and_summarize(tmp_path):
    problems = load_problems("gsm8k", num_problems=3)
    rule = make_rule("pact", tau=0.3, gap=4)
    decoding = dict(num_samples=4, gen_length=16, block_length=16, temperature=0.6, seed=113)
    out = tmp_path / "toy" / "fake" / "gsm8k" / "pact"
    write_run_info(out, model="fake", benchmark="gsm8k", rule=rule, decoding=decoding,
                   num_shards=1, num_problems=3)
    run_rule(FakeLM(), rule, problems, out / "gen.jsonl", **decoding)
    records = read_answers(out / "gen.jsonl")
    assert [r["problem_id"] for r in records] == [p["id"] for p in problems]
    assert all(len(r["answers"]) == 4 and 1 <= r["forward_passes"] <= 16 for r in records)

    run_rule(FakeLM(), rule, problems, out / "gen.jsonl", **decoding)   # resumes: nothing new
    assert len(read_answers(out / "gen.jsonl")) == 3

    summary = evaluate(out)
    assert summary["problems"] == 3 and summary["answers_per_problem"] == 4
    assert 0.0 <= summary["pass@1"] <= 1.0 and summary["pass@8"] is None
    assert summary["speed"] == 16 / summary["forward_passes"]
    assert json.loads((out / "summary.json").read_text())["rule"] == "pact"

    table = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), str(tmp_path / "toy")],
                           capture_output=True, text=True, check=True).stdout
    assert "GSM8K Acc." in table and "| fake | pact |" in table
