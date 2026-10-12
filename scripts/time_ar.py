#!/usr/bin/env python3
"""Time the autoregressive reference, Qwen3-8B, on the same problems (Appendix D.4).

    python scripts/time_ar.py --out results/ar_comparison/qwen3-8b.json

Qwen3-8B decodes with its key-value cache through Hugging Face ``generate``,
writes at most 256 tokens, and runs without its thinking mode.  ``greedy``
gives one answer per problem (Appendix D.4); ``sampled`` draws eight answers
at temperature 0.6 in one call (the comparison in Section 4.2).  Run it on
the GPU that timed the diffusion models (configs/ar_comparison.yaml,
configs/timing.yaml).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pact.eval import load_problems, mean_pass_at_k, score_answers  # noqa: E402

MODEL = "Qwen/Qwen3-8B"


def run_mode(model, tokenizer, benchmark: str, problems: list[dict], mode: str,
             max_new_tokens: int, seed: int) -> dict:
    import torch
    sampled = mode == "sampled"
    answers_per_problem = 8 if sampled else 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    answers, seconds, lengths = {}, [], []
    for problem in problems:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem["question"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=sampled,
                      pad_token_id=tokenizer.eos_token_id,
                      num_return_sequences=answers_per_problem, top_p=None, top_k=None,
                      temperature=0.6 if sampled else None)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        prompt_length = inputs["input_ids"].shape[1]
        texts = []
        for sequence in generated:
            ids = sequence[prompt_length:].tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id)]
            lengths.append(len(ids))
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        answers[problem["id"]] = texts
    correct = score_answers(benchmark, {p["id"]: p for p in problems}, answers)
    passed = [correct[p["id"]] for p in problems]
    return {
        "mode": mode, "benchmark": benchmark, "problems": len(problems),
        "answers_per_problem": answers_per_problem,
        "pass@1": mean_pass_at_k(passed, 1),
        "pass@8": mean_pass_at_k(passed, 8) if sampled else None,
        "seconds_per_problem": sum(seconds) / len(seconds),
        "seconds_per_answer": sum(seconds) / len(seconds) / answers_per_problem,
        "answer_tokens": sum(lengths) / len(lengths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--benchmarks", nargs="+", default=["humaneval", "gsm8k"])
    parser.add_argument("--modes", nargs="+", default=["greedy", "sampled"],
                        choices=["greedy", "sampled"])
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=113)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("a GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    results = {"model": args.model, "gpu": torch.cuda.get_device_name(0),
               "max_new_tokens": args.max_new_tokens, "runs": []}
    for mode in args.modes:
        for benchmark in args.benchmarks:
            problems = load_problems(benchmark, num_problems=args.num_problems)
            result = run_mode(model, tokenizer, benchmark, problems, mode,
                              args.max_new_tokens, args.seed)
            results["runs"].append(result)
            print(f"[{mode} {benchmark}] pass@1 {100 * result['pass@1']:.1f}  "
                  f"{result['seconds_per_problem']:.2f} s/problem", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
