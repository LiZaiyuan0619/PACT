# Problem files

Each file holds the problems of one benchmark, one JSON object per line, in the
order in which they are decoded.  Every method in the paper decodes exactly these
problems.  The files are written by `prepare_data.py`; running
`python data/prepare_data.py --check` rebuilds them from the sources below and
compares.

| File | Problems | Source | License |
|---|---|---|---|
| `humaneval.jsonl` | all 164, sorted by task id | [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval), test split | MIT |
| `mbpp.jsonl` | the 500 test problems (task ids 11-510) | [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp), `full`, test split | CC BY 4.0 |
| `gsm8k.jsonl` | first 500 of the 1,319 test problems after one shuffle with seed 113 | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), `main`, test split | MIT |
| `math500.jsonl` | all 500 | [`ankner/math-500`](https://huggingface.co/datasets/ankner/math-500), test split | MIT |
| `ifeval.jsonl` | first 500 of the 541 prompts | official input file, `third_party/instruction_following_eval/data/input_data.jsonl` | Apache-2.0 |

## Fields

All files have `id` (the benchmark's own id; for GSM8K `test:<index in the test split>`)
and `question` (the text of the single user turn).

* `humaneval.jsonl`: `entry_point`, `prompt`, `test` (the benchmark's function
  stub and tests, used for grading).
* `mbpp.jsonl`: `entry_point`, `test_list` (the three assertions used for grading).
* `gsm8k.jsonl`, `math500.jsonl`: `gold`, the reference final answer (for MATH500
  extracted from the reference solution with `pact.eval.math_parser.extract_answer`).
* `ifeval.jsonl`: `instruction_id_list`, `kwargs` for the official checker.

## Questions

The questions are built in `src/pact/eval/benchmarks.py`.  A code question states
the function to complete (MBPP also lists its tests), a MATH500 question ends
with an instruction to box the final answer, and GSM8K and IFEval questions are
used as they are.  No system message and no examples are added.
