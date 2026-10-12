# Problem files

Each file holds the problems of one benchmark, one JSON object per line. The problems
are in the order in which they are decoded, and every method in the paper decodes
exactly these problems. The script `prepare_data.py` writes the files. Running
`python data/prepare_data.py --check` rebuilds them from the sources below and compares
the result with the files.

| File | Problems | Source | License |
|---|---|---|---|
| `humaneval.jsonl` | all 164, sorted by task id | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval), test split | MIT |
| `mbpp.jsonl` | the 500 test problems, task ids 11 to 510 | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp), config `full`, test split | CC BY 4.0 |
| `gsm8k.jsonl` | the first 500 of the 1,319 test problems after one shuffle with seed 113 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), config `main`, test split | MIT |
| `math500.jsonl` | all 500 | [ankner/math-500](https://huggingface.co/datasets/ankner/math-500), test split | MIT |
| `ifeval.jsonl` | the first 500 of the 541 prompts | the official input file in `third_party/instruction_following_eval/data/` | Apache-2.0 |

## Fields

Every file has the fields `id` and `question`. The id is the benchmark's own id, and
for GSM8K it is `test:` followed by the index in the test split. The question is the
text of the single user turn.

* `humaneval.jsonl` adds `entry_point`, `prompt` and `test`. They hold the function
  stub and the tests of the benchmark, which grading uses.
* `mbpp.jsonl` adds `entry_point` and `test_list`. Grading uses the three assertions
  in `test_list`.
* `gsm8k.jsonl` and `math500.jsonl` add `gold`, the reference final answer. For
  MATH500 it is extracted from the reference solution with
  `pact.eval.math_parser.extract_answer`.
* `ifeval.jsonl` adds `instruction_id_list` and `kwargs` for the official checker.

## Questions

The questions are built in `src/pact/eval/benchmarks.py`. A code question states the
function to complete, and an MBPP question also lists its tests. A MATH500 question
ends with an instruction to box the final answer. GSM8K and IFEval questions are used
as they are. No system message and no examples are added.
