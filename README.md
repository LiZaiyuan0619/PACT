# PACT: Certifying What Is Written in Parallel Decoding of Diffusion Language Models

This repository holds the code of the paper. It contains PACT, the commit rules we
compare it with, and the decoding loop they share. It also contains the four models,
the five benchmarks with their grading, and the configs of every experiment.

Parallel decoding speeds up masked diffusion language models by writing several
tokens per forward pass. A commit rule decides which of the drawn tokens a step
writes. Published rules decide from the predicted distributions alone. Under sampling
the written tokens are drawn, and these tests do not see them. PACT draws the tokens
first, certifies the set it will write, and spaces the positions of each step apart.

## One step of PACT

Algorithm 1 in Appendix B.1 gives one step of PACT. For each answer, the step works
on the masked positions of the current block.

1. A token `x_i` is drawn at every masked position at temperature `T`. Its
   written-word score is `s_i = log p_i(x_i)`, computed before temperature.
2. The positions are sorted by score, highest first. This gives the order `π`.
3. The set-level cut keeps the longest prefix of `π` whose scores sum to at least
   `log τ`. Its length is `k*`, or 1 if no prefix passes.
4. Spacing scans `π` and keeps a position only if it lies at least `g` away from
   every kept position. The scan stops at `k*` positions. If it ends early, the gap
   drops by one and the scan repeats.
5. The kept positions are put in the order of `π` and cut again at `log τ`. The step
   writes the drawn tokens at the positions that pass.

The code is in `src/pact/rules/pact.py`, about 65 lines without comments.

## Repository layout

```
PACT/
├── src/pact/
│   ├── rules/            PACT and the compared commit rules, one file each
│   ├── decoding.py       the decoding loop shared by every rule
│   ├── models.py         LLaDA-8B, LLaDA-MoE, LLaDA2.1-mini and Dream-7B
│   ├── runner.py         one rule over the problems of one benchmark
│   ├── config.py         experiment configs
│   └── eval/             benchmarks, grading, pass@k and speed
├── configs/              Table 1, Figure 3, Figure 4, Appendix D.2 and D.4
├── data/                 the problems of the five benchmarks and the script that builds them
├── scripts/              decoding, grading, tables and the timing of Qwen3-8B
├── tests/                CPU tests of the rules, the decoding loop and the grading
└── third_party/          the official IFEval checker and third-party licenses
```

## Installation

```bash
conda create -n pact python=3.11 -y
conda activate pact
pip install -r requirements.txt
pip install -e .
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
pytest -q
```

The file `requirements.txt` pins the versions used in the paper. The IFEval checker
needs the NLTK data. The tests run on CPU in about ten seconds.

The experiments ran on single NVIDIA H200 GPUs with Python 3.11, PyTorch 2.9.1 and
Transformers 4.57.1. Grading needs only CPUs.

## Models

| Name in configs | Checkpoint on Hugging Face | License |
|---|---|---|
| `llada-8b` | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | MIT |
| `llada-moe` | [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) | Apache-2.0 |
| `llada2.1-mini` | [inclusionAI/LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini) | Apache-2.0 |
| `dream-7b` | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | Apache-2.0 |
| timing only | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 |

Transformers downloads the weights on first use. A local copy can be passed in place
of a name, for example `--model /path/to/LLaDA-8B-Instruct`.

## Data

The files `data/*.jsonl` hold the problems that every method decodes. They contain
all 164 HumanEval problems, the 500 MBPP test problems and all 500 MATH500 problems.
They also contain 500 GSM8K test problems and the first 500 of the 541 IFEval prompts.
The GSM8K problems are the first 500 of the test split after one shuffle with seed 113.

Running `python data/prepare_data.py --check` rebuilds the files from Hugging Face and
compares them. The file `data/README.md` lists their sources, licenses and fields.

## Quick start

The commands below decode the first 20 GSM8K problems with PACT on LLaDA-8B and grade
the answers.

```bash
python scripts/decode.py --model llada-8b --benchmark gsm8k \
    --rule pact --param tau=0.5 --param gap=8 --num_problems 20 \
    --out results/demo/llada-8b/gsm8k/pact
python scripts/evaluate.py results/demo/llada-8b/gsm8k/pact
```

The same decoding from Python is shown below.

```python
from pact.decoding import decode_question
from pact.models import load_model
from pact.rules import make_rule

lm = load_model("llada-8b")
rule = make_rule("pact", tau=0.5, gap=8)
answers, result = decode_question(lm, "What is 17 * 23?", rule,
                                  num_samples=8, temperature=0.6, seed=113)
print(answers[0], result.forward_passes)
```

The table lists each rule with its name for `--rule` and its parameters for `--param`.
The last column gives the setting of Table 1.

| Rule | Name | Parameters | Setting in Table 1 |
|---|---|---|---|
| PACT | `pact` | `tau`, `gap` | `tau` 0.5, `gap` 8, and `gap` 1 on LLaDA2.1-mini |
| Fast-dLLM | `fast_dllm` | `threshold` | 0.9 |
| Fast-dLLM++ | `fast_dllm_pp` | `delta` | 0 |
| EB-Sampler | `eb_sampler` | `gamma` | 0.5 |
| KLASS | `klass` | `confidence`, `kl_threshold`, `history_length` | 0.9, 0.001, 2 |
| TACG | `tacg` | `threshold` and the defaults of its paper | 0.9 |
| Fixed k | `fixed_k` | `k` | 4 |
| One token per step | `one_token_per_step` | none | none |

A gap of 1 turns spacing off.

## Reproducing the experiments

Each config lists models, benchmarks and rule settings. The script
`scripts/run_config.py` runs a config, `scripts/evaluate.py` grades the answers, and
`scripts/summarize.py` prints a table. The option `--dry_run` lists the runs without
decoding. The options `--models`, `--benchmarks` and `--runs` select a subset, for
example to spread the work over several GPUs.

| Paper | Command | Runs |
|---|---|---|
| Table 1 | `bash scripts/reproduce_table1.sh` | 160 |
| Figure 3 | `python scripts/run_config.py configs/sweeps.yaml` | 816 |
| Figure 4 | `python scripts/run_config.py configs/timing.yaml` | 18 |
| Appendix D.2 | `python scripts/run_config.py configs/blocks32.yaml` | 40 |
| Appendix D.4 | `python scripts/run_config.py configs/ar_comparison.yaml` | 16 |

Figure 4 and Appendix D.4 compare decoding times, so the runs of one model should use
one GPU. For Appendix D.4, also run `python scripts/time_ar.py --out
results/ar_comparison/qwen3-8b.json` on that GPU. The seconds are stored with the
answers and reported in `summary.json`.

Each run directory holds `run.json` with the settings and `gen.jsonl` with the answers.
The answers of each problem come with their forward passes and seconds. Grading adds
`scores.jsonl` and `summary.json`.

### Decoding protocol

Each answer has 256 positions, decoded as a single block. The commit rule alone
therefore decides the order in which positions are filled. Appendix D.2 uses blocks
of 32 positions instead. Tokens are drawn at temperature 0.6 by Gumbel-max sampling,
and every rule writes the drawn tokens. No key-value cache is used.

Eight answers per problem are decoded together as a batch. The answers to the i-th
problem of a run are drawn with seed 113 + i. MBPP runs as two shards of 250 problems,
as in the paper, and the other benchmarks run as one shard.

Each step costs one forward pass, and a step runs while any of the eight answers
still has a masked position. The cost of a problem is its number of forward passes
until all eight answers are filled. This count includes the padding after the end of
an answer. Speed is 256 divided by this count. It is the number of tokens written per
forward pass and equals the speed-up over one token per step.

The pass@1 and pass@8 values are unbiased estimates over the eight answers of each
problem. A code answer is correct when its first code block passes the tests of the
benchmark. A math answer is correct when its final answer is equivalent to the
reference. An IFEval answer is correct when it follows every instruction under the
official strict check.

## Paper sections and code

| Paper | Code |
|---|---|
| Section 3.1, drawn tokens and the two reliabilities | `src/pact/rules/base.py`, `src/pact/decoding.py` |
| Section 3.2 and Algorithm 1, PACT | `src/pact/rules/pact.py` |
| Proposition 3 | `tests/test_pact.py` |
| Section 4.1 and Appendix C.1, models | `src/pact/models.py` |
| Appendix C.2, benchmarks, prompts and grading | `src/pact/eval/`, `data/` |
| Appendix C.3 and Table 3, published rules and their settings | `src/pact/rules/`, `configs/table1.yaml`, `configs/sweeps.yaml` |
| Appendix C.4, cost and timing | `src/pact/decoding.py`, `src/pact/eval/metrics.py`, `configs/timing.yaml` |
| Table 1 | `configs/table1.yaml`, `scripts/reproduce_table1.sh` |
| Figure 3 | `configs/sweeps.yaml` |
| Figure 4 and Appendix D.4 | `configs/timing.yaml`, `configs/ar_comparison.yaml`, `scripts/time_ar.py` |
| Appendix D.2 | `configs/blocks32.yaml` |

## Compared rules

The five published rules run in the same decoder as PACT. Each one is re-implemented
from the equations and algorithms of its paper. Every rule writes the tokens drawn at
the sampling temperature, so the rules differ only in which positions they commit.

* [Fast-dLLM](https://openreview.net/forum?id=3Z3Is6hnOT) scores a position by the
  probability of its most likely token, as in Algorithm 1 of its paper.
* [Fast-dLLM++](https://arxiv.org/abs/2606.02955) commits the longest prefix allowed
  by a Fréchet bound.
* [EB-Sampler](https://proceedings.neurips.cc/paper_files/paper/2025/file/510e22f4a2da5212a64f1591736e2eaf-Paper-Conference.pdf)
  commits the longest prefix whose entropy bound stays below a threshold.
* [KLASS](https://openreview.net/forum?id=gOG9Zoyn4R) commits positions whose
  distribution has stopped changing.
* [TACG](https://arxiv.org/abs/2607.03236) gates each position by the history of its
  most likely token.

Fixed k commits the k positions whose drawn tokens are most probable. One token per
step is the reference sampler of LLaDA, with one step per position.

## License

This repository is released under the MIT License. The file `THIRD_PARTY.md` lists
the files that come from other projects and their licenses. These are the math
grader, the code execution sandbox and the official IFEval checker.
