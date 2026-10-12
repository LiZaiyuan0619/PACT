# PACT: Certifying What Is Written in Parallel Decoding of Diffusion Language Models

This repository contains the code of the paper (under anonymous review): PACT, the
commit rules it is compared with, the decoding loop they share, the four models,
the five benchmarks with their grading, and the configs of the experiments.

Parallel decoding speeds up masked diffusion language models by writing several
tokens per forward pass.  A *commit rule* decides which of the drawn tokens a step
writes.  Published rules decide from the predicted distributions alone; under
sampling the written tokens are drawn, and these tests do not see them.  PACT
draws the tokens first, certifies the set it will write, and spaces the positions
of each step apart.

## One step of PACT

For each answer, given the masked positions of the current block, their model
distributions `p_i` and the sampling temperature `T` (Algorithm 1 in Appendix B.1):

1. draw `x_i ~ q_i ∝ p_i^(1/T)` and score each position by its **written-word score**
   `s_i = log p_i(x_i)` (the probability *before* temperature);
2. sort the positions by score, highest first: the order `π`;
3. **set-level cut**: `k*` is the longest prefix of `π` with `Σ s ≥ log τ` (or 1 if none passes);
4. **spacing**: scan `π` and keep positions at least `g` apart until `k*` are kept,
   lowering the gap one step at a time if needed;
5. list the kept positions in the order of `π`, cut again at `log τ`, and write the
   drawn tokens there.

The implementation is `src/pact/rules/pact.py` (about 65 lines without comments).

## Repository layout

```
PACT/
├── src/pact/
│   ├── rules/            PACT and the compared commit rules, one file each
│   ├── decoding.py       the decoding loop shared by every rule
│   ├── models.py         LLaDA-8B, LLaDA-MoE, LLaDA2.1-mini, Dream-7B
│   ├── runner.py         one rule over the problems of one benchmark
│   ├── config.py         experiment configs
│   └── eval/             benchmarks, grading (math, code, IFEval), pass@k and speed
├── configs/              Table 1, Figure 3, Figure 4, Appendix D.2 and D.4
├── data/                 the problems of the five benchmarks and the script that builds them
├── scripts/              decode, run a config, grade, build the table, time Qwen3-8B
├── tests/                CPU tests (rules, Proposition 3, decoding loop, grading, pipeline)
└── third_party/          the official IFEval checker and third-party licenses
```

## Installation

```bash
conda create -n pact python=3.11 -y && conda activate pact
pip install -r requirements.txt          # versions used in the paper
pip install -e .
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"   # IFEval checker
pytest -q                                # CPU only, about 10 seconds
```

The experiments ran on single NVIDIA H200 GPUs with Python 3.11, PyTorch 2.9.1 and
Transformers 4.57.1.  Grading needs only CPUs.

## Models

| Name in configs | Checkpoint on Hugging Face | License |
|---|---|---|
| `llada-8b` | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | MIT |
| `llada-moe` | [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) | Apache-2.0 |
| `llada2.1-mini` | [inclusionAI/LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini) | Apache-2.0 |
| `dream-7b` | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | Apache-2.0 |
| (timing only) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 |

Weights are downloaded by Transformers on first use (`trust_remote_code=True`).  A
local copy can be passed instead of a name, e.g. `--model /path/to/LLaDA-8B-Instruct`.

## Data

`data/*.jsonl` hold the problems every method decodes: all 164 HumanEval problems,
the 500 MBPP test problems, 500 GSM8K test problems (the test split shuffled once
with seed 113, first 500), all 500 MATH500 problems and the first 500 of the 541
IFEval prompts.  `python data/prepare_data.py --check` rebuilds them from Hugging
Face and compares; `data/README.md` lists sources, licenses and fields.

## Quick start

Decode the first 20 GSM8K problems with PACT on LLaDA-8B and grade them:

```bash
python scripts/decode.py --model llada-8b --benchmark gsm8k \
    --rule pact --param tau=0.5 --param gap=8 --num_problems 20 \
    --out results/demo/llada-8b/gsm8k/pact
python scripts/evaluate.py results/demo/llada-8b/gsm8k/pact
```

From Python:

```python
from pact.decoding import decode_question
from pact.models import load_model
from pact.rules import make_rule

lm = load_model("llada-8b")
answers, result = decode_question(lm, "What is 17 * 23?", make_rule("pact", tau=0.5, gap=8),
                                  num_samples=8, temperature=0.6, seed=113)
print(answers[0], result.forward_passes, 256 / result.forward_passes)
```

Rules and their parameters (`--rule`, `--param`):

| Rule | Name | Parameters (setting of Table 1) |
|---|---|---|
| PACT | `pact` | `tau` (0.5), `gap` (8; 1 = no spacing, used on LLaDA2.1-mini) |
| Fast-dLLM | `fast_dllm` | `threshold` (0.9) |
| Fast-dLLM++ | `fast_dllm_pp` | `delta` (0) |
| EB-Sampler | `eb_sampler` | `gamma` (0.5) |
| KLASS | `klass` | `confidence` (0.9), `kl_threshold` (0.001), `history_length` (2) |
| TACG | `tacg` | `threshold` (0.9) and the defaults of the paper |
| fixed k | `fixed_k` | `k` (4) |
| one token per step | `one_token_per_step` | -- |

## Reproducing the experiments

Each config lists models, benchmarks and rule settings; `scripts/run_config.py`
runs them (`--dry_run` lists the runs; `--models`, `--benchmarks` and `--runs`
select a subset, e.g. to spread the work over GPUs), `scripts/evaluate.py` grades
them and `scripts/summarize.py` prints the table.

| Paper | Command |
|---|---|
| Table 1 | `bash scripts/reproduce_table1.sh` (160 runs; `--models llada-8b` for one model) |
| Figure 3 | `python scripts/run_config.py configs/sweeps.yaml` |
| Figure 4 | `python scripts/run_config.py configs/timing.yaml` on one GPU; time is in `summary.json` |
| Appendix D.2 | `python scripts/run_config.py configs/blocks32.yaml` |
| Appendix D.4 | `python scripts/run_config.py configs/ar_comparison.yaml` and `python scripts/time_ar.py --out results/ar_comparison/qwen3-8b.json` on the same GPU |

Each run directory holds `run.json` (settings), `gen.jsonl` (answers, forward passes
and seconds per problem), and after grading `scores.jsonl` and `summary.json`.

### Protocol

* **Decoding.** Each answer has 256 positions, decoded as a single block, so the
  commit rule alone decides the order (Appendix D.2 uses blocks of 32).  Tokens are
  drawn at temperature 0.6 by Gumbel-max sampling; every rule writes the drawn tokens.
  No key-value cache is used.
* **Samples and seeds.** Eight answers per problem are decoded together as a batch.
  The answers to the i-th problem of a run are drawn with seed 113 + i; MBPP runs as
  two shards of 250 problems, as in the paper, and the other benchmarks as one.
* **Cost and speed.** A step runs while any of the eight answers still has a masked
  position, and each step costs one forward pass.  The cost of a problem is its number
  of forward passes until all eight answers are filled, padding after the end of an
  answer included.  Speed is `256 / forward passes`, the tokens written per forward
  pass, which equals the speed-up over one token per step.
* **Accuracy.** pass@1 and pass@8 are the unbiased estimates over the eight answers.
  Code answers pass when the first code block passes the benchmark tests; math answers
  when the final answer is equivalent to the reference; IFEval answers when they follow
  every instruction under the official strict check.

## Where each part of the paper lives

| Paper | Code |
|---|---|
| Section 3.1: commit rules, drawn tokens, actual and reported reliability | `src/pact/decoding.py` (`StepView.written_prob`, `StepView.top1_prob`) |
| Section 3.2, Algorithm 1: PACT | `src/pact/rules/pact.py` (`passing_prefix`, `spaced_scan`, `PACT.select_row`) |
| Proposition 3: what PACT commits | `tests/test_pact.py` |
| Section 4.1, Appendix C.1: models | `src/pact/models.py` |
| Appendix C.2: benchmarks, prompts, grading | `src/pact/eval/benchmarks.py`, `src/pact/eval/scoring.py`, `data/` |
| Appendix C.3, Table 3: the published rules and their settings | `src/pact/rules/`, `configs/table1.yaml`, `configs/sweeps.yaml` |
| Appendix C.4: cost and timing | `src/pact/decoding.py`, `src/pact/eval/metrics.py`, `configs/timing.yaml` |
| Table 1 | `configs/table1.yaml`, `scripts/reproduce_table1.sh` |
| Figure 3 | `configs/sweeps.yaml` |
| Figure 4, Appendix D.4 | `configs/timing.yaml`, `configs/ar_comparison.yaml`, `scripts/time_ar.py` |
| Appendix D.2: 32-token blocks | `configs/blocks32.yaml` |

## Compared rules

The five published rules are implemented in the same decoder as PACT, following the
equations and algorithms of their papers.  Every rule writes the tokens drawn at the
sampling temperature, so the rules differ only in which positions they commit.

* Fast-dLLM: [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://openreview.net/forum?id=3Z3Is6hnOT).
  A position is scored by the probability of its most likely token, as in Algorithm 1 of the paper.
* Fast-dLLM++: [Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference](https://arxiv.org/abs/2606.02955).
* EB-Sampler: [Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking](https://proceedings.neurips.cc/paper_files/paper/2025/file/510e22f4a2da5212a64f1591736e2eaf-Paper-Conference.pdf).
* KLASS: [KLASS: KL-Guided Fast Inference in Masked Diffusion Models](https://openreview.net/forum?id=gOG9Zoyn4R).
* TACG: [TACG: Trajectory-Aware Commit Gating for Diffusion Language Model Decoding](https://arxiv.org/abs/2607.03236).

Fixed k commits the k positions whose drawn tokens are most probable, and one token
per step is the reference sampler of LLaDA with one step per position.

## Third-party code and license

The code of this repository is released under the MIT License (`LICENSE`).
`THIRD_PARTY.md` lists the files that come from other projects and their licenses:
the math grader, the code-execution sandbox, and the official IFEval checker.
