# Configs

Each config describes one experiment of the paper.

| Config | Paper | Models | Benchmarks | Setting |
|---|---|---|---|---|
| `table1.yaml` | Table 1 and Appendix Table 3 | all four | all five | every rule at one setting |
| `sweeps.yaml` | Figure 3 | all four | code and math | each rule over its range in Appendix Table 3 |
| `blocks32.yaml` | Appendix D.2 | LLaDA-8B and Dream-7B | code and math | blocks of 32 tokens, PACT with g = 4 |
| `timing.yaml` | Figure 4 | LLaDA-8B, LLaDA-MoE and Dream-7B | 50 HumanEval problems | every setting of a model on one GPU |
| `ar_comparison.yaml` | Appendix D.4 | LLaDA-8B and Dream-7B | 50 HumanEval and 50 GSM8K problems | one greedy answer per problem in blocks of 32 tokens |

The autoregressive model of Appendix D.4 is timed by `scripts/time_ar.py`.

| Name in configs | Checkpoint on Hugging Face |
|---|---|
| `llada-8b` | GSAI-ML/LLaDA-8B-Instruct |
| `llada-moe` | inclusionAI/LLaDA-MoE-7B-A1B-Instruct |
| `llada2.1-mini` | inclusionAI/LLaDA2.1-mini |
| `dream-7b` | Dream-org/Dream-v0-Instruct-7B |

## Format

The example below uses every field of a config.

```yaml
name: table1
decoding:
  block_length: 256
models: [llada-8b, llada-moe]
benchmarks: [humaneval, mbpp]
shards: {mbpp: 2}
num_problems: 0
runs:
  - name: pact
    rule: pact
    params: {tau: 0.5, gap: 8}
    per_model:
      llada2.1-mini: {gap: 1}
sweeps:
  pact: {param: tau, values: [0.05, 0.3], params: {gap: 8}}
```

* `name` sets the output directory. Results go to `results/<name>/<model>/<benchmark>/<run>/`.
* `decoding` overrides the defaults. These are temperature 0.6, eight answers per
  problem, 256 positions in one block, and seed 113.
* `shards` decodes a benchmark in several shards. The seeds restart in each shard, and
  the paper ran MBPP as two shards.
* `num_problems` keeps the first problems of each benchmark. The value 0 keeps all.
* `runs` lists the rule settings, one entry each. The field `per_model` sets the
  parameters that differ on some models.
* `sweeps` adds one run for each value of a parameter. These runs are named
  `<rule>-<param><value>`.

The names and parameters of the rules are listed in `src/pact/rules/__init__.py`.
