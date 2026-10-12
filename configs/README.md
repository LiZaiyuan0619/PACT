# Configs

| Config | Paper | Models | Benchmarks | Settings |
|---|---|---|---|---|
| `table1.yaml` | Table 1, Appendix Table 3 | all four | all five | every rule at one setting |
| `sweeps.yaml` | Figure 3 | all four | code and math | each rule over its range in Appendix Table 3 |
| `blocks32.yaml` | Appendix D.2 | LLaDA-8B, Dream-7B | code and math | 32-token blocks; PACT with g = 4 |
| `timing.yaml` | Figure 4 | LLaDA-8B, LLaDA-MoE, Dream-7B | 50 HumanEval problems | one GPU for all settings of a model |
| `ar_comparison.yaml` | Appendix D.4 | LLaDA-8B, Dream-7B | 50 HumanEval and 50 GSM8K problems | greedy, one answer, 32-token blocks; Qwen3-8B via `scripts/time_ar.py` |

Model names: `llada-8b` (GSAI-ML/LLaDA-8B-Instruct), `llada-moe`
(inclusionAI/LLaDA-MoE-7B-A1B-Instruct), `llada2.1-mini` (inclusionAI/LLaDA2.1-mini),
`dream-7b` (Dream-org/Dream-v0-Instruct-7B).

## Format

```yaml
name: table1                  # results go to results/<name>/<model>/<benchmark>/<run>/
decoding:                     # defaults: temperature 0.6, 8 answers, 256 positions,
  block_length: 256           #   one block of 256, seed 113
models: [llada-8b, ...]
benchmarks: [humaneval, ...]
shards: {mbpp: 2}             # benchmarks decoded in several shards (seeds restart per shard)
num_problems: 0               # 0 = all problems
runs:                         # one entry per rule setting
  - name: pact
    rule: pact
    params: {tau: 0.5, gap: 8}
    per_model:                # parameters that differ on some models
      llada2.1-mini: {gap: 1}
sweeps:                       # one run per value; run names are <rule>-<param><value>
  pact: {param: tau, values: [0.05, 0.3], params: {gap: 8}}
```

Rule names and parameters are listed in `src/pact/rules/__init__.py`.
