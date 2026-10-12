# Third-party code

The code of this repository is released under the MIT License. The files below come
from other projects and keep their licenses.

| Files | Origin | License |
|---|---|---|
| `third_party/instruction_following_eval/` | The official IFEval checker of Google Research, from [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) at commit `807d4a2f41202059bac2446259d135a89ed3630a`. The files are unchanged. The sample responses and the unit tests of the original directory are left out. | Apache-2.0, see `third_party/instruction_following_eval/LICENSE` |
| `src/pact/eval/math_grader.py` and `src/pact/eval/math_parser.py` | Copied without changes from `utils/grader.py` and `utils/parser.py` of [JustGRPO](https://github.com/LeapLabTHU/JustGRPO). The docstring of the grader lists the projects these files were adapted from. | MIT, see `third_party/licenses/JustGRPO-MIT.txt` |
| `src/pact/eval/code_exec.py` | The sandbox and the assembly of the test program are adapted from `utils/code_exec.py` of JustGRPO. That file follows the execution harness of [OpenAI's HumanEval](https://github.com/openai/human-eval), which is also under MIT. | MIT, see `third_party/licenses/JustGRPO-MIT.txt` |
| `humaneval_question` and `mbpp_question` in `src/pact/eval/benchmarks.py` | The wording of the code prompts follows the evaluation script of JustGRPO. | MIT, see `third_party/licenses/JustGRPO-MIT.txt` |

Other parts follow the descriptions of earlier work without copying its code.

* `add_gumbel_noise` in `src/pact/decoding.py` and the rule in
  `src/pact/rules/one_token.py` follow the reference sampler of
  [LLaDA](https://github.com/ML-GSAI/LLaDA), which is under MIT. They use Gumbel-max
  sampling and low-confidence remasking.
* `llada2_block_inputs` in `src/pact/models.py` builds the block-causal attention mask
  of the reference code released with LLaDA2.1-mini.
* The five published commit rules in `src/pact/rules/` are re-implemented from the
  equations and algorithms of their papers. The main README links each paper.

The benchmark data under `data/` keep the licenses listed in `data/README.md`.
