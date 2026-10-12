# Third-party code

The code of this repository is released under the MIT License (`LICENSE`), except for
the files below, which come from other projects and keep their licenses.

| Files | Origin | License |
|---|---|---|
| `third_party/instruction_following_eval/` | The official IFEval checker of Google Research, [`google-research/instruction_following_eval`](https://github.com/google-research/google-research/tree/master/instruction_following_eval), commit `807d4a2f41202059bac2446259d135a89ed3630a`.  The files are unchanged; the sample responses and the unit tests of the original directory are left out. | Apache-2.0, `third_party/instruction_following_eval/LICENSE` |
| `src/pact/eval/math_grader.py`, `src/pact/eval/math_parser.py` | Copied without changes from the evaluation utilities (`utils/grader.py`, `utils/parser.py`) of [JustGRPO](https://github.com/LeapLabTHU/JustGRPO).  The grader's docstring lists the projects these utilities were adapted from (the MATH release of Hendrycks et al., PRM800K, ToRA, DeepSeek-Math, CRITIC). | MIT, `third_party/licenses/JustGRPO-MIT.txt` |
| `src/pact/eval/code_exec.py` | The sandbox (`check_program` and its helpers) and the assembly of the test program are adapted from `utils/code_exec.py` of JustGRPO, which follows the execution harness of [OpenAI's HumanEval](https://github.com/openai/human-eval) (MIT). | MIT, `third_party/licenses/JustGRPO-MIT.txt` |
| `src/pact/eval/benchmarks.py` (`humaneval_question`, `mbpp_question`) | The wording of the code prompts follows the evaluation script of JustGRPO. | MIT, `third_party/licenses/JustGRPO-MIT.txt` |

Further acknowledgements, for code written from a description rather than copied:

* `src/pact/decoding.py` (`add_gumbel_noise`) and `src/pact/rules/one_token.py` follow
  the reference sampler released with [LLaDA](https://github.com/ML-GSAI/LLaDA) (MIT):
  Gumbel-max sampling and low-confidence remasking.
* `src/pact/models.py` (`llada2_block_inputs`) builds the block-causal attention mask
  used by the reference code released with the LLaDA2.1-mini checkpoint.
* The five published commit rules in `src/pact/rules/` (Fast-dLLM, Fast-dLLM++,
  EB-Sampler, KLASS, TACG) are re-implemented following the equations and algorithms
  of their papers; the README links each paper.

The benchmark data under `data/` keep the licenses listed in `data/README.md`.
