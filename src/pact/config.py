"""Experiment configs (``configs/*.yaml``) and their expansion into single runs.

A config lists models, benchmarks and runs.  A run names a rule and its
parameters; ``per_model`` overrides parameters on some models (PACT runs
without spacing on LLaDA2.1-mini).  A ``sweeps`` section expands into one run
per value of a parameter.  See configs/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DECODING_DEFAULTS = {"temperature": 0.6, "num_samples": 8, "gen_length": 256,
                     "block_length": 256, "seed": 113}


@dataclass
class Run:
    model: str
    benchmark: str
    name: str
    rule: str
    params: dict = field(default_factory=dict)
    decoding: dict = field(default_factory=dict)
    num_shards: int = 1
    num_problems: int = 0


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("name", Path(path).stem)
    return config


def _value_name(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def config_runs(config: dict) -> list[dict]:
    """The run entries of a config, with sweeps expanded."""
    runs = [dict(run) for run in config.get("runs", [])]
    for rule, sweep in (config.get("sweeps") or {}).items():
        for value in sweep["values"]:
            params = dict(sweep.get("params", {}))
            params[sweep["param"]] = value
            runs.append({"name": f"{sweep.get('name', rule)}-{sweep['param']}{_value_name(value)}",
                         "rule": sweep.get("rule", rule), "params": params,
                         "per_model": sweep.get("per_model", {})})
    for run in runs:
        run.setdefault("name", run["rule"])
        run.setdefault("params", {})
    return runs


def expand(config: dict, *, models=None, benchmarks=None, names=None) -> list[Run]:
    """All (model, benchmark, run) combinations of a config, optionally filtered."""
    decoding = {**DECODING_DEFAULTS, **(config.get("decoding") or {})}
    shards = config.get("shards") or {}
    out = []
    for model in config["models"]:
        if models and model not in models:
            continue
        for benchmark in config["benchmarks"]:
            if benchmarks and benchmark not in benchmarks:
                continue
            for run in config_runs(config):
                if names and run["name"] not in names:
                    continue
                per_model = (run.get("per_model") or {}).get(model, {})
                params = {**run["params"], **per_model}
                out.append(Run(model=model, benchmark=benchmark, name=run["name"],
                               rule=run["rule"], params=params,
                               decoding={**decoding, **(run.get("decoding") or {})},
                               num_shards=int(shards.get(benchmark, 1)),
                               num_problems=int(config.get("num_problems", 0))))
    return out
