"""Scoring code answers: run the first code block against the benchmark tests.

The sandbox follows the execution harness of OpenAI's HumanEval release, as
adapted in the evaluation utilities of JustGRPO (both MIT License, see
THIRD_PARTY.md).  It runs each program in a fresh process with a time limit,
disables destructive system calls, and swallows all I/O.  It is not a security
boundary; run untrusted code in a container.
"""

from __future__ import annotations

import contextlib
import faulthandler
import io
import multiprocessing
import os
import re
import signal
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def extract_code(answer: str) -> str:
    """The first ```python block of an answer, or the whole answer if there is none."""
    match = CODE_BLOCK_RE.search(answer)
    return match.group(1).strip() if match else answer


def build_test_program(problem: dict, answer: str, benchmark: str) -> str:
    """The program that passes exactly when the answer is correct."""
    code = extract_code(answer)
    if benchmark == "mbpp":
        return code + "\n" + "\n".join(problem["test_list"])
    return (problem["prompt"] + code + "\n" + problem["test"] + "\n"
            + f"check({problem['entry_point']})")


class _Timeout(Exception):
    pass


class _WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class _StdinRedirect(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def _time_limit(seconds: float):
    def handler(signum, frame):
        raise _Timeout()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), \
            _StdinRedirect(stream):
        yield


def _reliability_guard():
    faulthandler.disable()
    import builtins
    import shutil
    import subprocess
    import sys
    builtins.exit = builtins.quit = builtins.help = None
    os.environ["OMP_NUM_THREADS"] = "1"

    def _disabled(*args, **kwargs):
        raise RuntimeError("Restricted")
    for attr in ("system", "kill", "setuid", "fork"):
        if hasattr(os, attr):
            setattr(os, attr, _disabled)
    for attr in ("putenv", "remove", "removedirs", "rmdir", "fchdir", "forkpty",
                 "killpg", "rename", "renames", "truncate", "replace", "unlink",
                 "fchmod", "fchown", "chmod", "chown", "chroot",
                 "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
        if hasattr(os, attr):
            setattr(os, attr, None)
    shutil.rmtree = shutil.move = shutil.chown = None
    subprocess.Popen = None
    for module in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[module] = None


def check_program(program: str, timeout: float = 3.0) -> bool:
    """Run ``program`` in a separate process; True if it finishes without an exception."""
    def unsafe_execute():
        import shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            chdir, rmdir, rmtree = os.chdir, os.rmdir, shutil.rmtree
            try:
                _reliability_guard()
                with _swallow_io():
                    with _time_limit(timeout):
                        exec(program, {})
                result.append("passed")
            except _Timeout:
                result.append("timed out")
            except AssertionError:
                result.append("failed: AssertionError")
            except BaseException as exc:  # noqa: BLE001
                result.append(f"failed: {exc}")
            finally:
                os.chdir, os.rmdir, shutil.rmtree = chdir, rmdir, rmtree

    manager = multiprocessing.Manager()
    result = manager.list()
    process = multiprocessing.Process(target=unsafe_execute)
    process.start()
    process.join(timeout=timeout + 1)
    if process.is_alive():
        process.kill()
    return bool(result) and result[0] == "passed"


def score_code(benchmark: str, problems: dict, answers: dict, *, n_workers: int = 8,
               timeout: float = 3.0) -> dict:
    """``{problem_id: [passed, ...]}`` in the order of the answers of each problem.

    ``problems`` maps a problem id to its record, ``answers`` maps it to a list
    of answer strings.
    """
    passed = {pid: [False] * len(texts) for pid, texts in answers.items()}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for pid, texts in answers.items():
            for slot, text in enumerate(texts):
                program = build_test_program(problems[pid], text, benchmark)
                futures[pool.submit(check_program, program, timeout)] = (pid, slot)
        # Results arrive in completion order; put each one back in its slot.
        for future in as_completed(futures):
            pid, slot = futures[future]
            passed[pid][slot] = bool(future.result())
    return passed
