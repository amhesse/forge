"""forge's own regression suite: a fixed set of scenarios exercising the
real aider_loop pipeline (worktrees, merge, park, checks) against stub
editors, so a change to the loop's own logic can be checked against
known-good outcomes in seconds - no real model, no GPU, deterministic.

Each Scenario writes a throwaway git repo with a checklist and a stub
"aider" script, runs the real installed `forge run` command against it
as a subprocess (never in-process - aider_loop.py keeps module-level
CONFIG state that isn't safe to reuse across repeated calls in one
process), and asserts the real, observable outcome.
"""

import dataclasses
import subprocess
import tempfile
from pathlib import Path


@dataclasses.dataclass
class Scenario:
    name: str
    description: str
    todo: str
    aiderloop_toml: str = '[validate]\nchecks = ["python"]\n'
    extra_args: list = dataclasses.field(default_factory=list)
    check: object = None  # callable(project_dir: Path, log: str) -> (bool, str)


def make_stub_aider(bin_dir: Path) -> None:
    """Writes an executable stub `aider` into bin_dir, driven by markers
    in the --message text rather than a real model: NOCHANGE exits
    cleanly with nothing touched, CRASH exits non-zero, GARBAGE writes a
    file named after a fragment of reasoning text (the real failure mode
    the loop's garbage-filename check exists for), and anything else
    writes to the first backtick-quoted path named in the message.
    """
    raise NotImplementedError


def run_scenario(scenario: "Scenario") -> tuple[bool, str]:
    """Sets up a throwaway repo for `scenario`, runs the real `forge run`
    command against it as a subprocess, and returns scenario.check()'s
    result. Always uses a COPY of the real environment with `bin_dir`
    prepended to PATH, never a replacement - replacing it strips HOME
    along with everything else, and a stub's `git commit` then fails
    with "Author identity unknown" because it can't find ~/.gitconfig,
    which looks exactly like aider itself failing.
    """
    raise NotImplementedError


SCENARIOS: list = []


def main(argv: list[str] | None = None) -> int:
    """Runs every scenario in SCENARIOS, prints PASS/FAIL for each, and
    returns 0 if all passed, 1 otherwise."""
    raise NotImplementedError
