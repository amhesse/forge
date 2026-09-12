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
import sys
import tempfile
from pathlib import Path

STUB_AIDER = '''#!/usr/bin/env python3
import re, subprocess, sys, pathlib
msg = sys.argv[sys.argv.index("--message") + 1]
if "NOCHANGE" in msg:
    sys.exit(0)
if "CRASH" in msg:
    sys.exit(3)
if "GARBAGE" in msg:
    pathlib.Path("File Listing: junk.py").write_text("bad")
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "stub garbage"], check=True)
    sys.exit(0)
paths = re.findall(r"`([\\w./-]+\\.[A-Za-z]{1,7})`", msg)
if not paths:
    sys.exit(0)
p = pathlib.Path(paths[0])
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("def broken(:\\n" if "BADSYNTAX" in msg else "value = 1\\n")
subprocess.run(["git", "add", "-A"], check=True)
subprocess.run(["git", "commit", "-q", "-m", "stub"], check=True)
'''


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
    stub_path = bin_dir / "aider"
    stub_path.write_text(STUB_AIDER)
    stub_path.chmod(0o755)


def _run(project_dir: Path, todo_name: str, bin_dir: Path, extra_args: list) -> str:
    import os
    # A copy of the REAL environment with bin_dir prepended - not a
    # replacement. Replacing it strips HOME along with everything else,
    # so git can't find ~/.gitconfig and every commit the stub makes
    # fails with "Author identity unknown" - which looks exactly like
    # aider itself failing (git exits non-zero), and produced a false
    # [!] on a scenario the loop actually handles correctly. Found by
    # running this against the real command before trusting a single
    # assertion in it.
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["forge", "run", "--project-dir", str(project_dir), "--todo-file", todo_name,
         "--sleep-between", "0", "--max-validation-retries", "1", *extra_args],
        cwd=str(project_dir), capture_output=True, text=True, timeout=60, env=env,
    )
    return result.stdout + result.stderr


def run_scenario(scenario: "Scenario") -> tuple[bool, str]:
    """Sets up a throwaway repo for `scenario`, runs the real `forge run`
    command against it as a subprocess, and returns scenario.check()'s
    result."""
    with tempfile.TemporaryDirectory() as d:
        project_dir = Path(d) / "proj"
        bin_dir = Path(d) / "bin"
        project_dir.mkdir()
        bin_dir.mkdir()
        make_stub_aider(bin_dir)

        subprocess.run(["git", "init", "-q", "-b", "master", "."], cwd=project_dir, check=True)
        (project_dir / "TODO.md").write_text(scenario.todo)
        (project_dir / ".aiderloop.toml").write_text(scenario.aiderloop_toml)
        subprocess.run(["git", "add", "-A"], cwd=project_dir, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "-m", "base"], cwd=project_dir, check=True)

        log = _run(project_dir, "TODO.md", bin_dir, scenario.extra_args)
        try:
            return scenario.check(project_dir, log)
        except Exception as e:
            return False, f"check raised {type(e).__name__}: {e}"


# ---- checks, each verified against real behavior before being written here ----

def check_merges_clean(project_dir: Path, log: str) -> tuple[bool, str]:
    todo = (project_dir / "TODO.md").read_text()
    if "- [x]" not in todo:
        return False, f"expected item to merge (marked [x]); TODO.md:\n{todo}"
    return True, "merged as expected"


def check_negation_not_falsely_parked(project_dir: Path, log: str) -> tuple[bool, str]:
    todo = (project_dir / "TODO.md").read_text()
    if "- [x]" not in todo:
        return False, f"'do not touch' item should merge, not park; TODO.md:\n{todo}"
    return True, "negation handled correctly"


def check_garbage_filename_blocked(project_dir: Path, log: str) -> tuple[bool, str]:
    if (project_dir / "File Listing: junk.py").exists():
        return False, "garbage filename reached the real checkout"
    todo = (project_dir / "TODO.md").read_text()
    if "- [!]" not in todo:
        return False, f"expected item blocked; TODO.md:\n{todo}"
    return True, "garbage filename correctly blocked, never reached checkout"


def check_crash_blocked_after_retries(project_dir: Path, log: str) -> tuple[bool, str]:
    todo = (project_dir / "TODO.md").read_text()
    if "- [!]" not in todo:
        return False, f"expected item blocked after aider crash; TODO.md:\n{todo}"
    return True, "crash correctly blocked"


def check_needs_review_when_untouched(project_dir: Path, log: str) -> tuple[bool, str]:
    todo = (project_dir / "TODO.md").read_text()
    if "- [?]" not in todo:
        return False, f"expected needs-review when named file untouched; TODO.md:\n{todo}"
    if (project_dir / "never.py").exists():
        return False, "file should never have been created"
    return True, "needs-review correctly applied, checkout untouched"


def check_byte_exact_mismatch_blocked(project_dir: Path, log: str) -> tuple[bool, str]:
    todo = (project_dir / "TODO.md").read_text()
    if "- [!]" not in todo:
        return False, f"expected byte-exact mismatch to block; TODO.md:\n{todo}"
    if (project_dir / "exact.py").exists():
        return False, "mismatched content should never reach the real checkout"
    return True, "byte-exact mismatch correctly blocked"


SCENARIOS = [
    Scenario(
        name="basic_merge",
        description="A plain single-file item merges cleanly.",
        todo="- [ ] In `good.py`, set a value.\n",
        check=check_merges_clean,
    ),
    Scenario(
        name="negation_at_end_of_item",
        description="'do not touch X' at the END of an item must not force X to be touched.",
        todo=("- [ ] In `tests/test_x.py`, add a test. Do not change any existing test in "
              "the file, and do not touch `x.py`.\n"),
        check=check_negation_not_falsely_parked,
    ),
    Scenario(
        name="garbage_filename_blocked",
        description="A malformed edit that invents a filename never reaches the real checkout.",
        todo="- [ ] GARBAGE: in `real.py`, do the thing.\n",
        check=check_garbage_filename_blocked,
    ),
    Scenario(
        name="aider_crash_blocked",
        description="aider exiting non-zero blocks the item after retries, doesn't hang.",
        todo="- [ ] CRASH: in `x.py`, do the thing.\n",
        check=check_crash_blocked_after_retries,
    ),
    Scenario(
        name="needs_review_when_untouched",
        description="An item whose named file was never touched needs review, not silent success.",
        todo="- [ ] NOCHANGE: in `never.py`, ignored by the stub.\n",
        check=check_needs_review_when_untouched,
    ),
    Scenario(
        name="byte_exact_mismatch_blocked",
        description="A byte-exact spec the stub doesn't satisfy is caught byte-for-byte, not by trust.",
        todo=('- [ ] Replace `exact.py` with exactly:\n```python\nEXPECTED = True\n```\n'),
        check=check_byte_exact_mismatch_blocked,
    ),
]


def main(argv: list[str] | None = None) -> int:
    """Runs every scenario in SCENARIOS, prints PASS/FAIL for each, and
    returns 0 if all passed, 1 otherwise."""
    results = []
    for s in SCENARIOS:
        ok, detail = run_scenario(s)
        results.append((s.name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {s.name}: {detail}")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
