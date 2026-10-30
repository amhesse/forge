"""The supervisor's commit: one code path for every agent (design step 1b).

The gate judges `git diff base HEAD`, so an attempt's work has to be committed
before the checks run. Agents used to do that themselves, three different
ways; now none of them commits and forge does it here, after every attempt
that says it produced something.
"""
import subprocess
from pathlib import Path

from .. import worktree as wt


def _git(worktree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(worktree), capture_output=True, text=True)


def _materialized(worktree: Path) -> list[str]:
    """The files and links worktree.materialize() put here that git doesn't
    track: config forge copied in for the editor, and links to the project's
    .venv / node_modules. They aren't the agent's work, so they never get
    staged. A tracked file of the same name isn't excluded, so an item that
    edits a project's tracked .aiderloop.toml still commits."""
    names = [*wt.MATERIALIZE_FILES, *wt.MATERIALIZE_LINKS]
    tracked = set(_git(worktree, "ls-files", "--", *names).stdout.split())
    return [n for n in names if (worktree / n).exists() and n not in tracked]


def commit_attempt(worktree: Path, message: str) -> tuple[bool, str]:
    """Stages everything the attempt changed (except materialized files) and
    commits it. Returns (committed, error): committed is False with no error
    when there was nothing to commit, and False with an error when git
    failed."""
    excludes = [f":(exclude){n}" for n in _materialized(worktree)]
    add = _git(worktree, "add", "-A", "--", ".", *excludes)
    if add.returncode != 0:
        return False, f"git add failed: {add.stderr.strip()}"
    if _git(worktree, "diff", "--cached", "--quiet").returncode == 0:
        return False, ""
    commit = _git(worktree, "commit", "-q", "-m", message)
    if commit.returncode != 0:
        return False, f"git commit failed: {commit.stderr.strip()}"
    return True, ""
