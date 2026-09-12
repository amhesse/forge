"""
Per-item git worktree isolation for aider_loop.

Why this exists: the loop used to run every item directly in the project
checkout, which forced two behaviours that made unattended running
impossible. A failed item left its damage in the live tree until
git_revert_to() cleaned it up, and the run had to *stop* at the first
needs-review item rather than let the next one build on an unreviewed
change. Both are the same problem -- there was only one working tree, so
"keep going" and "don't build on this" were in conflict.

A worktree per item resolves it. Each item gets a throwaway checkout on
its own branch, branched from the project's current HEAD. Everything the
loop already does -- aider, the touched-files check, syntax, the project's
own test command -- happens in there against a tree nothing else shares. A
passing item fast-forwards the real branch. A failing one is simply never
merged: its branch is left behind for a human to look at, the project
checkout never saw it, and the next item starts clean from the same base.

Nothing is reverted, because nothing was ever applied.

## What a worktree does not get

`git worktree add` populates from the commit, so a fresh worktree contains
tracked files and nothing else. Measured across this machine's projects,
that is not enough to run an item:

  - blockroad     -- everything needed is tracked. Works with no help.
  - CableTrayFillCalc -- .aider.conf.yml, .aider.model.settings.yml,
                    .aiderignore and .aiderloop.toml are all *gitignored*.
                    A bare worktree would run aider with no model config
                    (silently falling back to whatever ~/.aider.conf.yml
                    says) and no .aiderignore (see the README on what a
                    missing .aiderignore does to the context window).
                    Its `[validate] commands` also runs `.venv/bin/pytest`
                    against a 317MB venv that isn't tracked either.
  - chore-tracker -- gitignored .aider.conf.yml, plus node_modules.

So the worktree is *materialised* after creation: small config files are
copied, and heavyweight dependency directories are symlinked rather than
copied (317MB per item is not a thing to duplicate). A symlinked venv was
verified to work from a different working directory -- the interpreter
resolves its own prefix through the link, so `.venv/bin/pytest` runs and
imports normally.
"""

import shutil
import subprocess
import time
from pathlib import Path

# Gitignored-but-required files copied into each worktree when the project
# checkout has them and the worktree doesn't. These are all small, and each
# one changes how aider behaves: silently running without them is worse
# than not running at all, because the run still produces commits.
MATERIALIZE_FILES = (
    ".aider.conf.yml",
    ".aider.model.settings.yml",
    ".aiderignore",
    ".aiderloop.toml",
)

# Dependency directories symlinked (never copied) when present. Copying
# CableTrayFillCalc's 317MB .venv per item would cost more time and disk
# than the item itself.
MATERIALIZE_LINKS = (
    ".venv",
    "venv",
    "node_modules",
)


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd),
                          capture_output=True, text=True)


def is_clean(project_dir: Path, exempt: tuple[str, ...] = ()) -> tuple[bool, str]:
    """True if the project checkout has no changes to tracked files.

    Checked before a run starts, not as politeness: a passing item is
    merged into this checkout with `git merge --ff-only`, which refuses to
    run over local modifications. Finding that out after an item has
    already spent thirty minutes in the model is the wrong time.

    `exempt` is the loop's own bookkeeping -- the checklist and the run
    log. Both had to be excused, and not as a convenience:

      - The log is written into the project directory by the loop itself.
        Any project that has ever committed it (nothing stops that; one of
        this machine's projects has an empty .gitignore) would be refused
        a run because of a file the previous run created.
      - The checklist is the thing the user just finished editing. Being
        told to commit your todo list before the loop will read it is
        backwards, and the loop rewrites that file itself anyway.

    Neither can reach a merge: restore_todo() reverts any change an item
    makes to the checklist, and a modified file the item's branch does not
    touch is not something a fast-forward has to overwrite (verified: git
    fast-forwards happily past unrelated local modifications). Everything
    else still blocks the run.
    """
    result = _git(["status", "--porcelain", "--untracked-files=no"], project_dir)
    dirty = [line for line in result.stdout.splitlines()
             if line.strip() and line[3:].strip().strip('"') not in exempt]
    return (not dirty), "\n".join(dirty)


def materialize(project_dir: Path, worktree: Path) -> list[str]:
    """Copy/link the untracked files an item needs. Returns what it did."""
    actions = []
    for name in MATERIALIZE_FILES:
        src, dst = project_dir / name, worktree / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            actions.append(f"copied {name}")
    for name in MATERIALIZE_LINKS:
        src, dst = project_dir / name, worktree / name
        if src.is_dir() and not dst.exists():
            dst.symlink_to(src.resolve(), target_is_directory=True)
            actions.append(f"linked {name}")
    return actions


def worktrees_root(project_dir: Path) -> Path:
    """Where this project's item worktrees live.

    Deliberately outside the project: a worktree inside the checkout would
    be walked by aider's repo map and by the loop's own preflight file
    scan, and would need a .gitignore entry in every project to stay out of
    the way.
    """
    return Path.home() / ".cache" / "aider-loop" / project_dir.name / "worktrees"


def runs_root(project_dir: Path) -> Path:
    """Where this project's per-run JSON records live (see
    aider_loop.record_run) - one directory per invocation, one
    item-NNN.json per item processed in it. Shared here so a review tool
    can find them without duplicating the path."""
    return Path.home() / ".cache" / "aider-loop" / project_dir.name / "runs"


def prune_stale(project_dir: Path, log=print) -> list[str]:
    """Remove worktrees left behind by a run that was killed.

    A run that dies between `create()` and `remove()` -- OOM, Ctrl+C, a
    reboot -- leaves its worktree checked out and registered with git.
    Nothing cleans those up on its own, so they accumulate, each one
    holding a branch that `git branch -D` will then refuse to delete.
    Measured: an OOM kill during an item left exactly this behind.

    Only the working directories go. The branches stay, because a killed
    item's branch may hold real work and is the same thing a parked item's
    branch is -- something to read, not something to silently discard.
    """
    root = worktrees_root(project_dir)
    if not root.is_dir():
        return []
    removed = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        _git(["worktree", "remove", "--force", str(path)], project_dir)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    _git(["worktree", "prune"], project_dir)
    if removed:
        log(f"Cleaned up {len(removed)} worktree(s) left by an earlier run that was "
            f"killed: {', '.join(removed)}. Their branches were kept.")
    return removed


def branch_name(index: int) -> str:
    return f"aider-loop/item-{index}-{time.strftime('%Y%m%d-%H%M%S')}"


def create(project_dir: Path, index: int, base: str) -> tuple[Path, str]:
    """Add a worktree for one item, branched from `base`. Returns (path, branch)."""
    branch = branch_name(index)
    path = worktrees_root(project_dir) / branch.rsplit("/", 1)[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    result = _git(["worktree", "add", "-q", "-b", branch, str(path), base], project_dir)
    if result.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
    return path, branch


def merge_ff(project_dir: Path, branch: str) -> tuple[bool, str]:
    """Fast-forward the project's checked-out branch onto a passing item.

    ff-only on purpose. Items are branched from the checkout's current
    HEAD and merged back one at a time, so a fast-forward is always
    possible in the intended flow; if it isn't, something moved the branch
    underneath the run and a merge commit would be the wrong way to find
    that out.
    """
    result = _git(["merge", "--ff-only", branch], project_dir)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def restore_todo(worktree: Path, base: str, todo_name: str) -> bool:
    """Put the todo file back if the item edited it. Returns True if it did.

    The checklist is the loop's own bookkeeping, not the item's work, and
    the authoritative copy is the one in the project checkout. But in a
    project where the todo file is tracked (blockroad's TODO.md is), the
    worktree contains a copy that aider can see and, given a task about
    checklists or documentation, edit -- and a passing item's branch gets
    fast-forwarded into the real checkout, carrying that edit into the file
    the loop is concurrently rewriting status markers into.
    """
    changed = _git(["diff", "--name-only", base, "HEAD", "--", todo_name], worktree)
    if not changed.stdout.strip():
        return False
    _git(["checkout", base, "--", todo_name], worktree)
    _git(["commit", "-q", "-m", f"aider-loop: restore {todo_name} (loop bookkeeping, not the item's work)",
          "--", todo_name], worktree)
    return True


def remove(project_dir: Path, worktree: Path, branch: str, keep_branch: bool) -> None:
    """Tear down a worktree. The branch survives when the item didn't pass.

    A parked item's branch is the only remaining copy of what the model
    produced -- the whole point of parking rather than reverting is that
    it's still there to look at (`git log aider-loop/item-N`) once the run
    is over.
    """
    _git(["worktree", "remove", "--force", str(worktree)], project_dir)
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
    _git(["worktree", "prune"], project_dir)
    if not keep_branch:
        _git(["branch", "-D", branch], project_dir)
