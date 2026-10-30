"""What every agent is to forge: a Task in, an Attempt out.

The gate (worktree, scope, content, validate, TDD, review) is what decides
whether work merges; an agent only does the work inside the worktree. See
docs/design-pluggable-agents.md.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class Task:
    text: str                       # item text, --doer-preamble already applied
    worktree: Path
    log_path: Path
    files: list[str] | None = None  # expected_files(); None or empty: agent decides
    model: str | None = None
    hard: bool = False              # estimate_difficulty() == HARD
    think_off: bool = False         # set by the retry loop after a truncated attempt
    scope: list[str] = field(default_factory=list)  # the item's `scope:` globs, if any


@dataclass
class Attempt:
    ok: bool                        # the agent produced something; merging is the gate's call
    output: str
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens, seconds, ...


class Agent(Protocol):
    name: str
    traits: frozenset[str]          # "single_shot", "explores", "runs_shell", "local", "metered"
    commit_label: str               # prefix of the commit forge makes for its work

    def attempt(self, task: Task, log: Callable[[str], None]) -> Attempt: ...
