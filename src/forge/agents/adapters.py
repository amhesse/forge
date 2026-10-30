"""The four editors forge already had, each behind the Agent interface.

Each attempt() does exactly what its branch of runner._dispatch_editor did
before the registry existed, so swapping the dispatch for a registry lookup
changes nothing a run can observe.
"""
from .. import agy_editor, claude_editor, lite_editor
from ..config import cfg
from .base import Attempt, Task


class LiteAgent:
    commit_label = "lite_editor"
    name = "lite"
    traits = frozenset({"single_shot", "local"})

    def attempt(self, task: Task, log) -> Attempt:
        # Default (lite_editor.DEFAULT_NUM_PREDICT, 8000) is sized for editing
        # existing files. A project whose items generate new, bounded-length
        # content should set lite_num_predict lower, so a runaway generation
        # fails in ~15 s instead of burning the full budget.
        kwargs = {}
        num_predict = cfg("model", "lite_num_predict", default=None)
        if num_predict:
            kwargs["num_predict"] = num_predict
        # Hard-item overrides (see lite_editor.OUTPUT_BUDGET_DIRECTIVE): on an
        # open-ended item the model can spend the whole budget reasoning and
        # emit no file block. Exact-spec items transcribe, so they keep the
        # plain prompt.
        if task.hard:
            hard_np = cfg("model", "lite_hard_num_predict", default=None)
            if hard_np:
                kwargs["num_predict"] = hard_np
            if cfg("model", "lite_hard_budget_directive", default=False):
                kwargs["budget_directive"] = True
            if cfg("model", "lite_hard_think_off", default=False):
                kwargs["think"] = False
        # Set by the retry loop after an attempt was truncated mid-reasoning:
        # a measured fact about the last attempt, so it applies to any item.
        if task.think_off:
            kwargs["think"] = False
        ok, output = lite_editor.run_lite_on_item(
            task.worktree, task.text, task.log_path, files=task.files,
            model=task.model or cfg("model", "author"), **kwargs)
        return Attempt(ok, output, lite_editor.last_usage())


class AiderAgent:
    commit_label = "aider"
    name = "aider"
    traits = frozenset({"single_shot", "local"})

    def attempt(self, task: Task, log) -> Attempt:
        from ..runner import parse_aider_token_line, run_aider_on_item  # runner imports agents
        ok, output = run_aider_on_item(task.worktree, task.text, task.log_path,
                                       files=task.files, model=task.model)
        return Attempt(ok, output, parse_aider_token_line(output))


class ClaudeCodeAgent:
    commit_label = "claude"
    name = "claude"
    traits = frozenset({"explores", "metered"})

    def attempt(self, task: Task, log) -> Attempt:
        ok, output = claude_editor.run_claude_on_item(task.worktree, task.text, log,
                                                      files=task.files, model=task.model,
                                                      scope=task.scope)
        return Attempt(ok, output, claude_editor.last_usage())


class AgyAgent:
    commit_label = "agy"
    name = "agy"
    traits = frozenset({"explores", "metered"})

    def attempt(self, task: Task, log) -> Attempt:
        ok, output = agy_editor.run_agy_on_item(task.worktree, task.text, log,
                                                files=task.files, model=task.model,
                                                scope=task.scope)
        return Attempt(ok, output, agy_editor.last_usage())
