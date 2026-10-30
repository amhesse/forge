"""An exploring agent takes a TDD item only with --review (2026-10-30).

On the Leash honest set, exploring agents merged 3 wrong TDD changes in 96
trials: their own test agreed with their own wrong code. See
runner.tdd_exploring_problem.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_leash_hooks as hooks
from forge import runner


def problem(backend, review=None, settings=None):
    settings = settings or {}
    with mock.patch.object(runner, "cfg", lambda *k, default=None: settings.get(k, default)):
        return runner.tdd_exploring_problem(SimpleNamespace(review=review), backend)


class TestRule(unittest.TestCase):
    def test_single_shot_agents_are_never_held(self):
        for backend in ("lite", "aider"):
            self.assertIsNone(problem(backend))

    def test_exploring_agents_need_a_reviewer(self):
        for backend in ("opencode", "claude", "agy"):
            self.assertIn("needs --review", problem(backend), backend)
            self.assertIsNone(problem(backend, review="ollama"), backend)

    def test_the_project_can_opt_out(self):
        self.assertIsNone(problem("opencode", settings={("tdd", "exploring_without_review"): True}))

    def test_the_configured_backend_counts_when_none_is_passed(self):
        self.assertIn("opencode", problem(None, settings={("model", "backend"): "opencode"}))


STUB_AGY = r'''#!{python}
import pathlib, sys
pathlib.Path({called!r}).write_text("called")
'''


class EndToEnd(unittest.TestCase):
    setUp = hooks.LeashHookTest.setUp
    init_repo = hooks.LeashHookTest.init_repo
    forge_run = hooks.LeashHookTest.forge_run
    record = hooks.LeashHookTest.record

    def test_a_tdd_item_for_an_exploring_agent_without_review_is_parked_unrun(self):
        self.init_repo("- [ ] TDD: implement `calc.py` so `test_calc.py` passes: add(a, b) returns the sum.\n",
                       '[validate]\nchecks = ["python"]\n')
        called = Path(self.tmp.name) / "agy-called"
        stub = self.bin / "agy"
        stub.write_text(STUB_AGY.format(python=sys.executable, called=str(called)))
        stub.chmod(0o755)
        log = self.forge_run("--backend", "agy", "--models", "m")
        self.assertIn("- [?]", (self.proj / "TODO.md").read_text(), log)
        self.assertIn("needs --review", self.record()["reason"])
        self.assertFalse(called.exists(), "the agent ran anyway")


if __name__ == "__main__":
    unittest.main()
