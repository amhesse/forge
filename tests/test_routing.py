"""Routing (design step 5): [agents] exact / explore / escalate / review.

Unit tests for the rules, and real `forge run`s with stub aider and agy for
the paths: each item goes to its kind's agent, a parked item escalates to
the next agent in a fresh worktree, and a typo in [agents] stops the run
before anything runs.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_leash_hooks as hooks
from forge import runner
from forge.agents import routing


def with_cfg(settings):
    return mock.patch.object(routing, "cfg", lambda *k, default=None: settings.get(k, default))


class TestRules(unittest.TestCase):
    def test_parse_entry_keeps_colons_in_the_model(self):
        self.assertEqual(routing.parse_entry("claude"), ("claude", None))
        self.assertEqual(routing.parse_entry("ollama:gemma4:26b"), ("ollama", "gemma4:26b"))

    def test_item_kind(self):
        self.assertEqual(routing.item_kind("In `mod.py`, add x."), "exact")
        self.assertEqual(routing.item_kind("Add enemies.\nscope: src/game/**"), "explore")
        self.assertEqual(routing.item_kind("In `README.md` too.\nscope: src/**"), "explore")
        self.assertEqual(routing.item_kind("Add enemies."), "explore")

    def test_first_step_follows_the_kind_unless_backend_was_explicit(self):
        settings = {("agents", "exact"): "lite", ("agents", "explore"): "opencode:qwen3.8-64k"}
        pick = lambda text, explicit: (lambda s: (s.backend, s.model))(routing.first_step(text, "m", explicit))
        with with_cfg(settings):
            self.assertEqual(pick("In `a.py`, x.", False), ("lite", "m"))
            self.assertEqual(pick("scope: src/**", False), ("opencode", "qwen3.8-64k"))
            self.assertEqual(pick("In `a.py`, x.", True), (None, "m"))
        with with_cfg({}):
            self.assertEqual(pick("In `a.py`, x.", False), (None, "m"))

    def test_escalate_chain_skips_the_agent_that_just_failed(self):
        first = routing.Step("opencode", None, "")
        with with_cfg({("agents", "escalate"): ["opencode", "claude", "agy:gemini-3.8"]}):
            steps = routing.escalation_steps(SimpleNamespace(), first)
        self.assertEqual([(s.backend, s.model) for s in steps], [("claude", None), ("agy", "gemini-3.8")])

    def test_without_a_chain_the_old_fallback_rules_hold(self):
        first = routing.Step(None, "qwen", "")
        with with_cfg({}):
            esc = lambda **a: [(s.backend, s.model) for s in routing.escalation_steps(SimpleNamespace(**a), first)]
            self.assertEqual(esc(fallback_backend="claude", fallback_model=None), [("claude", None)])
            self.assertEqual(esc(fallback_backend="aider", fallback_model="big"), [("aider", "big")])
            self.assertEqual(esc(fallback_backend="aider", fallback_model="qwen"), [])
            self.assertEqual(esc(fallback_backend="aider", fallback_model=None), [])

    def test_default_review_and_cli_wins(self):
        with mock.patch.object(routing, "cfg", lambda *k, default=None: "ollama:gemma4:26b" if k == ("agents", "review") else default):
            args = SimpleNamespace(review=None, review_model=None)
            runner.apply_default_review(args)
            self.assertEqual((args.review, args.review_model), ("ollama", "gemma4:26b"))
            args = SimpleNamespace(review="claude", review_model=None)
            runner.apply_default_review(args)
            self.assertEqual((args.review, args.review_model), ("claude", None))

    def test_config_typos_are_caught(self):
        settings = {("agents", "exact"): "lyte", ("agents", "escalate"): ["claude", "opencod"],
                    ("agents", "review"): "olama"}
        cfg = lambda *k, default=None: settings.get(k, default)
        with mock.patch.object(runner, "cfg", cfg), mock.patch.object(routing, "cfg", cfg):
            problems = runner.check_agents_config()
        self.assertEqual(len(problems), 3, problems)


STUB_AGY = r'''#!{python}
import json, pathlib, re, sys
prompt = sys.argv[-1]
named = re.findall(r"`([\w./-]+\.[A-Za-z]{{1,7}})`", prompt)
path = named[0] if named else "src/game/enemy.py"
p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text("value = 'agy'\n")
print(json.dumps({{"status": "SUCCESS", "response": "done", "duration_seconds": 1,
                   "usage": {{"input_tokens": 1, "output_tokens": 1}}}}))
'''
STUB_AIDER_NOTHING = "#!/bin/sh\nexit 0\n"   # aider that never edits anything


class RoutingEndToEnd(unittest.TestCase):
    setUp = hooks.LeashHookTest.setUp
    init_repo = hooks.LeashHookTest.init_repo
    record = hooks.LeashHookTest.record

    def forge_run(self, *extra):
        """hooks.forge_run without its fixed --backend aider: routing only
        applies when no backend is given on the command line."""
        env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "PYTHONPATH": str(hooks.SRC),
               "STUB_LOG": str(self.stub_log), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        r = subprocess.run([sys.executable, "-m", "forge.cli", "run", "--project-dir", str(self.proj),
                            "--todo-file", "TODO.md", "--sleep-between", "0", "--max-validation-retries", "1",
                            *extra], cwd=self.proj, capture_output=True, text=True, timeout=120, env=env)
        return r.stdout + r.stderr, r.returncode

    def stub(self, name, text):
        p = self.bin / name
        p.write_text(text)
        p.chmod(0o755)

    def records(self):
        import json
        return [json.loads(p.read_text()) for p in sorted((self.cache / "runs").glob("*/item-*.json"))]

    def test_each_kind_goes_to_its_agent(self):
        self.init_repo("- [ ] In `mod.py`, set a value.\n- [ ] Add an enemy.\n  scope: src/game/**\n",
                       '[validate]\nchecks = ["python"]\n[agents]\nexact = "aider"\nexplore = "agy"\n')
        self.stub("agy", STUB_AGY.format(python=sys.executable))
        log, rc = self.forge_run()
        todo = (self.proj / "TODO.md").read_text()
        self.assertEqual(todo.count("- [x]"), 2, log)
        self.assertEqual((self.proj / "mod.py").read_text(), "value = 1\n")         # the aider stub's work
        self.assertEqual((self.proj / "src/game/enemy.py").read_text(), "value = 'agy'\n")
        self.assertEqual([r.get("agent") for r in self.records()], ["aider", "agy"])
        self.assertIn("Routing to agy (explore item)", log)

    def test_a_parked_item_escalates_to_the_next_agent(self):
        self.init_repo("- [ ] In `mod.py`, set a value.\n",
                       '[validate]\nchecks = ["python"]\n[agents]\nexact = "aider"\nescalate = ["aider", "agy"]\n')
        self.stub("aider", STUB_AIDER_NOTHING)
        self.stub("agy", STUB_AGY.format(python=sys.executable))
        log, rc = self.forge_run()
        self.assertIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        self.assertEqual((self.proj / "mod.py").read_text(), "value = 'agy'\n")
        rec = self.records()[-1]
        self.assertEqual(rec["fallback_used"], "agy")
        self.assertIn("Escalating to agy", log)
        self.assertNotIn("Escalating to aider", log)   # not handed straight back to the agent that failed

    def test_a_typo_in_agents_stops_the_run_before_anything_runs(self):
        self.init_repo("- [ ] In `mod.py`, set a value.\n", '[agents]\nexact = "lyte"\n')
        log, rc = self.forge_run()
        self.assertEqual(rc, 1)
        self.assertIn("unknown agent 'lyte'", log)
        self.assertIn("- [ ]", (self.proj / "TODO.md").read_text())


if __name__ == "__main__":
    unittest.main()
