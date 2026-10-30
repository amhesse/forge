"""forge.agents: the registry that replaced runner._dispatch_editor's if-chain.

Step 1 of docs/design-pluggable-agents.md is a pure refactor, so these check
that each agent does exactly what its old branch did, and that usage is per
thread for every editor (claude and agy used module globals before, which
parallel workers could read across items).
"""
import threading
import unittest
from pathlib import Path
from unittest import mock

from forge import agents, agy_editor, claude_editor, lite_editor, runner
from forge.agents import adapters

WT = Path("/tmp/wt")
LOG = Path("/tmp/wt.log")


def fake_cfg(values):
    def cfg(*keys, default=None):
        return values.get(keys, default)
    return cfg


class TestRegistry(unittest.TestCase):
    def test_every_old_backend_is_registered(self):
        self.assertEqual(set(agents.REGISTRY), {"lite", "aider", "claude", "agy"})

    def test_unknown_or_empty_name_gets_aider_like_the_old_fallthrough(self):
        for name in (None, "", "no-such-backend"):
            self.assertEqual(agents.get_agent(name).name, "aider")

    def test_traits_say_which_agents_explore(self):
        self.assertIn("single_shot", agents.get_agent("lite").traits)
        self.assertIn("explores", agents.get_agent("claude").traits)


class TestLiteAgent(unittest.TestCase):
    def run_lite(self, values, hard=False, think_off=False, model=None):
        with mock.patch.object(adapters, "cfg", fake_cfg(values)), \
             mock.patch.object(lite_editor, "run_lite_on_item", return_value=(True, "out")) as run, \
             mock.patch.object(lite_editor, "last_usage", return_value={"prompt_tokens": 7}):
            result = agents.get_agent("lite").attempt(
                agents.Task(text="item", worktree=WT, log_path=LOG, files=["a.py"], model=model,
                            hard=hard, think_off=think_off), print)
        return run.call_args, result

    def test_plain_item_passes_only_the_model(self):
        call, result = self.run_lite({("model", "author"): "qwen3.8:27b"})
        self.assertEqual(call.args, (WT, "item", LOG))
        self.assertEqual(call.kwargs, {"files": ["a.py"], "model": "qwen3.8:27b"})
        self.assertEqual((result.ok, result.output, result.usage), (True, "out", {"prompt_tokens": 7}))

    def test_hard_item_gets_the_hard_overrides(self):
        values = {("model", "lite_num_predict"): 2000, ("model", "lite_hard_num_predict"): 12000,
                  ("model", "lite_hard_budget_directive"): True, ("model", "lite_hard_think_off"): True}
        call, _ = self.run_lite(values, hard=True, model="m")
        self.assertEqual(call.kwargs, {"files": ["a.py"], "model": "m", "num_predict": 12000,
                                       "budget_directive": True, "think": False})

    def test_easy_item_ignores_the_hard_overrides(self):
        values = {("model", "lite_num_predict"): 2000, ("model", "lite_hard_think_off"): True}
        call, _ = self.run_lite(values, hard=False, model="m")
        self.assertEqual(call.kwargs, {"files": ["a.py"], "model": "m", "num_predict": 2000})

    def test_think_off_after_a_truncated_attempt_applies_to_any_item(self):
        call, _ = self.run_lite({}, hard=False, think_off=True, model="m")
        self.assertIs(call.kwargs["think"], False)


class TestOtherAgents(unittest.TestCase):
    def task(self):
        return agents.Task(text="item", worktree=WT, log_path=LOG, files=["a.py"], model="m")

    def test_claude_passes_through_and_reads_its_usage(self):
        with mock.patch.object(claude_editor, "run_claude_on_item", return_value=(True, "done")) as run, \
             mock.patch.object(claude_editor, "last_usage", return_value={"seconds": 3.0}):
            result = agents.get_agent("claude").attempt(self.task(), print)
        self.assertEqual(run.call_args.args[:2], (WT, "item"))
        self.assertEqual(run.call_args.kwargs, {"files": ["a.py"], "model": "m"})
        self.assertEqual(result.usage, {"seconds": 3.0})

    def test_agy_passes_through(self):
        with mock.patch.object(agy_editor, "run_agy_on_item", return_value=(False, "no")) as run, \
             mock.patch.object(agy_editor, "last_usage", return_value={}):
            result = agents.get_agent("agy").attempt(self.task(), print)
        self.assertEqual(run.call_args.kwargs, {"files": ["a.py"], "model": "m"})
        self.assertFalse(result.ok)

    def test_aider_uses_runner_functions_and_parses_tokens(self):
        with mock.patch.object(runner, "run_aider_on_item", return_value=(True, "Tokens: 1k sent")) as run, \
             mock.patch.object(runner, "parse_aider_token_line", return_value={"prompt_tokens": 1000}):
            result = agents.get_agent("aider").attempt(self.task(), print)
        self.assertEqual(run.call_args.args, (WT, "item", LOG))
        self.assertEqual(result.usage, {"prompt_tokens": 1000})


class TestDispatch(unittest.TestCase):
    def test_backend_override_wins_and_difficulty_is_estimated_when_not_given(self):
        seen = {}

        class Spy:
            name, traits = "spy", frozenset()

            def attempt(self, task, log):
                seen["task"] = task
                return agents.Attempt(True, "o", {"seconds": 1.0})

        with mock.patch.dict(agents.REGISTRY, {"spy": Spy()}), \
             mock.patch.object(runner, "estimate_difficulty", return_value=runner.DIFFICULTY_HARD):
            out = runner._dispatch_editor(WT, "item", LOG, files=None, model="m", backend_override="spy",
                                          force_think_off=True)
        self.assertEqual(out, (True, "o", {"seconds": 1.0}))
        self.assertTrue(seen["task"].hard)
        self.assertTrue(seen["task"].think_off)


class TestUsageIsPerThread(unittest.TestCase):
    def test_claude_and_agy_usage_do_not_cross_threads(self):
        for module in (claude_editor, agy_editor):
            got, ready = {}, threading.Barrier(2)

            def worker(n):
                module._usage_local.usage = {"prompt_tokens": n}
                ready.wait()  # both have written before either reads
                got[n] = module.last_usage()["prompt_tokens"]

            threads = [threading.Thread(target=worker, args=(n,)) for n in (1, 2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(got, {1: 1, 2: 2}, module.__name__)


if __name__ == "__main__":
    unittest.main()
