"""forge.agents: the registry that replaced runner._dispatch_editor's if-chain.

Step 1 of docs/design-pluggable-agents.md is a pure refactor, so these check
that each agent does exactly what its old branch did, and that usage is per
thread for every editor (claude and agy used module globals before, which
parallel workers could read across items). Step 1b moved committing out of
the agents and into the supervisor: TestCommitAttempt and TestDispatch run
against a real throwaway repository.
"""
import subprocess
import tempfile
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


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


class RepoTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        git(self.repo, "init", "-q", "-b", "master")
        git(self.repo, "config", "user.email", "t@t")
        git(self.repo, "config", "user.name", "t")
        (self.repo / "mod.py").write_text("value = 0\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")

    def files_in_head(self):
        return set(git(self.repo, "diff", "--name-only", self.base, "HEAD").split())


class TestCommitAttempt(RepoTest):
    def test_commits_the_work_but_never_what_materialize_put_there(self):
        (self.repo / "mod.py").write_text("value = 1\n")
        (self.repo / "new.py").write_text("x = 1\n")
        (self.repo / ".aiderloop.toml").write_text("[validate]\n")   # copied in, untracked
        (self.repo / "venv").symlink_to(self.repo, target_is_directory=True)
        self.assertEqual(agents.commit_attempt(self.repo, "lite_editor: item"), (True, ""))
        self.assertEqual(self.files_in_head(), {"mod.py", "new.py"})
        self.assertEqual(git(self.repo, "log", "-1", "--format=%s"), "lite_editor: item")

    def test_a_tracked_config_file_the_item_edits_is_still_committed(self):
        (self.repo / ".aiderloop.toml").write_text("[validate]\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "track config")
        self.base = git(self.repo, "rev-parse", "HEAD")
        (self.repo / ".aiderloop.toml").write_text("[validate]\ncommands = []\n")
        self.assertEqual(agents.commit_attempt(self.repo, "m"), (True, ""))
        self.assertEqual(self.files_in_head(), {".aiderloop.toml"})

    def test_nothing_to_commit_is_not_an_error(self):
        (self.repo / ".forge.toml").write_text("x = 1\n")   # only a materialized file
        self.assertEqual(agents.commit_attempt(self.repo, "m"), (False, ""))


class Spy:
    name, traits, commit_label = "spy", frozenset(), "spy"

    def __init__(self, ok=True, edit=None, self_commit=False):
        self.ok, self.edit, self.self_commit, self.task = ok, edit, self_commit, None

    def attempt(self, task, log):
        self.task = task
        if self.edit:
            (task.worktree / "mod.py").write_text(self.edit)
            if self.self_commit:
                git(task.worktree, "commit", "-qam", "agent's own commit")
        return agents.Attempt(self.ok, "o", {"seconds": 1.0})


class TestDispatch(RepoTest):
    def dispatch(self, spy, **kw):
        with mock.patch.dict(agents.REGISTRY, {"spy": spy}):
            return runner._dispatch_editor(self.repo, "set value\nmore", LOG, files=["mod.py"], model="m",
                                           backend_override="spy", **kw)

    def test_supervisor_commits_with_the_agents_label(self):
        out = self.dispatch(Spy(edit="value = 1\n"))
        self.assertEqual(out, (True, "o", {"seconds": 1.0}))
        self.assertEqual(git(self.repo, "log", "-1", "--format=%s"), "spy: set value")
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")

    def test_ok_but_no_change_is_a_failure(self):
        out = self.dispatch(Spy(edit=None))
        self.assertEqual(out[:2], (False, "spy made no changes"))
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.base)

    def test_an_agent_that_committed_itself_needs_nothing_more(self):
        out = self.dispatch(Spy(edit="value = 2\n", self_commit=True))
        self.assertTrue(out[0])
        self.assertEqual(git(self.repo, "log", "-1", "--format=%s"), "agent's own commit")

    def test_a_failed_attempt_is_not_committed(self):
        out = self.dispatch(Spy(ok=False, edit="value = 3\n"))
        self.assertFalse(out[0])
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.base)

    def test_difficulty_is_estimated_when_not_given(self):
        spy = Spy(edit="value = 1\n")
        with mock.patch.object(runner, "estimate_difficulty", return_value=runner.DIFFICULTY_HARD):
            self.dispatch(spy, force_think_off=True)
        self.assertTrue(spy.task.hard)
        self.assertTrue(spy.task.think_off)


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
