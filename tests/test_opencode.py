"""OpenCode as a sandboxed forge agent (design step 4).

The end-to-end tests run the real `opencode` binary inside the real bwrap
sandbox, driven by a scripted fake model server (fake_openai.py), so they
need no GPU but do need opencode and bwrap installed (skipped otherwise).
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_leash_hooks as hooks
from fake_openai import FakeOpenAI
from forge import agents
from forge.agents import opencode, sandbox

HAVE_REAL = bool(shutil.which("opencode")) and sandbox.available()

SAMPLE = "\n".join(json.dumps(e) for e in [
    {"type": "step_start", "part": {"type": "step-start"}},
    {"type": "tool_use", "part": {"tool": "write", "state": {"status": "completed"}}},
    {"type": "step_finish", "part": {"reason": "tool-calls", "tokens": {"input": 100, "output": 20, "reasoning": 5,
                                                                        "cache": {"read": 50, "write": 0}}}},
    {"type": "text", "part": {"type": "text", "text": "Added the enemy."}},
    {"type": "step_finish", "part": {"reason": "stop", "tokens": {"input": 200, "output": 10, "reasoning": 0}}},
]) + "\nnot json\n"


class TestPieces(unittest.TestCase):
    def test_parse_events(self):
        text, usage, errors = opencode.parse_events(SAMPLE)
        self.assertEqual(text, "Added the enemy.")
        self.assertEqual(usage, {"prompt_tokens": 350, "completion_tokens": 35, "steps": 2})
        self.assertEqual(errors, [])
        self.assertTrue(opencode.parse_events(json.dumps({"type": "error", "error": {"m": "x"}}))[2])

    def test_final_answer_is_a_step_that_ended_without_calling_tools(self):
        line = lambda reason: json.dumps({"type": "step_finish", "part": {"reason": reason}})
        self.assertTrue(opencode.final_answer(line("stop")))
        self.assertTrue(opencode.final_answer(line("length")))
        self.assertFalse(opencode.final_answer(line("tool-calls")))
        self.assertFalse(opencode.final_answer(json.dumps({"type": "text", "part": {"text": "x"}})))
        self.assertFalse(opencode.final_answer("not json"))

    def test_model_names(self):
        self.assertEqual(opencode.provider_model("qwen3.8-64k"), ("forge", "qwen3.8-64k"))
        self.assertEqual(opencode.provider_model("ollama/qwen3.8-64k"), ("ollama", "qwen3.8-64k"))
        self.assertEqual(opencode.provider_model("hf.co/org/m:Q4"), ("forge", "hf.co/org/m:Q4"))

    def test_config_is_self_contained_and_cannot_fetch_or_push(self):
        with mock.patch.object(opencode, "cfg", lambda *k, default=None: "http://x:1/v1" if k == ("opencode", "base_url") else default):
            c = opencode.config_for("m")
        self.assertEqual(c["model"], "forge/m")
        self.assertEqual(c["provider"]["forge"]["options"]["baseURL"], "http://x:1/v1")
        self.assertEqual(c["permission"]["webfetch"], "deny")
        self.assertEqual(c["permission"]["bash"]["git push*"], "deny")

    def test_prompt_states_files_and_scope(self):
        p = opencode.build_prompt(agents.Task(text="add enemy", worktree=Path("/w"), log_path=Path("/l"),
                                              files=["README.md"], scope=["src/game/**"]))
        self.assertIn("Only modify these file(s): README.md; and files matching: src/game/**", p)
        self.assertIn("Do not commit", p)

    def test_refuses_to_run_without_the_sandbox(self):
        with mock.patch.object(sandbox, "available", return_value=False):
            a = agents.get_agent("opencode").attempt(agents.Task(text="t", worktree=Path("/w"), log_path=Path("/l")), print)
        self.assertFalse(a.ok)
        self.assertIn("only sandboxed", a.output)

    def test_registered_with_the_right_traits(self):
        self.assertEqual(agents.get_agent("opencode").traits, frozenset({"explores", "runs_shell", "local"}))


@unittest.skipUnless(sandbox.available(), "bwrap not installed")
class TestSandbox(unittest.TestCase):
    def test_the_worktree_is_writable_and_home_is_not_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp) / "wt"
            wt.mkdir()
            escape = Path.home() / "forge-sandbox-escape-test"
            r = sandbox.run(["sh", "-c", f"echo in > inside.txt; echo out > {escape}; "
                                         f"ls {Path.home() / '.ssh'} 2>&1; touch /usr/x 2>&1; true"], wt)
            self.assertEqual((wt / "inside.txt").read_text(), "in\n")
            self.assertFalse(escape.exists(), "a write outside the worktree reached the real home")
            self.assertIn("No such file", r.stdout)
            self.assertIn("Read-only", r.stdout)


    def test_a_sandbox_that_lingers_after_the_final_answer_is_stopped(self):
        # The hang seen live: the agent had exited, bwrap had not.
        import time
        with tempfile.TemporaryDirectory() as tmp:
            t = time.monotonic()
            r = sandbox.run(["sh", "-c", "echo FINAL; sleep 300 & sleep 300"], Path(tmp), timeout=60,
                            done_when=lambda line: "FINAL" in line, grace=1)
            self.assertLess(time.monotonic() - t, 15)
            self.assertEqual(r.returncode, 0)
            self.assertIn("stopped it", r.stderr)

    def test_the_agent_never_inherits_stdin(self):
        # `cat` would wait forever on an inherited stdin that stays open.
        with tempfile.TemporaryDirectory() as tmp:
            r = sandbox.run(["cat"], Path(tmp), timeout=10)
            self.assertEqual((r.returncode, r.stdout), (0, ""))

    def test_a_hard_timeout_still_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.TimeoutExpired):
                sandbox.run(["sleep", "300"], Path(tmp), timeout=1)


@unittest.skipUnless(HAVE_REAL, "needs opencode and bwrap")
class OpenCodeEndToEnd(unittest.TestCase):
    """The real `forge run --backend opencode`, the real OpenCode binary in the
    real sandbox, and a fake model that writes in scope, writes out of scope
    with a shell command, tries to escape the sandbox, and stops."""
    setUp = hooks.LeashHookTest.setUp
    init_repo = hooks.LeashHookTest.init_repo
    record = hooks.LeashHookTest.record

    def forge_run(self, *extra):
        """hooks' forge_run, but a hang fails with what forge was doing.
        That's how the stdin hang (sandbox.run's docstring) was found: these
        tests hung whenever the test runner's own stdin was an open pipe."""
        try:
            return hooks.LeashHookTest.forge_run(self, *extra)
        except subprocess.TimeoutExpired:
            log = self.proj / "aider_loop.log"
            tail = log.read_text()[-3000:] if log.exists() else "(no forge log)"
            procs = subprocess.run(["ps", "-eo", "pid,etimes,stat,cmd", "--forest"], capture_output=True,
                                   text=True).stdout
            procs = "\n".join(l[:200] for l in procs.splitlines() if any(k in l for k in ("opencode", "bwrap", "forge")))
            self.fail(f"forge run hung.\n--- forge log tail:\n{tail}\n--- processes:\n{procs}")

    def run_with(self, script, todo="- [ ] Add an enemy.\n  scope: src/game/**\n"):
        fake = FakeOpenAI(script)
        self.addCleanup(fake.close)
        self.init_repo(todo, f'[validate]\nchecks = ["python"]\n[opencode]\nbase_url = "{fake.url}"\ntimeout = 60\n')
        (self.proj / "src").mkdir()
        (self.proj / "src/other.py").write_text("touched = False\n")
        hooks.git(self.proj, "add", "-A")
        hooks.git(self.proj, "commit", "-q", "-m", "other")
        return fake, self.forge_run("--backend", "opencode", "--models", "fake-model")

    def test_in_scope_work_merges_out_of_scope_work_is_put_back_and_nothing_escapes(self):
        escape = Path.home() / "forge-opencode-escape-test"
        fake, log = self.run_with([
            {"tool": "write", "args": {"filePath": "src/game/enemy.py", "content": "ENEMY = 1\n"}},
            {"tool": "bash", "args": {"command": "echo 'touched = True' > src/other.py", "description": "x"}},
            {"tool": "bash", "args": {"command": f"echo pwned > {escape}", "description": "escape"}},
            {"text": "Added the enemy."},
        ])
        self.assertIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        self.assertEqual((self.proj / "src/game/enemy.py").read_text(), "ENEMY = 1\n")
        self.assertEqual((self.proj / "src/other.py").read_text(), "touched = False\n")
        self.assertFalse(escape.exists())
        rec = self.record()
        self.assertGreater(rec["tokens"]["prompt_tokens"], 0)
        subject = hooks.git(self.proj, "log", "--format=%s", "-3")
        self.assertIn("opencode: Add an enemy.", subject)
        self.assertTrue(any(r.get("tools") for r in fake.requests))

    def test_a_model_that_never_edits_is_made_no_changes_not_success(self):
        _, log = self.run_with([{"text": "I would add an enemy."}])
        self.assertNotIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        self.assertIn("finished but changed nothing", log)


if __name__ == "__main__":
    unittest.main()
