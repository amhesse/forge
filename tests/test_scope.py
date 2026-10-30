"""Item scope (design step 3): a `scope:` line, and what happens to an item
that names no file and declares no scope ([scope] unscoped)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_leash_hooks as hooks
from forge import claude_editor, runner, spec_compiler, validator
from forge.todo import TodoItem
from forge.validator import expected_files, in_scope, item_scope


class TestParsing(unittest.TestCase):
    def test_scope_lines_commas_backticks_and_bullets(self):
        text = "Add enemies.\nscope: `src/game/**`, assets/*.json\n- Scope: ./docs/\n"
        self.assertEqual(item_scope(text), ["assets/*.json", "docs/", "src/game/**"])

    def test_a_scope_line_inside_a_fence_is_content_not_scope(self):
        self.assertEqual(item_scope("Write `notes.md` exactly:\n```\nscope: everything\n```\n"), [])

    def test_no_scope_line(self):
        self.assertEqual(item_scope("In `mod.py`, add x."), [])

    def test_a_scope_line_names_no_file(self):
        self.assertEqual(expected_files("Fix it.\nscope: `src/app.py`\nAlso `README.md`."), ["README.md"])


class TestMatching(unittest.TestCase):
    CASES = [
        ("src/game/**", "src/game/a.js", True), ("src/game/**", "src/game/x/y.js", True),
        ("src/game/**", "src/gameplay.js", False), ("src/game/**", "src/other.js", False),
        ("assets/*.json", "assets/e.json", True), ("assets/*.json", "assets/sub/e.json", False),
        ("**/*.py", "a.py", True), ("**/*.py", "d/e/a.py", True), ("**/*.py", "a.pyc", False),
        ("src/", "src/x/y.js", True), ("src/", "srcx/y.js", False),
        ("lib/?.c", "lib/a.c", True), ("lib/?.c", "lib/ab.c", False),
    ]

    def test_globs(self):
        for glob, path, want in self.CASES:
            self.assertEqual(in_scope(path, [glob]), want, (glob, path))


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


class TestRestore(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        git(self.repo, "init", "-q", "-b", "master")
        for f in ("src/game/a.py", "src/other.py", "README.md"):
            (self.repo / f).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / f).write_text("old\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")
        for f in ("src/game/a.py", "src/game/new.py", "src/other.py", "README.md"):
            (self.repo / f).write_text("new\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "work")

    def restore(self, expected, scope):
        with mock.patch.object(validator, "cfg", lambda *k, default=None: default):
            return validator.restore_unnamed_files(self.repo, self.base, expected, self.repo / "log", scope=scope)

    def test_inside_the_scope_and_named_files_stay_everything_else_goes_back(self):
        restored = self.restore(["README.md"], ["src/game/**"])
        self.assertEqual(restored, ["src/other.py"])
        self.assertEqual((self.repo / "src/other.py").read_text(), "old\n")
        self.assertEqual((self.repo / "src/game/new.py").read_text(), "new\n")
        self.assertEqual((self.repo / "README.md").read_text(), "new\n")

    def test_no_names_and_no_scope_restores_nothing_as_before(self):
        self.assertEqual(self.restore([], []), [])


class TestUnscopedPolicy(unittest.TestCase):
    def problem(self, settings, review=None):
        def cfg(*keys, default=None):
            return settings.get(keys, default)
        with mock.patch.object(runner, "cfg", cfg):
            return runner.unscoped_item_problem(SimpleNamespace(review=review))

    def test_refuse_is_the_default(self):
        self.assertIn("refuse", self.problem({}))

    def test_guarded_needs_validate_commands_and_a_review(self):
        guarded = {("scope", "unscoped"): "guarded"}
        self.assertIn("[validate] commands and --review", self.problem(guarded))
        self.assertIn("--review", self.problem({**guarded, ("validate", "commands"): ["pytest"]}))
        self.assertIsNone(self.problem({**guarded, ("validate", "commands"): ["pytest"]}, review="claude"))

    def test_allow_is_the_old_open_behaviour(self):
        self.assertIsNone(self.problem({("scope", "unscoped"): "allow"}))


class TestPreflight(unittest.TestCase):
    def fatal(self, text, policy=None):
        settings = {("scope", "unscoped"): policy} if policy else {}
        with mock.patch.object(spec_compiler, "cfg", lambda *k, default=None: settings.get(k, default)):
            return spec_compiler.fatal_problems(TodoItem(0, "", text, " "), Path("/nonexistent"))

    def test_a_scoped_item_is_not_fatal(self):
        self.assertEqual(self.fatal("Add enemies.\nscope: src/game/**"), [])

    def test_unscoped_is_fatal_unless_the_project_chose_otherwise(self):
        self.assertTrue(self.fatal("Add enemies to the game."))
        self.assertEqual(self.fatal("Add enemies to the game.", policy="guarded"), [])


class TestAgentPrompt(unittest.TestCase):
    def test_claude_is_told_the_scope(self):
        with mock.patch.object(claude_editor, "_run", return_value=(False, "x")) as run:
            claude_editor.run_claude_on_item(Path("/tmp"), "add enemies", print, files=["README.md"],
                                             scope=["src/game/**"])
        prompt = run.call_args.args[1]
        self.assertIn("Only modify these file(s): README.md; and files matching: src/game/**", prompt)


STUB_AGY = r'''#!{python}
import json, pathlib
for path, text in (("src/game/enemy.py", "ENEMY = 1\n"), ("src/other.py", "touched = True\n")):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
print(json.dumps({{"status": "SUCCESS", "response": "added", "duration_seconds": 1,
                   "usage": {{"input_tokens": 1, "output_tokens": 1}}}}))
'''


class ScopeEndToEnd(unittest.TestCase):
    """The real `forge run` against a throwaway repo, with a stub agy that
    writes one file inside the scope and one outside it."""
    setUp = hooks.LeashHookTest.setUp
    init_repo = hooks.LeashHookTest.init_repo
    forge_run = hooks.LeashHookTest.forge_run
    record = hooks.LeashHookTest.record

    def stub_agy(self):
        stub = self.bin / "agy"
        stub.write_text(STUB_AGY.format(python=sys.executable))
        stub.chmod(0o755)

    def test_scoped_item_merges_what_is_in_scope_and_puts_back_the_rest(self):
        self.init_repo("- [ ] Add an enemy.\n  scope: src/game/**\n")
        (self.proj / "src").mkdir()
        (self.proj / "src/other.py").write_text("touched = False\n")
        hooks.git(self.proj, "add", "-A")
        hooks.git(self.proj, "commit", "-q", "-m", "other")
        self.stub_agy()
        log = self.forge_run("--backend", "agy", "--models", "m")
        self.assertIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        self.assertEqual((self.proj / "src/game/enemy.py").read_text(), "ENEMY = 1\n")
        self.assertEqual((self.proj / "src/other.py").read_text(), "touched = False\n")
        self.assertEqual(self.record()["scope"], ["src/game/**"])

    def test_guarded_unscoped_item_without_a_review_is_parked_not_run(self):
        self.init_repo("- [ ] Add an enemy to the game.\n",
                       '[validate]\nchecks = ["python"]\ncommands = ["true"]\n[scope]\nunscoped = "guarded"\n')
        self.stub_agy()
        log = self.forge_run("--backend", "agy", "--models", "m")
        self.assertIn("- [?]", (self.proj / "TODO.md").read_text(), log)
        self.assertIn("needs --review", self.record()["reason"])
        self.assertFalse((self.proj / "src/game/enemy.py").exists())


if __name__ == "__main__":
    unittest.main()
