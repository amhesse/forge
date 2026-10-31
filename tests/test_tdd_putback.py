"""TDD: an edit to the test after the red phase is undone, not parked (2026-10-30).

Claude Code edits a test it believes is wrong, despite the prompt; that used
to park correct work. Now forge puts the red-phase test back and judges the
implementation against it, which is exactly as strict as before.
"""
import sys
import unittest

import test_leash_hooks as hooks

# Red phase: a real test. Green phase: an implementation (right or wrong per
# IMPL) AND a rewrite of the test that would pass anything.
STUB = r'''#!{python}
import os, pathlib, re, subprocess, sys
msg = sys.argv[sys.argv.index("--message") + 1]
def write(path, text):
    pathlib.Path(path).write_text(text)
if "Write ONLY the test" in msg:
    write("test_calc.py", "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n"
          "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n")
else:
    write("calc.py", {impl!r})
    write("test_calc.py", "import unittest\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        pass\n")
subprocess.run(["git", "add", "-A"], check=True)
subprocess.run(["git", "commit", "-q", "-m", "stub"], check=True)
'''


class PutBack(unittest.TestCase):
    setUp = hooks.LeashHookTest.setUp
    init_repo = hooks.LeashHookTest.init_repo
    forge_run = hooks.LeashHookTest.forge_run
    record = hooks.LeashHookTest.record

    def run_with(self, impl):
        self.init_repo("- [ ] TDD: implement `calc.py` so `test_calc.py` passes: add(a, b) returns the sum.\n",
                       '[validate]\nchecks = ["python"]\ncommands = ["python3 -m unittest -q test_calc"]\n')
        stub = self.bin / "aider"
        stub.write_text(STUB.format(python=sys.executable, impl=impl))
        stub.chmod(0o755)
        return self.forge_run("--max-validation-retries", "0")

    def test_right_code_with_an_edited_test_merges_against_the_original_test(self):
        log = self.run_with("def add(a, b):\n    return a + b\n")
        self.assertIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        self.assertIn("assertEqual(add(2, 3), 5)", (self.proj / "test_calc.py").read_text())
        self.assertEqual(self.record()["test_restores"], 1)
        self.assertIn("put it back as the red phase wrote it", log)

    def test_wrong_code_that_only_passed_the_edited_test_is_not_merged(self):
        log = self.run_with("def add(a, b):\n    return a - b\n")
        self.assertNotIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        self.assertFalse((self.proj / "calc.py").exists())


if __name__ == "__main__":
    unittest.main()
