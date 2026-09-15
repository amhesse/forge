"""End-to-end tests for the benchmark hooks in `forge run`:
--snapshot-first-attempt, --repair-from/--repair-note, and --backend claude.

Like forge eval, these run the real `forge run` as a subprocess against a
throwaway repo with a stub `aider` on PATH - no model involved - so what is
tested is the loop's actual behaviour, including git state it leaves behind.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# Driven by the prompt text forge sends, and logs every prompt it receives.
STUB_AIDER = r'''#!/usr/bin/env python3
import os, pathlib, re, subprocess, sys
msg = sys.argv[sys.argv.index("--message") + 1]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(msg + "\n=====\n")

def write(path, text):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "stub"], check=True)

paths = re.findall(r"`([\w./-]+\.[A-Za-z]{1,7})`", msg)
if "Write ONLY the test" in msg:
    write(paths[0], "import unittest\nfrom calc import add\n\n"
          "class T(unittest.TestCase):\n    def test_add(self):\n"
          "        self.assertEqual(add(2, 3), 5)\n")
elif "currently fails because" in msg:
    impl = re.search(r"Implement `([^`]+)`", msg).group(1)
    write(impl, "def add(a, b):\n    return a + b\n")
elif "did not pass validation" in msg or "already committed in this repository" in msg:
    write(paths[0], "value = 2\n")
elif "BADFIRST" in msg:
    write(paths[0], "def broken(:\n")
else:
    write(paths[0], "value = 1\n")
'''


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class LeashHookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.name = f"forge-hooktest-{os.getpid()}-{id(self)}"
        self.proj = root / self.name
        self.proj.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        stub = self.bin / "aider"
        stub.write_text(STUB_AIDER)
        stub.chmod(0o755)
        self.stub_log = root / "prompts.log"
        self.stub_log.write_text("")
        self.cache = Path.home() / ".cache" / "aider-loop" / self.name
        self.addCleanup(shutil.rmtree, self.cache, True)
        self.addCleanup(self.tmp.cleanup)

    def init_repo(self, todo: str, toml: str = '[validate]\nchecks = ["python"]\n'):
        git(self.proj, "init", "-q", "-b", "master")
        (self.proj / "TODO.md").write_text(todo)
        (self.proj / ".aiderloop.toml").write_text(toml)
        git(self.proj, "add", "-A")
        git(self.proj, "commit", "-q", "-m", "base")
        return git(self.proj, "rev-parse", "HEAD")

    def forge_run(self, *extra: str) -> str:
        env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
               "PYTHONPATH": str(SRC), "STUB_LOG": str(self.stub_log),
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        r = subprocess.run([sys.executable, "-m", "forge.cli", "run", "--project-dir", str(self.proj),
                            "--todo-file", "TODO.md", "--sleep-between", "0",
                            "--max-validation-retries", "1", "--backend", "aider", *extra],
                           cwd=self.proj, capture_output=True, text=True, timeout=120, env=env)
        return r.stdout + r.stderr

    def record(self) -> dict:
        recs = sorted((self.cache / "runs").glob("*/item-*.json"))
        self.assertTrue(recs, "no run record written")
        return json.loads(recs[-1].read_text())

    def first_attempt_refs(self) -> list[str]:
        out = git(self.proj, "for-each-ref", "--format=%(refname) %(objectname)", "refs/leash/")
        return [line for line in out.splitlines() if line]

    def test_snapshot_pins_first_attempt_before_validation_retry(self):
        self.init_repo("- [ ] BADFIRST: in `mod.py`, set a value.\n")
        log = self.forge_run("--snapshot-first-attempt")
        self.assertIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        rec = self.record()
        snap = rec["first_attempt_commit"]
        [ref] = self.first_attempt_refs()
        self.assertTrue(ref.startswith("refs/leash/first-attempt/aider-loop/item-1-"))
        self.assertTrue(ref.endswith(snap))
        # The snapshot is the broken first attempt; what merged is the fixed retry.
        self.assertEqual(git(self.proj, "show", f"{snap}:mod.py"), "def broken(:")
        self.assertEqual((self.proj / "mod.py").read_text(), "value = 2\n")
        self.assertNotEqual(snap, git(self.proj, "rev-parse", "HEAD"))

    def test_no_snapshot_without_flag(self):
        self.init_repo("- [ ] In `mod.py`, set a value.\n")
        self.forge_run()
        self.assertEqual(self.first_attempt_refs(), [])
        self.assertNotIn("first_attempt_commit", self.record())

    def test_tdd_snapshot_contains_test_and_implementation(self):
        self.init_repo(
            "- [ ] TDD: implement `calc.py` so `test_calc.py` passes: add(a, b) returns the sum.\n",
            '[validate]\nchecks = ["python"]\ncommands = ["python3 -m unittest -q test_calc"]\n')
        log = self.forge_run("--snapshot-first-attempt")
        self.assertIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        snap = self.record()["first_attempt_commit"]
        files = git(self.proj, "show", "--name-only", "--format=", snap)
        tree = git(self.proj, "ls-tree", "-r", "--name-only", snap)
        self.assertIn("calc.py", files)
        self.assertIn("test_calc.py", tree)

    def test_repair_starts_from_parked_attempt_and_checks_full_diff(self):
        base = self.init_repo("- [ ] In `mod.py`, set a value.\n")
        # A parked earlier attempt: broken file plus an unrelated stray file.
        git(self.proj, "checkout", "-q", "-b", "aider-loop/item-1-parked")
        (self.proj / "mod.py").write_text("def broken(:\n")
        (self.proj / "stray.txt").write_text("not part of the task\n")
        git(self.proj, "add", "-A")
        git(self.proj, "commit", "-q", "-m", "parked attempt")
        git(self.proj, "checkout", "-q", "master")
        note = Path(self.tmp.name) / "note.txt"
        note.write_text("validation failed: SyntaxError in mod.py")

        log = self.forge_run("--repair-from", "aider-loop/item-1-parked", "--repair-note", str(note))

        self.assertIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        rec = self.record()
        self.assertEqual(rec["mode"], "repair")
        self.assertEqual(rec["repair_from"], "aider-loop/item-1-parked")
        self.assertEqual(rec["base"], base)
        self.assertEqual((self.proj / "mod.py").read_text(), "value = 2\n")
        # restore_unnamed_files judged the whole change against the original
        # base, so the parked attempt's stray file never reached the checkout.
        self.assertFalse((self.proj / "stray.txt").exists())
        prompts = self.stub_log.read_text()
        self.assertIn("already committed in this repository", prompts)
        self.assertIn("SyntaxError in mod.py", prompts)

    def test_repair_from_unknown_rev_blocks(self):
        self.init_repo("- [ ] In `mod.py`, set a value.\n")
        self.forge_run("--repair-from", "no-such-branch")
        self.assertIn("- [!]", (self.proj / "TODO.md").read_text())
        self.assertIn("not found", self.record()["reason"])

    def test_repair_note_requires_repair_from(self):
        self.init_repo("- [ ] In `mod.py`, set a value.\n")
        log = self.forge_run("--repair-note", "x.txt")
        self.assertIn("--repair-note requires --repair-from", log)

    def test_agy_backend_edits_commits_and_repairs(self):
        base = self.init_repo("- [ ] In `mod.py`, set a value.\n")
        git(self.proj, "checkout", "-q", "-b", "aider-loop/item-1-parked")
        (self.proj / "mod.py").write_text("def broken(:\n")
        git(self.proj, "add", "-A")
        git(self.proj, "commit", "-q", "-m", "parked attempt")
        git(self.proj, "checkout", "-q", "master")
        args_log = Path(self.tmp.name) / "agy-args.json"
        stub = self.bin / "agy"
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, pathlib, sys\n"
            f"pathlib.Path({str(args_log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
            "pathlib.Path('mod.py').write_text('value = 2\\n')\n"
            "print(json.dumps({'status': 'SUCCESS', 'response': 'fixed', 'duration_seconds': 3.5,\n"
            "                  'usage': {'input_tokens': 100, 'output_tokens': 20, 'thinking_tokens': 5,\n"
            "                            'cache_read_tokens': 50}}))\n")
        stub.chmod(0o755)
        note = Path(self.tmp.name) / "note.txt"
        note.write_text("validation failed")

        log = self.forge_run("--backend", "agy", "--models", "gemini-3.8-flash-high",
                             "--repair-from", "aider-loop/item-1-parked", "--repair-note", str(note))

        self.assertIn("- [x]", (self.proj / "TODO.md").read_text(), log)
        self.assertEqual((self.proj / "mod.py").read_text(), "value = 2\n")
        argv = json.loads(args_log.read_text())
        # --print must be the last flag, immediately followed by the prompt.
        self.assertEqual(argv[-2], "--print")
        self.assertIn("already committed in this repository", argv[-1])
        self.assertEqual(argv[argv.index("--model") + 1], "gemini-3.8-flash-high")
        for flag in ("--sandbox", "--output-format"):
            self.assertIn(flag, argv)
        rec = self.record()
        self.assertEqual(rec["tokens"]["prompt_tokens"], 150)
        self.assertEqual(rec["tokens"]["completion_tokens"], 25)
        self.assertEqual(rec["base"], base)

    def test_agy_error_is_logged_and_blocks(self):
        self.init_repo("- [ ] In `mod.py`, set a value.\n")
        stub = self.bin / "agy"
        stub.write_text(f"#!{sys.executable}\nimport json, sys\n"
                        "print(json.dumps({'status': 'ERROR', 'error': 'quota exceeded', 'usage': {}}))\n"
                        "sys.exit(1)\n")
        stub.chmod(0o755)
        log = self.forge_run("--backend", "agy", "--max-retries", "0")
        self.assertIn("agy reported an error: quota exceeded", log)
        self.assertIn("- [!]", (self.proj / "TODO.md").read_text())
        self.assertIn("itself failed", self.record()["reason"])

    def test_backend_claude_is_accepted(self):
        self.init_repo("- [ ] In `mod.py`, set a value.\n")
        stub = self.bin / "claude"
        stub.write_text("#!/bin/sh\nexit 1\n")
        stub.chmod(0o755)
        log = self.forge_run("--backend", "claude")
        self.assertNotIn("invalid choice", log)
        self.assertIn("Editing backend: claude", log)


if __name__ == "__main__":
    unittest.main()
