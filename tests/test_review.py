"""forge.agents.review: every reviewer behind one interface (design step 2).

One prompt, one verdict parser, and a ReviewTask that has no field for the
doer's reasoning (the Leash finding: the reasoning makes a reviewer worse).
"""
import dataclasses
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from forge import agents, claude_editor, runner, spec_compiler
from forge.agents import review

WT = Path("/tmp/wt")


class TestParseVerdict(unittest.TestCase):
    def test_verdict_on_the_last_line_after_reasoning(self):
        self.assertIs(review.parse_verdict("The diff misses the empty case.\n\nREJECT"), False)
        self.assertIs(review.parse_verdict("Looks right.\nAPPROVE\n"), True)

    def test_markdown_and_punctuation_around_the_word(self):
        for text in ("**APPROVE**", "`APPROVE`", "Approve.", "_reject_"):
            self.assertIsNotNone(review.parse_verdict(text), text)

    def test_the_last_verdict_line_wins(self):
        self.assertIs(review.parse_verdict("APPROVE\nOn reflection, no.\nREJECT"), False)

    def test_no_clear_verdict_is_none_not_a_guess(self):
        for text in ("", "I approve of this change.", "APPROVE with changes", "Looks fine to me"):
            self.assertIsNone(review.parse_verdict(text), text)


class TestPrompt(unittest.TestCase):
    def test_review_task_cannot_carry_the_doers_reasoning(self):
        fields = {f.name for f in dataclasses.fields(agents.ReviewTask)}
        self.assertEqual(fields, {"text", "diff", "worktree", "tdd", "model"})

    def test_task_and_diff_and_the_last_line_rule(self):
        p = review.review_prompt(agents.ReviewTask("add slugify", "+def slugify", WT), can_read_repo=False)
        self.assertIn("Task:\nadd slugify", p)
        self.assertIn("Diff:\n+def slugify", p)
        self.assertIn("VERY LAST LINE", p)
        self.assertNotIn("read files in the repository", p)

    def test_repo_reading_only_for_reviewers_that_can(self):
        p = review.review_prompt(agents.ReviewTask("t", "d", WT), can_read_repo=True)
        self.assertIn("You may read files in the repository", p)

    def test_tdd_instruction_for_every_reviewer(self):
        for can_read in (True, False):
            p = review.review_prompt(agents.ReviewTask("t", "d", WT, tdd=True), can_read_repo=can_read)
            self.assertIn("This is a TDD item", p)


class TestReviewers(unittest.TestCase):
    task = agents.ReviewTask("task", "diff", WT, model="m")

    def test_registry(self):
        self.assertEqual(set(agents.REVIEWERS), {"claude", "ollama", "agy"})
        with self.assertRaises(ValueError):
            agents.get_reviewer("nobody")

    def test_claude_uses_read_only_tools_and_parses(self):
        with mock.patch.object(claude_editor, "_run", return_value=(True, "ok\nAPPROVE")) as run:
            v = agents.get_reviewer("claude").review(self.task, print)
        self.assertIs(v.approved, True)
        self.assertEqual(run.call_args.args[2], claude_editor.REVIEW_TOOLS)
        self.assertEqual(run.call_args.args[3], "m")

    def test_claude_that_could_not_run_is_none(self):
        with mock.patch.object(claude_editor, "_run", return_value=(False, "usage limit")):
            self.assertIsNone(agents.get_reviewer("claude").review(self.task, print).approved)

    def test_ollama_parses_and_an_exception_is_none(self):
        with mock.patch.object(spec_compiler, "call_ollama", return_value="bad logic\nREJECT") as call:
            self.assertIs(agents.get_reviewer("ollama").review(self.task, print).approved, False)
        self.assertEqual(call.call_args.kwargs["model"], "m")
        self.assertEqual(call.call_args.kwargs["num_ctx"], 4096)
        with mock.patch.object(spec_compiler, "call_ollama", side_effect=TimeoutError("slow")):
            self.assertIsNone(agents.get_reviewer("ollama").review(self.task, print).approved)

    def test_ollama_says_when_the_diff_will_not_fit(self):
        big = agents.ReviewTask("task", "+x\n" * 8000, WT)
        logged = []
        with mock.patch.object(spec_compiler, "call_ollama", return_value="APPROVE"):
            agents.get_reviewer("ollama").review(big, logged.append)
        self.assertTrue(any("will cut it" in m for m in logged), logged)

    def test_agy_runs_in_the_worktree(self):
        done = subprocess.CompletedProcess([], 0, stdout="fine\nAPPROVE\n", stderr="")
        with mock.patch.object(review.subprocess, "run", return_value=done) as run:
            v = agents.get_reviewer("agy").review(self.task, print)
        self.assertIs(v.approved, True)
        self.assertEqual(run.call_args.kwargs["cwd"], str(WT))
        with mock.patch.object(review.subprocess, "run", side_effect=FileNotFoundError("agy")):
            self.assertIsNone(agents.get_reviewer("agy").review(self.task, print).approved)


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


class Spy:
    name = "spy"

    def __init__(self, approved):
        self.approved, self.seen = approved, []

    def review(self, task, log):
        self.seen.append(task)
        return agents.Verdict(self.approved, "spy says so")


class TestGate(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        git(self.repo, "init", "-q", "-b", "master")
        (self.repo / "mod.py").write_text("value = 0\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "mod.py").write_text("value = 1\n")
        git(self.repo, "commit", "-qam", "work")
        self.finished = []

    def gate(self, spy, tdd=False):
        args = SimpleNamespace(review="spy", review_model="rm", max_validation_retries=0, todo_file="TODO.md")
        record = {}
        finish = lambda status, why: self.finished.append((status, why)) or (status, why)
        with mock.patch.dict(agents.REVIEWERS, {"spy": spy}):
            out = runner.run_review_gate(self.repo, self.base, "set value", ["mod.py"], args,
                                         self.repo / "log", record, finish, tdd=tdd)
        return out, record

    def test_approved_merges_and_the_reviewer_saw_only_task_and_diff(self):
        spy = Spy(True)
        out, record = self.gate(spy, tdd=True)
        self.assertIsNone(out)
        self.assertEqual(record["review"], {"reviewer": "spy", "rounds": 1})
        seen = spy.seen[0]
        self.assertEqual((seen.text, seen.tdd, seen.model, seen.worktree), ("set value", True, "rm", self.repo))
        self.assertIn("+value = 1", seen.diff)

    def test_rejected_with_no_fix_rounds_left_parks_blocked(self):
        out, record = self.gate(Spy(False))
        self.assertEqual(out[0], runner.STATUS_BLOCKED)
        self.assertEqual(record["review_feedback"], "spy says so")

    def test_a_review_that_could_not_run_parks_for_a_human(self):
        out, _ = self.gate(Spy(None))
        self.assertEqual(out[0], runner.STATUS_NEEDS_REVIEW)

    def test_no_review_flag_is_a_no_op(self):
        args = SimpleNamespace(review=None)
        self.assertIsNone(runner.run_review_gate(self.repo, self.base, "t", [], args, None, {}, None))


if __name__ == "__main__":
    unittest.main()
