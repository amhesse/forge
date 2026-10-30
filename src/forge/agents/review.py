"""Reviewers: the second model that reads a diff before it merges (design step 2).

Every reviewer gets the same thing: the task and the diff (plus, for one that
can read the repository, the worktree to look around in). Never the doer's
reasoning. That is a Leash finding, made structural here: gemma4:26b caught
100% of planted bugs from the diff alone and 89% once the doer's reasoning was
added, because the reasoning quotes an injected instruction as a requirement
and the reviewer accepts it. ReviewTask has no field to put the reasoning in.

One prompt and one verdict parser for all of them. Before this, claude had
its own prompt (with the TDD instruction and a robust parser), while ollama
and agy shared a copied, weaker parser and never got the TDD instruction.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .. import claude_editor
from ..config import cfg


@dataclass(frozen=True)
class ReviewTask:
    text: str            # the item's task text
    diff: str            # git diff base..HEAD of the worktree
    worktree: Path
    tdd: bool = False
    model: str | None = None


@dataclass
class Verdict:
    approved: bool | None  # None: the review didn't run, or gave no clear verdict
    feedback: str


class Reviewer(Protocol):
    name: str

    def review(self, task: ReviewTask, log: Callable[[str], None]) -> Verdict: ...


TDD_INSTRUCTION = (
    # A real case, not a hypothetical one: a TDD item can merge a wrong
    # implementation that passes its OWN red-phase test when that test is
    # narrower than the task (tdd_budget's test never exercised a category
    # with both income and spending, so summing both signs passed it).
    "This is a TDD item: one of the changed files is a test the same model wrote "
    "for itself before implementing the rest. Do not just check that the "
    "implementation passes ITS OWN test - separately check whether that test's "
    "assertions actually cover every case the task describes (edge cases, boundary "
    "values, combinations the task mentions - e.g. mixed positive and negative "
    "values when the task says both matter, empty/missing input if the task "
    "specifies behavior for it). A test that is narrower than the task can pass "
    "against a wrong implementation; reject if the test doesn't prove the task is "
    "actually satisfied, even if the diff currently passes its own test.\n\n"
)


def review_prompt(task: ReviewTask, can_read_repo: bool) -> str:
    prompt = "Review this diff against the task it was meant to implement. "
    if can_read_repo:
        prompt += "You may read files in the repository for context. "
    prompt += "Look for incorrect logic, missed requirements and broken behaviour - not style.\n\n"
    if task.tdd:
        prompt += TDD_INSTRUCTION
    prompt += ("Explain your reasoning first if you want, but the VERY LAST LINE of your "
               "reply must be exactly one word, nothing else on that line: APPROVE if it "
               "correctly implements the task, or REJECT if it doesn't.\n\n"
               f"Task:\n{task.text}\n\nDiff:\n{task.diff}")
    return prompt


def parse_verdict(text: str) -> bool | None:
    """The last line that is exactly APPROVE or REJECT (markdown stripped).
    Models don't reliably put the verdict first even when asked (seen live:
    reasoning first, APPROVE last), and a model's prose can mention either
    word, so only a line that is nothing but the word counts. Neither: None,
    which parks the item for a human rather than guessing."""
    for line in reversed((text or "").strip().splitlines()):
        word = line.strip().strip(".*_` ").upper()
        if word in ("APPROVE", "REJECT"):
            return word == "APPROVE"
    return None


def _verdict(name: str, text: str, log) -> Verdict:
    approved = parse_verdict(text)
    if approved is None:
        log(f"{name} review gave no clear APPROVE/REJECT verdict: {text[-500:]}")
    return Verdict(approved, text)


class ClaudeReviewer:
    name = "claude"

    def review(self, task: ReviewTask, log) -> Verdict:
        ok, text = claude_editor._run(task.worktree, review_prompt(task, can_read_repo=True),
                                      claude_editor.REVIEW_TOOLS, task.model, 600, log)
        if not ok:
            return Verdict(None, text)
        return _verdict(self.name, text, log)


class OllamaReviewer:
    name = "ollama"
    DEFAULT_MODEL = "qwen3-coder:30b"
    DEFAULT_NUM_CTX = 4096

    def review(self, task: ReviewTask, log) -> Verdict:
        from .. import spec_compiler
        prompt = review_prompt(task, can_read_repo=False)
        num_ctx = int(cfg("review", "ollama_num_ctx", default=self.DEFAULT_NUM_CTX))
        # Ollama cuts a prompt longer than num_ctx without saying so, and a
        # reviewer that only saw half the diff can approve it. Say so.
        approx_tokens = len(prompt) // 4
        if approx_tokens > num_ctx * 0.9:
            log(f"ollama review prompt is ~{approx_tokens} tokens but num_ctx is {num_ctx}: Ollama "
                f"will cut it. Raise [review] ollama_num_ctx in .aiderloop.toml.")
        try:
            out = spec_compiler.call_ollama(model=task.model or self.DEFAULT_MODEL, prompt=prompt,
                                            url=spec_compiler.DEFAULT_OLLAMA_URL, num_ctx=num_ctx,
                                            num_predict=2048, timeout=90.0).strip()
        except Exception as e:  # unreachable, timed out: the review didn't run
            return Verdict(None, str(e))
        return _verdict(self.name, out, log)


class AgyReviewer:
    name = "agy"

    def review(self, task: ReviewTask, log) -> Verdict:
        # Runs in the worktree, so "read files for context" means this item's
        # files. Before, it ran in whatever directory forge was started from.
        try:
            out = subprocess.run(["agy", "--print", review_prompt(task, can_read_repo=True)],
                                 cwd=str(task.worktree), capture_output=True, text=True,
                                 check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as e:
            return Verdict(None, str(e))
        return _verdict(self.name, out, log)


REVIEWERS: dict[str, Reviewer] = {r.name: r for r in (ClaudeReviewer(), OllamaReviewer(), AgyReviewer())}


def get_reviewer(name: str) -> Reviewer:
    if name not in REVIEWERS:
        raise ValueError(f"unknown reviewer {name!r}; one of {', '.join(sorted(REVIEWERS))}")
    return REVIEWERS[name]
