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
import json
import subprocess
import urllib.request
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
    # The test is evidence, not the thing under review. Two real cases set
    # this. (1) A TDD item can merge a wrong implementation that passes its
    # OWN red-phase test: tdd_budget's test never exercised a category with
    # both income and spending, so summing both signs passed it; on the Leash
    # honest set, exploring agents' tests agreed with their own wrong code,
    # cents read as dollars. (2) The wording before 2026-10-30 told the
    # reviewer to reject a test narrower than the task. It rejected correct
    # implementations for test gaps that a fix round can't close, because
    # after the red phase only the implementation may change. That parked
    # 7 of 8 correct Claude Code trials.
    "This is a TDD item: one of the changed files is a test the same model wrote for "
    "itself before implementing the rest, so it may be as wrong as the code or narrower "
    "than the task, and passing it proves little. Judge the IMPLEMENTATION against the "
    "TASK TEXT yourself, case by case: every case the task describes (edge cases, "
    "boundary values, signs, units such as cents vs dollars, empty or missing input, "
    "combinations the task says matter). REJECT if the implementation is wrong for any "
    "of them, even when its own test passes. Do NOT reject only because the test is "
    "incomplete when the implementation itself is right; the test can't be changed now.\n\n"
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
    URL = "http://localhost:11434/api/generate"

    def review(self, task: ReviewTask, log) -> Verdict:
        prompt = review_prompt(task, can_read_repo=False)
        num_ctx = int(cfg("review", "ollama_num_ctx", default=self.DEFAULT_NUM_CTX))
        # Ollama cuts a prompt longer than num_ctx without saying so, and a
        # reviewer that only saw half the diff can approve it. Say so.
        approx_tokens = len(prompt) // 4
        if approx_tokens > num_ctx * 0.9:
            log(f"ollama review prompt is ~{approx_tokens} tokens but num_ctx is {num_ctx}: Ollama "
                f"will cut it. Raise [review] ollama_num_ctx in .aiderloop.toml.")
        try:
            out = self._generate(task.model or self.DEFAULT_MODEL, prompt, num_ctx)
        except (OSError, ValueError) as e:  # unreachable, timed out, bad reply: the review didn't run
            return Verdict(None, str(e))
        return _verdict(self.name, out, log)

    def _generate(self, model: str, prompt: str, num_ctx: int) -> str:
        """Its own request, not spec_compiler.call_ollama. That one calls
        sys.exit() when Ollama is unreachable, which inside a review ended
        the whole forge run. It also leaves thinking on: gemma4:26b, a
        thinking model, spent its whole budget reasoning and answered with
        nothing, so every review "gave no clear verdict" (2026-10-30).
        Thinking stays off here; the verdict is what's asked for."""
        payload = json.dumps({"model": model, "prompt": prompt, "stream": True, "think": False,
                              "options": {"num_ctx": num_ctx, "num_predict": 2048}}).encode()
        req = urllib.request.Request(self.URL, data=payload, headers={"Content-Type": "application/json"})
        chunks = []
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise ValueError(obj["error"])
                chunks.append(obj.get("response", ""))
                if obj.get("done"):
                    break
        return "".join(chunks).strip()


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
