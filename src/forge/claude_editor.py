"""Escalation backend: hand an item to the Claude Code CLI (`claude -p`).

Used as `--fallback-backend claude` (retry an item the local model
parked) and `--review claude` (a second opinion on a diff before merge).
Neither replaces a check: whatever Claude writes still goes through the
same scope, leakage, syntax and validate gates as a local model's work,
because it runs in the same per-item worktree.

Billing: `claude -p` uses the logged-in Claude subscription unless
ANTHROPIC_API_KEY is set, in which case it silently bills the API
instead. The key is stripped from the child's environment so escalation
can never turn into a per-token bill. It still counts against the
plan's usage limits, which is why this is an escalation path and not
the workhorse.
"""
import ipaddress
import json
import os
import subprocess
import urllib.parse
from pathlib import Path

CLAUDE_BIN = "claude"
DEFAULT_TIMEOUT = 1800

# No Bash: the loop runs the project's validate commands itself, and an
# unattended agent with a shell is a bigger blast radius than an item needs.
EDIT_TOOLS = "Read,Edit,Write,Glob,Grep"
REVIEW_TOOLS = "Read,Glob,Grep"

_last_usage: dict = {}


def last_usage() -> dict:
    return dict(_last_usage)


def _is_loopback_endpoint(url: str) -> bool:
    """True only for a base URL whose host is this machine.

    Parsed rather than pattern-matched on purpose: the host of
    "http://127.0.0.1.example.com/" is example.com's, not ours, and a
    substring test for "127.0.0.1" would hand it the credentials this
    function exists to withhold.
    """
    if not url:
        return False
    try:
        host = urllib.parse.urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _env() -> dict:
    """The child's environment, with API credentials removed.

    The exception is a base URL pointing at this machine - a local
    OpenAI/Anthropic-compatible server such as Strata. The CLI needs a
    token to talk to one at all, and there is no metered API behind it to
    bill, so stripping the token there only breaks the call without
    protecting anything. Anything not demonstrably loopback keeps the
    original behaviour: no credentials, so escalation cannot become a
    per-token bill.

    Note for anyone reading usage numbers from this path: the CLI prices
    its own reported usage from the model NAME, so a local endpoint still
    reports a non-zero total_cost_usd. Measured against Strata: 31k
    prompt tokens "cost" $0.09 while nothing was billed. Treat cost from
    this backend as meaningless whenever the base URL is local.
    """
    env = dict(os.environ)
    if _is_loopback_endpoint(env.get("ANTHROPIC_BASE_URL", "")):
        return env
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


def _run(cwd: Path, prompt: str, tools: str, model: str | None,
         timeout: int, log) -> tuple[bool, str]:
    """One `claude -p` call. Returns (ok, result text) and records usage."""
    global _last_usage
    _last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    cmd = [CLAUDE_BIN, "-p", "--output-format", "json",
           "--permission-mode", "acceptEdits", "--allowedTools", tools]
    if model:
        cmd += ["--model", model]
    try:
        result = subprocess.run(cmd, input=prompt, cwd=str(cwd), env=_env(),
                                capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        log(f"claude CLI not found on PATH ({CLAUDE_BIN})")
        return False, "claude CLI not found"
    except subprocess.TimeoutExpired:
        log(f"claude -p timed out after {timeout}s")
        return False, "timeout"
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        out = (result.stdout + result.stderr).strip()
        log(f"claude -p returned non-JSON output (exit {result.returncode}): {out[-1000:]}")
        return False, out
    usage = data.get("usage") or {}
    _last_usage = {
        "prompt_tokens": int(usage.get("input_tokens", 0))
                         + int(usage.get("cache_read_input_tokens", 0))
                         + int(usage.get("cache_creation_input_tokens", 0)),
        "completion_tokens": int(usage.get("output_tokens", 0)),
        "seconds": float(data.get("duration_ms") or data.get("duration_api_ms") or 0) / 1000,
    }
    text = data.get("result") or ""
    if data.get("is_error") or result.returncode != 0:
        log(f"claude -p reported an error: {text[-1000:] or result.stderr[-1000:]}")
        return False, text
    return True, text


def run_claude_on_item(worktree: Path, item_text: str, log, files: list[str] | None = None,
                       model: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> tuple[bool, str]:
    scope = ""
    if files:
        scope = ("\n\nOnly modify these file(s): " + ", ".join(files)
                 + ". Read any other file you need for context, but do not change it.")
    prompt = (f"Implement this task in the current repository.{scope}\n\n"
              f"Task:\n{item_text}\n\n"
              "Make the edits directly. Do not commit, and do not create files the task "
              "doesn't call for. When done, reply with one line summarising the change.")
    ok, text = _run(worktree, prompt, EDIT_TOOLS, model, timeout, log)
    if not ok:
        return False, text
    # The loop judges `git diff base HEAD`, so like the other backends this
    # one has to leave its work committed. Everything is staged: files
    # outside the item's scope are put back by restore_unnamed_files after.
    git = lambda *a: subprocess.run(["git", *a], cwd=str(worktree), capture_output=True, text=True)
    git("add", "-A")
    if not git("status", "--porcelain").stdout.strip():
        log("[claude] finished but changed nothing")
        return False, "claude made no changes"
    commit = git("commit", "-q", "-m", f"claude: {item_text.splitlines()[0][:72]}")
    if commit.returncode != 0:
        return False, f"git commit failed: {commit.stderr}"
    log(f"[claude] committed: {text.strip()[-300:]}")
    return True, text


def review_diff(worktree: Path, item_text: str, diff: str, log,
                model: str | None = None, timeout: int = 600, tdd: bool = False
                ) -> tuple[bool | None, str]:
    """Returns (approved, feedback). approved is None if the review itself failed.

    `tdd=True` adds an extra instruction found necessary by a real case,
    not a hypothetical one: a TDD item can merge a wrong implementation
    that passes its OWN red-phase test, when that test is narrower than
    the task it's supposed to verify (measured directly: tdd_budget's
    self-written test never exercised a category with both income and
    spending entries, so an implementation that summed both signs passed
    it while failing the real spec). A generic "is this diff correct"
    review can miss that, because the diff and its own test agree with
    each other - the mismatch is between the test and the task, which is
    exactly what this asks for explicitly instead of leaving implicit."""
    prompt = ("Review this diff against the task it was meant to implement. You may read "
              "files in the repository for context. Look for incorrect logic, missed "
              "requirements and broken behaviour - not style.\n\n")
    if tdd:
        prompt += (
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
    prompt += ("Explain your reasoning first if you want, but the VERY LAST LINE of your "
              "reply must be exactly one word, nothing else on that line: APPROVE if it "
              "correctly implements the task, or REJECT if it doesn't.\n\n"
              f"Task:\n{item_text}\n\nDiff:\n{diff}")
    ok, text = _run(worktree, prompt, REVIEW_TOOLS, model, timeout, log)
    if not ok:
        return None, text
    text = text.strip()
    # Claude doesn't reliably put the verdict on the first line even when
    # asked to (seen live: reasoning first, APPROVE last) - a naive
    # first-line check reads that as a rejection and escalates work that
    # was already fine. Take the last line that is exactly one of the two
    # words; only genuinely ambiguous output (neither, or reasoning that
    # happens to end in a line saying something else) falls through to None.
    for line in reversed(text.splitlines()):
        word = line.strip().strip(".*_ ").upper()
        if word in ("APPROVE", "REJECT"):
            return word == "APPROVE", text
    log(f"claude review gave no clear APPROVE/REJECT verdict: {text[-500:]}")
    return None, text
