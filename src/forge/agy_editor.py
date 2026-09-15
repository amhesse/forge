"""Escalation backend: hand an item to the `agy` CLI (Gemini and other
models it offers, e.g. `gemini-3.8-flash-high`).

Same contract as claude_editor: the CLI edits files in the item's own
worktree, this module commits whatever it changed, and every one of the
loop's checks still judges the result exactly as it would a local model's.

CLI details that matter here, found by running it rather than assumed:
- `--print` takes the prompt as its own value, so it must come last - a
  flag placed after `--print` is read as the prompt and the real prompt is
  silently dropped (agy exits 2 and says so).
- `--mode accept-edits` lets it write files without an interactive
  approval; `--sandbox` restricts its terminal access while still allowing
  edits. The loop runs the project's validate commands itself, so the
  agent has no need for a shell.
- `--output-format json` prints one object: status "SUCCESS" or "ERROR",
  `response`, `error`, `duration_seconds` and `usage` (input, output,
  thinking and cache-read tokens). Errors also exit non-zero.
- Print mode stops waiting after `--print-timeout` (default 5m) and then
  reports status SUCCESS with an empty response and exit 0, printing
  "[agy] print timeout after ... with turn in progress" to stderr - it
  looks exactly like a model that chose to change nothing. Found live: a
  Gemini repair that needed longer than 5 minutes was retried three times
  and parked as a real failure. The print timeout is therefore set to this
  module's own timeout, and a timeout message is treated as a timeout.
"""

import re

import json
import subprocess
from pathlib import Path

AGY_BIN = "agy"
DEFAULT_TIMEOUT = 1800
# How long past agy's own print timeout the process may run before it is killed.
_PROCESS_GRACE = 60
_PRINT_TIMEOUT_RE = re.compile(r"print timeout after (\S+) with turn in progress")

_last_usage: dict = {}


def last_usage() -> dict:
    return dict(_last_usage)


def _run(cwd: Path, prompt: str, model: str | None, timeout: int, log) -> tuple[bool, str]:
    """One `agy` print-mode call. Returns (ok, response text) and records usage."""
    global _last_usage
    _last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    cmd = [AGY_BIN, "--mode", "accept-edits", "--sandbox", "--output-format", "json",
           "--print-timeout", f"{timeout}s"]
    if model:
        cmd += ["--model", model]
    cmd += ["--print", prompt]  # must be last: --print consumes the next argument
    try:
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                                timeout=timeout + _PROCESS_GRACE, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        log(f"agy CLI not found on PATH ({AGY_BIN})")
        return False, "agy CLI not found"
    except subprocess.TimeoutExpired:
        log(f"agy timed out after {timeout}s")
        return False, "timeout"
    waited = _PRINT_TIMEOUT_RE.search(result.stderr or "")
    if waited:
        # Not a result: agy gave up waiting on a turn still in progress.
        log(f"agy timed out after {waited.group(1)} (print timeout, turn still in progress)")
        return False, "timeout"
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        out = (result.stdout + result.stderr).strip()
        log(f"agy returned non-JSON output (exit {result.returncode}): {out[-1000:]}")
        return False, out
    usage = data.get("usage") or {}
    _last_usage = {
        "prompt_tokens": int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_tokens", 0)),
        "completion_tokens": int(usage.get("output_tokens", 0)) + int(usage.get("thinking_tokens", 0)),
        "seconds": float(data.get("duration_seconds") or 0),
    }
    if data.get("status") != "SUCCESS" or result.returncode != 0:
        detail = data.get("error") or data.get("response") or result.stderr
        log(f"agy reported an error: {(detail or '').strip()[-1000:]}")
        return False, detail or ""
    return True, data.get("response") or ""


def run_agy_on_item(worktree: Path, item_text: str, log, files: list[str] | None = None,
                    model: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> tuple[bool, str]:
    scope = ""
    if files:
        scope = ("\n\nOnly modify these file(s): " + ", ".join(files)
                 + ". Read any other file you need for context, but do not change it.")
    prompt = (f"Implement this task in the current repository.{scope}\n\n"
              f"Task:\n{item_text}\n\n"
              "Make the edits directly. Do not run shell commands, do not commit, and do not "
              "create files the task doesn't call for. When done, reply with one line "
              "summarising the change.")
    ok, text = _run(worktree, prompt, model, timeout, log)
    if not ok:
        return False, text
    # Same as claude_editor: the loop judges `git diff base HEAD`, so the
    # work has to be committed; out-of-scope files are put back afterwards
    # by restore_unnamed_files.
    git = lambda *a: subprocess.run(["git", *a], cwd=str(worktree), capture_output=True, text=True)
    git("add", "-A")
    if not git("status", "--porcelain").stdout.strip():
        log("[agy] finished but changed nothing")
        return False, "agy made no changes"
    commit = git("commit", "-q", "-m", f"agy: {item_text.splitlines()[0][:72]}")
    if commit.returncode != 0:
        return False, f"git commit failed: {commit.stderr}"
    log(f"[agy] committed: {text.strip()[-300:]}")
    return True, text
