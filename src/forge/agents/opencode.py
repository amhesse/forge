"""OpenCode as a forge agent: the first one that explores the repository and
runs commands (design step 4). It only ever runs inside the sandbox
(agents/sandbox.py); without bubblewrap it refuses rather than run open.

`opencode run --format json --auto` is driven non-interactively. --auto
approves its tool calls, which is only acceptable because the sandbox is the
boundary: it sees this item's worktree and nothing else of $HOME. forge, not
OpenCode, commits the work and decides whether it merges.

The model server is OpenAI-compatible: Ollama's /v1 by default, or
llama-server via [opencode] base_url. Ollama's /v1 has broken tool calls
silently in agent frameworks before, so a run whose model never calls a
tool shows up as "made no changes" here rather than as a success.
"""
import json
import shutil
import time
from pathlib import Path

from ..config import cfg
from . import sandbox
from .base import Attempt, Task

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT = 1800
USER_CONFIG_DIR = Path.home() / ".config" / "opencode"   # where its provider packages are installed
USER_CACHE_DIR = Path.home() / ".cache" / "opencode"


def opencode_binary() -> Path | None:
    found = shutil.which("opencode")
    return Path(found).resolve() if found else None


def provider_model(model: str | None) -> tuple[str, str]:
    """("provider", "model") from "provider/model", or the project's author
    model on the forge-run provider when no provider is given."""
    model = model or cfg("opencode", "model", default=None) or cfg("model", "author", default=None)
    if not model:
        raise ValueError("no model: pass --models, or set [opencode] model or [model] author")
    if "/" in model and not model.startswith("hf.co/"):
        provider, name = model.split("/", 1)
        return provider, name
    return "forge", model


def config_for(model_name: str) -> dict:
    """OpenCode's whole config for one item, passed in OPENCODE_CONFIG_CONTENT
    so the user's own opencode.json is never read or changed."""
    return {
        "share": "disabled",
        "autoupdate": False,
        "enabled_providers": ["forge"],
        "model": f"forge/{model_name}",
        "provider": {"forge": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "forge",
            "options": {"baseURL": cfg("opencode", "base_url", default=DEFAULT_BASE_URL)},
            "models": {model_name: {"name": model_name}},
        }},
        # The sandbox is the boundary; inside it the agent may edit and run
        # anything. It may not fetch the web or push.
        "permission": {"edit": "allow", "webfetch": "deny",
                       "bash": {"*": "allow", "git push*": "deny", "sudo *": "deny"}},
    }


def build_prompt(task: Task) -> str:
    allowed = []
    if task.files:
        allowed.append("these file(s): " + ", ".join(task.files))
    if task.scope:
        allowed.append("files matching: " + ", ".join(task.scope))
    limit = ("\n\nOnly modify " + "; and ".join(allowed)
             + ". Read any other file you need for context, but do not change it.") if allowed else ""
    return (f"Implement this task in the current repository.{limit}\n\n"
            f"Task:\n{task.text}\n\n"
            "Make the edits directly. You may run the project's tests or other commands to check "
            "your work. Do not commit, and do not create files the task doesn't call for. When done, "
            "reply with one line summarising the change.")


def parse_events(stdout: str) -> tuple[str, dict, list[str]]:
    """(final text, usage, errors) from `opencode run --format json` output:
    one JSON event per line. Text comes from "text" parts (the last step's is
    the summary); tokens are summed over "step_finish" parts."""
    texts, errors = [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "steps": 0}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part") or {}
        if event.get("type") == "text" and part.get("text"):
            texts.append(part["text"])
        elif event.get("type") == "step_finish":
            tokens = part.get("tokens") or {}
            usage["prompt_tokens"] += int(tokens.get("input", 0)) + int((tokens.get("cache") or {}).get("read", 0))
            usage["completion_tokens"] += int(tokens.get("output", 0)) + int(tokens.get("reasoning", 0))
            usage["steps"] += 1
        elif event.get("type") == "error":
            errors.append(json.dumps(event.get("error") or event)[:500])
    return (texts[-1] if texts else ""), usage, errors


def _home_setup(home: Path) -> None:
    """The throwaway home gets OpenCode's provider packages (linked, read-only)
    and a copy of its cache, so nothing is downloaded and nothing of the
    user's own OpenCode state is visible or changed."""
    cfg_dir = home / ".config" / "opencode"
    cfg_dir.mkdir(parents=True)
    if (USER_CONFIG_DIR / "node_modules").is_dir():
        (cfg_dir / "node_modules").symlink_to(USER_CONFIG_DIR / "node_modules")
    if (USER_CONFIG_DIR / "package.json").is_file():
        shutil.copy(USER_CONFIG_DIR / "package.json", cfg_dir / "package.json")
    if USER_CACHE_DIR.is_dir():
        shutil.copytree(USER_CACHE_DIR, home / ".cache" / "opencode", symlinks=True)


class OpenCodeAgent:
    name = "opencode"
    commit_label = "opencode"
    traits = frozenset({"explores", "runs_shell", "local"})

    def attempt(self, task: Task, log) -> Attempt:
        start = time.monotonic()
        if not sandbox.available():
            return Attempt(False, "opencode runs only sandboxed, and bwrap (bubblewrap) is not installed")
        binary = opencode_binary()
        if binary is None:
            return Attempt(False, "opencode is not installed (not on PATH)")
        try:
            _, model_name = provider_model(task.model)
        except ValueError as e:
            return Attempt(False, str(e))
        env = {"OPENCODE_CONFIG_CONTENT": json.dumps(config_for(model_name)),
               "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
               "OPENCODE_DISABLE_LSP_DOWNLOAD": "1", "OPENCODE_DISABLE_CLAUDE_CODE": "1"}
        timeout = int(cfg("opencode", "timeout", default=DEFAULT_TIMEOUT))
        log(f"[opencode] {model_name}, sandboxed, in {task.worktree}")
        try:
            result = sandbox.run([str(binary), "run", "--format", "json", "--auto",
                                  "--model", f"forge/{model_name}", build_prompt(task)],
                                 task.worktree, ro=[binary.parent, USER_CONFIG_DIR / "node_modules"],
                                 env=env, home_setup=_home_setup, timeout=timeout)
        except FileNotFoundError as e:
            return Attempt(False, str(e))
        except Exception as e:  # subprocess.TimeoutExpired and friends: the attempt failed
            return Attempt(False, f"opencode did not finish: {e}")
        text, usage, errors = parse_events(result.stdout)
        usage["seconds"] = round(time.monotonic() - start, 1)
        if result.returncode != 0 or errors:
            detail = "; ".join(errors) or result.stderr.strip()[-1000:] or f"exit {result.returncode}"
            log(f"[opencode] failed: {detail}")
            return Attempt(False, detail, usage)
        log(f"[opencode] finished after {usage['steps']} step(s): {text.strip()[-300:]}")
        return Attempt(True, text, usage)
