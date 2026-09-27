"""Configuration and logging for forge."""

import datetime
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

CONFIG_FILENAMES = (".forge.toml", ".aiderloop.toml")
CONFIG_FILENAME = ".aiderloop.toml"  # retained for backward compatibility

# Filled in by load_config() at startup from the project's .aiderloop.toml.
# Defaults are deliberately inert: with no config file the loop still runs,
# it just can't check the author model's context size or run any
# project-specific test command.
CONFIG: dict = {}

DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_VALIDATION_RETRIES = 1  # separate from DEFAULT_MAX_RETRIES: this
# governs "validation failed, ask aider to fix it" retries, not "aider itself
# crashed" retries - see the validation loop in main().
# Opt-in, difficulty-gated override for DEFAULT_MAX_VALIDATION_RETRIES.
# None = disabled, every item uses the flat budget above (the historical
# behavior). When set, it applies ONLY to items estimate_difficulty()
# calls "hard" - exact-spec items are already checked byte-for-byte, so
# spending extra model time re-trying them buys nothing the byte check
# doesn't already give for free.
DEFAULT_HARD_VALIDATION_RETRIES = None
DEFAULT_MAX_ITEMS = None  # None = run until todo.md is empty of open items
DEFAULT_SLEEP_BETWEEN_ITEMS = 5  # seconds, gives you a window to Ctrl+C
DEFAULT_MAX_FILE_BYTES = 262144  # 256KB, ~85k tokens of text

# Directories to exclude from any file-discovery walk (validation, both
# Python and JS/HTML). Without this, a leftover generated artifact - e.g. a
# Plotly-bundled out/tray_interactive.html with huge inline <script> blocks -
# gets treated as source and either produces false validation failures or is
# just slow/noisy to lint for no reason; it's an output, not something aider
# wrote as part of a task.
EXCLUDE_DIRS = {
    "node_modules", "out", ".venv", "venv", "__pycache__",
    ".git", ".pytest_cache", "dist", "build",
}

# Guards log()'s own write, nothing else. With --parallel, multiple worker
# threads can log at the same moment; without this a long line from one
# thread and a long line from another can interleave mid-write. The lock
# doesn't order OUTPUT (that's fine, timestamps do that) - it just keeps
# each single write atomic so lines never merge into garbage.
_log_lock = threading.Lock()
_log_listeners = []
_stage_listeners = []
_token_listeners = []
_stop_requested = threading.Event()


def request_stop() -> None:
    _stop_requested.set()


def reset_stop() -> None:
    _stop_requested.clear()


def is_stop_requested() -> bool:
    return _stop_requested.is_set()


def add_log_listener(callback) -> None:
    with _log_lock:
        if callback not in _log_listeners:
            _log_listeners.append(callback)


def remove_log_listener(callback) -> None:
    with _log_lock:
        if callback in _log_listeners:
            _log_listeners.remove(callback)


def add_stage_listener(callback) -> None:
    with _log_lock:
        if callback not in _stage_listeners:
            _stage_listeners.append(callback)


def emit_stage(index: int, text: str, stage: str, files: list[str] | None = None) -> None:
    with _log_lock:
        listeners = list(_stage_listeners)
    for l in listeners:
        try:
            l(index, text, stage, files)
        except Exception:
            pass


def add_token_listener(callback) -> None:
    with _log_lock:
        if callback not in _token_listeners:
            _token_listeners.append(callback)


def emit_tokens(prompt_tokens: int, completion_tokens: int, seconds: float = 0.0, model: str | None = None) -> None:
    with _log_lock:
        listeners = list(_token_listeners)
    for l in listeners:
        try:
            try:
                l(prompt_tokens, completion_tokens, seconds, model)
            except TypeError:
                l(prompt_tokens, completion_tokens, seconds)
        except Exception:
            pass


def log(msg: str, log_path: Path | None = None) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with _log_lock:
        print(line, flush=True)
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        listeners = list(_log_listeners)
    for listener in listeners:
        try:
            listener(line)
        except Exception:
            pass


def load_config(project_dir: Path) -> dict:
    """Read `.forge.toml` or `.aiderloop.toml` from the project, if present.

    Everything project-specific lives here rather than in this file, which
    is what lets one copy of this script serve every project. Shape:

        [model]
        author = "qwen25-coder-aider"   # ollama name, no "ollama/" prefix
        edit_format = "diff"            # passed to aider; "whole" to override

        [validate]
        checks   = ["python", "js-html"]          # built-ins; omit to auto-detect
        commands = [".venv/bin/pytest tests -q"]  # this project's own gate

        [preflight]
        max_file_bytes = 262144   # warn about anything larger not in .aiderignore

    A missing file is not an error -- a brand-new project should be able to
    run before anyone has written config for it.
    """
    for name in CONFIG_FILENAMES:
        path = project_dir / name
        if path.is_file():
            try:
                return tomllib.loads(path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as e:
                print(f"Could not read {path}: {e}", file=sys.stderr)
                sys.exit(1)
    return {}


def cfg(*keys, default=None):
    """Nested lookup into CONFIG, e.g. cfg('model', 'author')."""
    node = CONFIG
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def check_ollama_context(log_path: Path, model_name: str | None = None) -> None:
    """
    Queries `ollama ps` before a task starts and logs the currently allocated
    context size, so a truncation risk shows up in the log even though it
    won't stop the run - Ollama truncates silently rather than erroring, so
    this is the only visibility we get short of watching live.
    """
    model_name = model_name or cfg("model", "author")
    if not model_name:
        return
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if model_name in line:
                log(f"Ollama context check before task: {line.strip()}", log_path)
                return
        log(f"Ollama context check: model {model_name} not currently loaded", log_path)
    except Exception as e:
        log(f"Ollama context check failed (non-fatal): {e}", log_path)


def resolve_models(args) -> list[str]:
    """Which model(s) to run with, overriding .aider.conf.yml's model: key.

    `--models` on the CLI wins; otherwise `[worker] models` in
    .aiderloop.toml; otherwise none at all, meaning "don't override
    anything" - the default, and the only path that existed before this
    override was added. One name changes only which model is used;
    concurrency is driven purely by how MANY names there are (see
    run_parallel), so there is no separate --parallel flag to keep in
    sync with the model list's length.
    """
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    configured = cfg("worker", "models", default=None)
    return list(configured) if configured else []
