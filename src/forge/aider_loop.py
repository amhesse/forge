#!/usr/bin/env python3
"""
aider_loop.py

Unattended looping runner for Aider. Reads a todo.md checklist, picks the
next unchecked item, has Aider (architect mode) plan and execute it, runs a
validation pass, then marks the item in todo.md before moving to the next
one. Designed to run for hours with no supervision.

Each item goes through, in order:
  1. Aider executes the task (auto-committing on success).
  2. A free, deterministic check: did the commit actually touch the file(s)
     the task's text names in backticks? If not, marked '?' (needs review)
     and the run stops there - no build/model call wasted on an item that
     plainly didn't do what was asked.
  3. Validation: Python projects get compile()-level syntax checking plus a
     fast pytest subset if tests/ exists; JS/HTML projects get node --check
     per .js/.html file, or `npm run build` if the project has one. A
     project can be either or both - see validate_syntax(). On failure,
     aider gets one (configurable) chance to fix it, re-invoked with the
     actual failure output; only after that's exhausted does the repo revert
     to the pre-item commit and the item get marked '!' (blocked). (Not
     relying on aider's own --auto-test for this - it isn't reliably applied
     in this exact headless --message mode, see validate_python()'s
     docstring.)
  Only once all of that passes does an item get marked done ('x').

Usage:
    python aider_loop.py --project-dir ~/projects/kids-maze-game
    python aider_loop.py --project-dir ~/projects/chore-tracker --max-items 1

Requires:
    - aider installed and on PATH (aider-chat)
    - a todo.md in the project dir with a markdown checklist:
        - [ ] Add walking animation for Aerie
        - [ ] Add sound effects
        - [x] Base maze game working
        - [?] Something the checks couldn't confirm - look at this one
        - [!] Something that failed and got reverted
      (a different filename is fine via --todo-file)
    - Ollama running locally with the model configured in .aider.conf.yml
"""

import argparse
import datetime
import json
import os
import tomllib
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

from . import claude_editor
from . import lite_editor
from . import worktree as wt

CHECKBOX_RE = re.compile(r"^(?P<indent>\s*)-\s\[(?P<mark>[ xX!?])\]\s(?P<text>.+)$")

STATUS_OPEN = " "
CONFIG_FILENAME = ".aiderloop.toml"

# Filled in by load_config() at startup from the project's .aiderloop.toml.
# Defaults are deliberately inert: with no config file the loop still runs,
# it just can't check the author model's context size or run any
# project-specific test command.
CONFIG: dict = {}


def load_config(project_dir: Path) -> dict:
    """Read `.aiderloop.toml` from the project, if present.

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
    path = project_dir / CONFIG_FILENAME
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Could not read {path}: {e}", file=sys.stderr)
        sys.exit(1)


def cfg(*keys, default=None):
    """Nested lookup into CONFIG, e.g. cfg('model', 'author')."""
    node = CONFIG
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


STATUS_DONE = "x"
STATUS_BLOCKED = "!"
STATUS_NEEDS_REVIEW = "?"  # aider produced *something* and it built, but an
# automated check couldn't confirm it actually satisfies the task - treated
# like done/blocked in that next_open_item() won't retry it, but the run
# stops here instead of continuing, so a human/Claude looks at it before
# anything else gets built on top of an unconfirmed change.

# File paths referenced in backticks in a task's text, e.g. "In `src/foo.jsx`,
# do X" - used to sanity-check that aider actually touched the file(s) the
# task named, rather than trusting "it built" alone.
# Any extension, as long as it starts with a letter (so `3.11` isn't a file).
# This used to be a fixed list of extensions, which was harmless while it only
# fed the touched-files check -- but restore_unnamed_files() reverts whatever
# isn't named, so an unlisted extension (a project's own `.story` files) would
# have had each task's real output deleted.
FILE_PATH_RE = re.compile(r"`([\w./-]*[\w-]\.[A-Za-z][A-Za-z0-9]{0,7})`")

# Extensions common enough to trust even on a name that looks like code.
# NOT a whitelist - an unlisted extension is still a file unless the name
# also has code shape (see _looks_like_code_reference).
_COMMON_EXTENSIONS = {
    "py", "js", "jsx", "ts", "tsx", "mjs", "cjs", "json", "md", "txt", "toml", "yaml",
    "yml", "ini", "cfg", "html", "css", "scss", "sh", "rs", "go", "java", "kt", "c",
    "h", "cpp", "hpp", "cs", "rb", "php", "lua", "sql", "xml", "csv", "lock", "env",
}


def _looks_like_code_reference(path: str) -> bool:
    """True for backticked code like `Ledger.add` or `ledger.store.Ledger`,
    which FILE_PATH_RE also matches. Found by forge calibrate: a rename
    item was parked on every trial for never "touching" `Ledger.add`, and
    a TDD item naming `ledger.store.Ledger` was rejected before any model
    ran for naming three files instead of two.

    Only slash-less names qualify, and only with a code shape: a
    capitalised stem or extension, or more than one dot. A real file
    written that way with an uncommon extension would be missed, which is
    the smaller risk than parking correct work."""
    if "/" in path:
        return False
    stem, _, ext = path.rpartition(".")
    if ext.lower() in _COMMON_EXTENSIONS and not ext[0].isupper():
        return False
    return stem[:1].isupper() or ext[:1].isupper() or "." in stem.lstrip(".")

DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_VALIDATION_RETRIES = 1  # separate from DEFAULT_MAX_RETRIES: this
# governs "validation failed, ask aider to fix it" retries, not "aider itself
# crashed" retries - see the validation loop in main().
DEFAULT_MAX_ITEMS = None  # None = run until todo.md is empty of open items
DEFAULT_SLEEP_BETWEEN_ITEMS = 5  # seconds, gives you a window to Ctrl+C

# Directories to exclude from any file-discovery walk (validation, both
# Python and JS/HTML). Without this, a leftover generated artifact - e.g. a
# Plotly-bundled out/tray_interactive.html with huge inline <script> blocks -
# gets treated as source and either produces false validation failures or is
# just slow/noisy to lint for no reason; it's an output, not something aider
# wrote as part of a task.
EXCLUDE_DIRS = {"node_modules", "out", ".venv", "venv", "__pycache__",
                 ".git", ".pytest_cache", "dist", "build"}


class TodoItem:
    def __init__(self, line_index: int, indent: str, text: str, status: str):
        self.line_index = line_index
        self.indent = indent
        # Full item text: the checkbox line's own text, plus every
        # continuation line up to the next item/checkpoint (see
        # parse_todo()) - may contain embedded newlines.
        self.text = text
        self.status = status

    def render(self) -> str:
        # Only the checkbox line itself (marker + first line of text) is
        # ever rewritten - see parse_todo()'s docstring for why any
        # continuation lines must NOT be re-emitted here: they're already
        # sitting untouched at their own positions in raw_lines, and
        # write_todo() only overwrites raw_lines[item.line_index].
        first_line = self.text.split("\n", 1)[0]
        return f"{self.indent}- [{self.status}] {first_line}"


# A checkpoint marker (an HTML comment, not a checkbox) ends the preceding
# item's continuation-line collection in parse_todo() the same way the next
# checkbox does.
_CHECKPOINT_RE = re.compile(r"^\s*<!--")


def parse_todo(todo_path: Path) -> tuple[list[str], list[TodoItem]]:
    """Returns (raw_lines, items). raw_lines is the full file split by line,
    items references positions within raw_lines so we can rewrite in place.

    An item's `.text` is not just its checkbox line. Checklists in this
    project routinely spell out a file's exact target content as a fenced
    code block spanning many lines after the checkbox - and until this fix,
    CHECKBOX_RE was matched line-by-line with `.` (which never crosses a
    newline), so every one of those continuation lines was silently
    dropped from item.text, and therefore from the actual prompt sent to
    aider (see run_aider_on_item()): the model was handed "replace `file`
    with exactly:" and nothing after the colon. Found by writing
    extract_exact_content_specs() and discovering it could never find the
    fenced block it expected in item.text at all, on tasks this script had
    itself just run - not a model problem, a parsing one, and one that
    plausibly explains failures blamed on the model throughout this
    project's aider-loop history whenever a task's spec spanned more than
    one line.

    An item's text now runs from its checkbox line up to (but not
    including) the next checkbox line, a checkpoint comment, or end of
    file, with trailing blank lines trimmed."""
    raw_lines = todo_path.read_text(encoding="utf-8").splitlines()
    return raw_lines, parse_todo_lines(raw_lines)


def parse_todo_lines(raw_lines: list[str]) -> list[TodoItem]:
    """The line-splitting half of parse_todo(), usable on text that hasn't
    (or shouldn't) touch disk - spec_compiler.py runs this on a model's
    draft output before any of it is written anywhere, so a malformed
    draft is caught by the exact same regex that will later govern how
    this same text is read back as real checklist items."""
    items = []
    n = len(raw_lines)
    i = 0
    while i < n:
        m = CHECKBOX_RE.match(raw_lines[i])
        if not m:
            i += 1
            continue
        start = i
        text_lines = [m.group("text")]
        j = i + 1
        while j < n and not CHECKBOX_RE.match(raw_lines[j]) and not _CHECKPOINT_RE.match(raw_lines[j]):
            text_lines.append(raw_lines[j])
            j += 1
        while text_lines and not text_lines[-1].strip():
            text_lines.pop()
        items.append(TodoItem(start, m.group("indent"), "\n".join(text_lines), m.group("mark")))
        i = j
    return items


def write_todo(todo_path: Path, raw_lines: list[str], items: list[TodoItem]) -> None:
    for item in items:
        raw_lines[item.line_index] = item.render()
    todo_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")


def next_open_item(items: list[TodoItem]) -> TodoItem | None:
    for item in items:
        if item.status == STATUS_OPEN:
            return item
    return None


# Guards log()'s own write, nothing else. With --parallel, multiple worker
# threads can log at the same moment; without this a long line from one
# thread and a long line from another can interleave mid-write. The lock
# doesn't order OUTPUT (that's fine, timestamps do that) - it just keeps
# each single write atomic so lines never merge into garbage.
_log_lock = threading.Lock()


def log(msg: str, log_path: Path | None = None) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with _log_lock:
        print(line, flush=True)
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


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


def run_aider_on_item(project_dir: Path, item_text: str, log_path: Path,
                      files: list[str] | None = None, model: str | None = None, backend_override: str | None = None) -> tuple[bool, str]:
    """
    Runs Aider once in architect mode with a message instructing it to plan
    and implement the given todo item (or, when called from the validation
    retry loop, a follow-up fix prompt - see main()). Aider picks up model/
    editor-model/architect settings from .aider.conf.yml in project_dir by
    default, so a project with a single worker still doesn't need this
    script to know its model name at all.

    `model` overrides that, on the CLI (which takes precedence over the
    conf file) rather than by touching .aider.conf.yml - needed for
    --parallel, where each concurrent worker is deliberately a different
    ollama model (see run_parallel()'s docstring for why: two workers
    sharing one model would just queue behind each other's GPU time
    unless ollama's own concurrency is configured, which this script
    doesn't assume).

    Returns (success, output) where success reflects whether the aider
    process exited cleanly (not whether the change is correct - that's the
    validation step's job).
    """
    check_ollama_context(log_path, model_name=model)

    prompt = (
        f"Work on this task from todo.md:\n\n{item_text}\n\n"
        "Plan the change, then implement it. Keep the change scoped to this "
        "task only - don't start on unrelated todo.md items. If the task "
        "depends on something not yet in the codebase, do your best with "
        "reasonable assumptions and note the assumption in a code comment."
    )

    cmd = [
        "aider",
        "--yes-always",       # don't prompt for confirmation, this is unattended
        "--no-gitignore",     # otherwise aider re-adds a blanket ".aider*" line to
                              # .gitignore on every run, which silently hides
                              # .aider.conf.yml/.aider.model.settings.yml from git
        "--stream",           # stream tokens live so you can watch progress
        "--no-restore-chat-history",  # each task starts with a clean context,
                                       # not the accumulated history of prior tasks
        # Central default, overridable per project in .aiderloop.toml. diff
        # (SEARCH/REPLACE) replaced whole-file output after a measured run on
        # qwen2.5-coder:14b where `whole` repeatedly dropped the filename line
        # before a file block, so aider applied nothing -- and a whole-file
        # edit costs two copies of the file out of the context window.
        "--edit-format", cfg("model", "edit_format", default="diff"),
        # Overrides .aider.conf.yml's model: key when given - see the
        # docstring above. aider takes the last --model wins, and CLI
        # flags win over the conf file regardless of order, so this is
        # safe to always include when `model` is set.
        *(["--model", f"ollama/{model}"] if model else []),
        # The files the item named, handed over as editable up front.
        # Without this, aider opens with "which files should I add?" and a
        # thinking model answers it in prose -- measured on a real run:
        # qwen3.8:27b spent four minutes and three reflections explaining
        # which files it *would* add ("stories/index.json, if it is not
        # already editable in the chat"), emitted no edit block at all, and
        # aider stopped with "Only 3 reflections allowed". 12k tokens sent,
        # 1.6k received, nothing written. The loop already parses these
        # paths for the touched-files check; it just wasn't telling aider.
        *[arg for f in (files or []) for arg in ("--file", f)],
        "--message", prompt,
    ]

    log(f"Running aider for: {item_text}", log_path)

    context_warning_logged = False
    output_lines = []
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )
        start_time = time.time()
        timeout_seconds = 60 * 30  # 30 min hard ceiling per item - adjust if needed

        for line in process.stdout:
            print(line, end="", flush=True)  # live to terminal
            output_lines.append(line)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)

            # Aider surfaces context/token warnings in its own output when it
            # thinks it's approaching a model's limit - watch for that text
            # and flag it loudly rather than letting it scroll past silently.
            lowered = line.lower()
            if not context_warning_logged and (
                "context window" in lowered
                or "context length" in lowered
                or "exceeds" in lowered and "token" in lowered
                or "trimming" in lowered
            ):
                context_warning_logged = True
                log(f"POSSIBLE CONTEXT LIMIT ISSUE on '{item_text}': {line.strip()}", log_path)

            if time.time() - start_time > timeout_seconds:
                process.kill()
                log(f"Aider timed out on: {item_text}", log_path)
                return False, "".join(output_lines)

        process.wait()

    except Exception as e:
        log(f"Aider failed to run on: {item_text} ({e})", log_path)
        return False, "".join(output_lines)

    output = "".join(output_lines)
    if process.returncode != 0:
        log(f"Aider exited with code {process.returncode} on: {item_text}", log_path)
        return False, output

    return True, output


# One aider stdout line, possibly several per call (aider makes its own
# internal retries/reflections): "Tokens: 3.6k sent, 5.0k received."
_AIDER_TOKENS_RE = re.compile(
    r"Tokens:\s*([\d.]+)(k?)\s*sent,\s*([\d.]+)(k?)\s*received", re.IGNORECASE)


def parse_aider_token_line(output: str) -> dict:
    """Best-effort token usage for the aider backend, summed across every
    such line in one call's output. This is aider's own self-reported
    figure from stdout, not a raw API response the way lite_editor's
    last_usage() is - aider is a subprocess whose API traffic this script
    never sees, so its own summary is the only source available. No
    wall-clock figure comes with it, unlike lite_editor's."""
    prompt = completion = 0.0
    for sent, sent_k, recv, recv_k in _AIDER_TOKENS_RE.findall(output):
        prompt += float(sent) * (1000 if sent_k else 1)
        completion += float(recv) * (1000 if recv_k else 1)
    return {"prompt_tokens": int(prompt), "completion_tokens": int(completion), "seconds": 0.0}


def _fallback_label(args) -> str:
    backend = getattr(args, "fallback_backend", "aider")
    name = getattr(args, "fallback_model", None) or "default model"
    return f"{backend} ({name})" if backend == "claude" else f"fallback model {name}"


def _should_fall_back(args, model: str | None) -> bool:
    if getattr(args, "fallback_backend", None) == "claude":
        return True
    return bool(getattr(args, "fallback_model", None)) and args.fallback_model != model


def run_editor_on_item(project_dir: Path, item_text: str, log_path: Path,
                       files: list[str] | None = None, model: str | None = None, backend_override: str | None = None
                       ) -> tuple[bool, str, dict]:
    """Dispatches to whichever editing backend the project asked for.

    `[model] backend = "lite"` in .aiderloop.toml (or --backend lite)
    swaps aider for lite_editor, which is built for exactly this loop's
    usage and structurally can't hit three failure classes aider produced
    on this project's real runs - see lite_editor's module docstring.
    Defaults to "aider", so a project that says nothing keeps the
    behaviour it already had.

    lite_editor always needs an explicit model: it has no
    .aider.conf.yml to fall back on the way aider does, so a run without
    --models falls back to `[model] author`, which projects using this
    loop already set for the context check.

    Returns (success, output, usage) - usage is real token counts from
    Ollama's own response for `lite`, a parse of aider's own printed
    summary for `aider`. "Cost" here means tokens spent, not money: these
    are local models with no per-token bill, so the point of tracking
    this is knowing where compute went (which items were expensive,
    whether a retry was worth it), not a dollar figure.
    """
    backend = backend_override if backend_override else cfg("model", "backend", default="aider")
    if backend == "claude":
        success, output = claude_editor.run_claude_on_item(
            project_dir, item_text, lambda m: log(m, log_path), files=files, model=model)
        return success, output, claude_editor.last_usage()
    if backend == "lite":
        # Default (lite_editor.DEFAULT_NUM_PREDICT, 8000) is sized for
        # editing existing files, where a genuinely large rewrite is
        # plausible. A project whose items generate NEW, bounded-length
        # content (measured: real successes here used 400-900 completion
        # tokens) should set this lower - a runaway generation that never
        # reaches a natural stop then fails in ~15s instead of burning
        # the full budget. Measured directly: one item took 140s and
        # 16,000 tokens (two attempts at the 8000 cap) before parking,
        # having produced nothing usable either time.
        kwargs = {}
        num_predict = cfg("model", "lite_num_predict", default=None)
        if num_predict:
            kwargs["num_predict"] = num_predict
        success, output = lite_editor.run_lite_on_item(
            project_dir, item_text, log_path, files=files,
            model=model or cfg("model", "author"), **kwargs)
        return success, output, lite_editor.last_usage()
    success, output = run_aider_on_item(project_dir, item_text, log_path, files=files, model=model)
    return success, output, parse_aider_token_line(output)


def _accumulate_usage(record: dict, usage: dict) -> None:
    """Adds one call's usage into an item's running total. Called after
    every editor call, including ones that end in failure - a parked
    item's tokens were still real compute spent, and are worth knowing
    about (an item that burned three retries' worth of tokens and still
    parked is a different kind of expensive than one that failed fast)."""
    totals = record.setdefault("tokens", {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})
    totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
    totals["completion_tokens"] += usage.get("completion_tokens", 0)
    totals["seconds"] += usage.get("seconds", 0.0)


def find_js_html_files(project_dir: Path) -> list[Path]:
    exts = {".js", ".html", ".mjs"}
    return [p for p in project_dir.rglob("*")
            if p.suffix in exts and not (EXCLUDE_DIRS & set(p.parts))]


def find_python_files(project_dir: Path) -> list[Path]:
    return [p for p in project_dir.rglob("*.py") if not (EXCLUDE_DIRS & set(p.parts))]


def run_configured_commands(project_dir: Path, log_path: Path) -> tuple[bool, str]:
    """Run whatever `[validate] commands` the project declared, in order.

    This replaces the hardcoded pytest invocation that used to live here as
    FAST_TEST_ARGS. A test suite is a project-wide gate and SHOULD run in
    full -- unlike the syntax checks below, which are scoped to the files an
    item actually touched. Failing tests in an untouched module are real
    signal that the change broke something; a syntax error in an untouched
    file is not.

    Commands run with cwd=project_dir through the shell, so a project can
    write ".venv/bin/pytest tests/fast -q" or "npm test --silent" without
    this script knowing anything about either.
    """
    commands = cfg("validate", "commands", default=[]) or []
    problems = []
    for command in commands:
        try:
            result = subprocess.run(command, shell=True, cwd=str(project_dir),
                                    capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as e:
            # A misconfigured or hanging command should degrade this check,
            # not kill an unattended run.
            log(f"validate command failed to run (skipping): {command}: {e}", log_path)
            continue
        if result.returncode != 0:
            output = ((result.stdout or "") + (result.stderr or "")).strip()
            problems.append(f"$ {command}\n{output[-4000:]}")
    if problems:
        msg = "Project validate command(s) failed:\n\n" + "\n\n".join(problems)
        log(msg[:2000], log_path)
        return False, msg
    if commands:
        log(f"Validation passed ({len(commands)} project command(s))", log_path)
    return True, "ok"


def validate_python(project_dir: Path, log_path: Path,
                    scope: list[Path] | None = None) -> tuple[bool, str]:
    """compile() every .py file in `scope` (default: all of them).

    `scope` is the list of files the item actually changed. Checking only
    those is the important part, not an optimisation: a syntax check that
    walks the whole repo will fail an item for a problem in a file it never
    opened, and the loop then feeds that failure back as "fix the failure
    above while still completing the original task" -- pointing a whole-file
    edit model at unrelated code. That is not hypothetical; it happened, and
    it is why this takes a scope argument at all.

    Deliberately NOT relying on aider's own --auto-test/--auto-lint: that is
    not reliably applied in the headless `--message` + `--yes-always` mode
    this script uses (see Aider-AI/aider#4923). This is the safety net, not
    a backstop for one that might not be running.
    """
    files = [f for f in (scope if scope is not None else find_python_files(project_dir))
             if f.suffix == ".py" and f.is_file()]
    problems = []
    for f in files:
        try:
            compile(f.read_text(encoding="utf-8"), str(f), "exec")
        except SyntaxError as e:
            problems.append(f"{f}:{e.lineno}: {e.msg}")
        except OSError:
            continue

    if problems:
        msg = "Python syntax errors:\n" + "\n".join(problems)
        log(msg, log_path)
        return False, msg

    log(f"Validation passed (python syntax, {len(files)} file(s))", log_path)
    return True, "ok"


# Server-side template syntax that is NOT JavaScript and must be removed
# before an inline <script> body can be handed to `node --check`. Without
# this, every Jinja template in a Flask project fails validation on its
# first `{{ ... }}`, which has nothing to do with whatever the model just
# edited: a one-line change to a .py file was reported as a SyntaxError in
# running.html, and the loop then asked the model to "fix" a template it had
# never touched. Stripping rather than skipping keeps real coverage -- a
# genuine JS error in a template's inline script is still caught.
_JINJA_EXPR_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)      # {{ value }} -> a literal
_JINJA_STMT_RE = re.compile(r"\{%.*?%\}", re.DOTALL)          # {% if %}    -> nothing
_JINJA_COMMENT_RE = re.compile(r"\{#.*?#\}", re.DOTALL)       # {# note #}  -> nothing


def strip_template_syntax(script_body: str) -> str:
    """Replace Jinja/Django expressions with a JS literal and drop control
    tags, so what's left is checkable JavaScript.

    `{{ ... }}` becomes `null` because it almost always appears in value
    position (`var runId = {{ run_id | tojson }};`), where deleting it would
    create a syntax error of our own making. Control tags are deleted
    outright: the statements they wrap are themselves ordinary JS and stay
    valid without the surrounding tag.
    """
    out = _JINJA_COMMENT_RE.sub("", script_body)
    out = _JINJA_STMT_RE.sub("", out)
    return _JINJA_EXPR_RE.sub("null", out)


def validate_js_html(project_dir: Path, log_path: Path,
                     scope: list[Path] | None = None) -> tuple[bool, str]:
    """
    Syntax/lint-level validation only, per your call - this does NOT run the
    game or check gameplay correctness, just confirms nothing is outright
    broken (JS parse errors, obviously malformed HTML).

    If project_dir has a package.json with a "build" script, that build is
    run instead of the per-file checks below - node --check can't parse JSX,
    so a project full of .jsx files would otherwise get zero real validation
    (every file silently skipped) while still being reported as "passed".
    Running the actual build catches JSX syntax errors correctly and is the
    only way to validate this project.

    Otherwise, prefers node --check for .js files if node is available.
    Falls back to a lightweight HTML sanity check (balanced <script> tags,
    no obvious truncation) since a full HTML validator isn't worth the
    dependency here.
    """
    package_json = project_dir / "package.json"
    # Only run a full build when the project asked for it; "package.json
    # exists" is not consent to run npm on every item.
    if "npm-build" in (cfg("validate", "checks", default=[]) or []) and package_json.is_file():
        try:
            has_build_script = "build" in json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
        except Exception:
            has_build_script = False
        if has_build_script:
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=str(project_dir), capture_output=True, text=True,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0:
                msg = f"npm run build failed:\n{output}"
                log(msg, log_path)
                return False, msg
            log("Validation passed (npm run build)", log_path)
            return True, "ok"

    problems = []

    node_available = subprocess.run(
        ["which", "node"], capture_output=True, text=True
    ).returncode == 0

    candidates = scope if scope is not None else find_js_html_files(project_dir)
    for f in [c for c in candidates if c.suffix in (".js", ".mjs", ".html") and c.is_file()]:
        if f.suffix in (".js", ".mjs"):
            if node_available:
                result = subprocess.run(
                    ["node", "--check", str(f)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    problems.append(f"{f}: {result.stderr.strip()}")
            # if node isn't available, we silently skip JS syntax checking
            # rather than failing the whole run over a missing tool
        elif f.suffix == ".html":
            content = f.read_text(encoding="utf-8", errors="replace")
            if content.count("<script") != content.count("</script>"):
                problems.append(f"{f}: mismatched <script> tags")
            # extract inline <script> blocks and check those too, if node available
            if node_available:
                for match in re.finditer(
                    r"<script(?:\s[^>]*)?>(.*?)</script>", content, re.DOTALL
                ):
                    script_body = strip_template_syntax(match.group(1))
                    if not script_body.strip():
                        continue
                    # Written to the system temp dir, not project_dir: a stray
                    # .js file inside the repo is something find_js_html_files
                    # would then try to validate, and something
                    # suspicious_new_paths would reasonably flag as the model
                    # having invented a file.
                    fd, tmp_name = tempfile.mkstemp(prefix="aider_inline_", suffix=".js")
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as fh:
                            fh.write(script_body)
                        result = subprocess.run(
                            ["node", "--check", tmp_name],
                            capture_output=True, text=True,
                        )
                    finally:
                        try:
                            os.unlink(tmp_name)
                        except OSError:
                            pass
                    if result.returncode != 0:
                        problems.append(f"{f} (inline script): {result.stderr.strip()}")

    if problems:
        msg = "Validation found issues:\n" + "\n".join(problems)
        log(msg, log_path)
        return False, msg

    log("Validation passed (syntax-level check only)", log_path)
    return True, "ok"


def validate_syntax(project_dir: Path, log_path: Path,
                    scope: list[Path] | None = None) -> tuple[bool, str]:
    """Run the built-in syntax checkers that apply, over `scope` only.

    Which checkers run comes from `[validate] checks` in .aiderloop.toml
    ("python", "js-html", "npm-build"). With no config, both syntax
    checkers are enabled and each no-ops when the changed set contains no
    files it understands -- so a brand-new project gets sensible behaviour
    before anyone writes config, and a project that wants something
    narrower can say so.

    `scope` is the set of files the item changed. See validate_python's
    docstring for why that scoping is load-bearing rather than an
    optimisation.
    """
    results = check_syntax_only(project_dir, log_path, scope)
    results.append(run_configured_commands(project_dir, log_path))

    ok = all(r[0] for r in results)
    combined = "\n\n".join(r[1] for r in results if r[1] and r[1] != "ok")
    return ok, combined or "ok"


def check_syntax_only(project_dir: Path, log_path: Path,
                      scope: list[Path] | None = None) -> list[tuple[bool, str]]:
    """The syntax-checker half of validate_syntax(), without the project's
    own `[validate] commands`.

    Split out because those two things answer different questions: "is
    this file well-formed" and "does the project's test suite pass" are
    not the same check, and TDD's red phase needs only the first - a
    crash in the new test isn't "a failing test" and shouldn't be treated
    as one, but the project's test command failing IS exactly what red
    phase expects and must not be mistaken for a syntax problem either.
    Calling validate_syntax() itself for this would run the project's
    commands a phase early and misreport the result: measured directly,
    the red-phase test importing a not-yet-written module made the
    project's own `commands` gate fail (correctly - that's the point of
    red phase), and validate_syntax() folded that into "syntax error",
    which it wasn't.
    """
    checks = cfg("validate", "checks", default=None)
    if checks is None:
        checks = ["python", "js-html"]

    results = []
    if "python" in checks:
        results.append(validate_python(project_dir, log_path, scope))
    if "js-html" in checks or "npm-build" in checks:
        results.append(validate_js_html(project_dir, log_path, scope))
    return results


def package_json_or_js_html_present(project_dir: Path) -> bool:
    return (project_dir / "package.json").is_file() or bool(find_js_html_files(project_dir))


DEFAULT_MAX_FILE_BYTES = 262144   # 256KB, ~85k tokens of text



def preflight_todo_rules(todo_path: Path, log_path: Path) -> None:
    """Aborts the whole run only over a FATAL shape problem - one where a
    load-bearing check (touched-files, byte-exact) would actually be
    disabled, or the item is concretely wrong (a hallucinated anchor).
    Everything validate_item() flags is still logged, including advisory
    ones (like naming 3+ files), because they're real signal for a human
    to read - just not signal that should block a legitimate item from
    running. See fatal_problems()'s docstring for the run this
    distinction fixed: a correctly 3-file item (rename_across_files,
    add_currency_field in bench/) was rejected on every single
    calibration trial, for both models, before either ever got a chance
    to run - 0 seconds, every time, not a model failure at all."""
    from forge.spec_compiler import validate_item, fatal_problems
    _, items = parse_todo(todo_path)
    open_items = [it for it in items if it.status == STATUS_OPEN]
    if not open_items:
        return

    any_problems = False
    any_fatal = False
    for i, item in enumerate(open_items, 1):
        problems = validate_item(item, todo_path.parent)
        if problems:
            if not any_problems:
                log("=== Preflight Checklist Validation ===", log_path)
                any_problems = True
            log(f"Item {i} ({item.text.splitlines()[0][:60]}...):", log_path)
            for p in problems:
                log(f"  ⚠ {p}", log_path)
        if fatal_problems(item, todo_path.parent):
            any_fatal = True

    if any_fatal:
        log("ERROR: One or more open items violate the formatting rules in the README "
            "in a way that disables a real check (missing backtick file names, an "
            "unverified fenced block, or a hallucinated anchor line).", log_path)
        log("The loop relies on these rules to safely scope edits.", log_path)
        log("Please fix the items in your checklist or remove them before running.", log_path)
        import sys
        sys.exit(1)
    elif any_problems:
        log("(the warning(s) above are advisory - the run is proceeding)", log_path)


def preflight_repo_size(project_dir: Path, log_path: Path) -> list[str]:
    """Warn about files big enough to wreck aider's context window.

    Aider builds its repo map by walking the working tree, so a few
    committed data files can dominate everything. Measured on a real
    project: ~3MB of golden CSV fixtures and a 6MB generated HTML report
    produced an estimated chat context of 1,584,443 tokens against a 32,768
    limit, for a task that touched one template -- and killed two runs by
    exhausting system memory before anyone understood why.

    The fix is a `.aiderignore` listing that data. This check exists so the
    next project finds that out in the first ten seconds instead of after
    two OOM kills, so it names the offenders and the file to put them in.
    Warn-only: a big file is a strong smell, not proof of a problem, and
    refusing to start would be the wrong call on a project that genuinely
    needs one.
    """
    limit = cfg("preflight", "max_file_bytes", default=DEFAULT_MAX_FILE_BYTES)
    ignore_path = project_dir / ".aiderignore"
    ignored = []
    if ignore_path.is_file():
        ignored = [l.strip() for l in ignore_path.read_text(encoding="utf-8").splitlines()
                   if l.strip() and not l.strip().startswith("#")]

    def is_ignored(rel: str) -> bool:
        import fnmatch
        return any(fnmatch.fnmatch(rel, pat) or rel.startswith(pat.rstrip("/") + "/")
                   for pat in ignored)

    big = []
    for f in project_dir.rglob("*"):
        if not f.is_file() or (EXCLUDE_DIRS & set(f.parts)) or ".git" in f.parts:
            continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size <= limit:
            continue
        rel = str(f.relative_to(project_dir))
        if not is_ignored(rel):
            big.append(f"{rel} ({size // 1024}KB)")

    if big:
        log(f"Preflight: {len(big)} file(s) over {limit // 1024}KB are visible to aider "
            f"and not in .aiderignore. These inflate every prompt and can exhaust the "
            f"context window:\n  " + "\n  ".join(sorted(big)[:15]), log_path)
    return big


def is_git_repo(project_dir: Path) -> bool:
    return (project_dir / ".git").exists()


def git_head(project_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(project_dir),
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


# NOTE: git_revert_to() used to live here, hard-resetting the project
# checkout after a failed item. Per-item worktrees removed the need for it:
# a failed item's commits are made on a throwaway branch and simply never
# merged, so there is nothing in the project to undo. Reverting was always
# the risky half of the design -- it wrote to the tree the user was working
# in, and it discarded the evidence.


# Phrases that, when they appear shortly after a backtick-quoted file path,
# mean that file was named as a "don't touch this" reference (e.g. "...inside
# `src/foo.jsx`, which is unrelated and should not be touched") rather than a
# file the task expects the commit to actually change. Found by testing this
# exact check against a real task and getting a false positive on exactly
# this phrasing.
# The (?!...) after "touch" stops "do not touch any other file"/"...other
# files" from matching: that phrase means "nothing ELSE should be edited",
# which is the opposite of negating the file just named - it showed up as
# this project's own boilerplate scoping instruction ("Do not touch any
# other file.") on every exact-content-spec task, and without this
# exclusion it silently negated that same task's real, single expected
# file, making expected_files() return empty for every one of them. Found
# by writing the exact-content-spec check and getting an empty result on a
# real task's real target file.
NEGATION_NEAR_RE = re.compile(
    r"should not be touched|do not touch(?!\s+(?:any\s+)?other)|not be touched|"
    r"should not touch(?!\s+(?:any\s+)?other)|"
    # "change" is a much more generic verb than "touch" in practice - real
    # items use "do not change any existing test" as a plain disclaimer
    # with no specific file in view. A backtick-lookahead here doesn't
    # work as a fix (tried first, reverted): _negates_before's `before`
    # slice deliberately stops right BEFORE a path's own backtick, so a
    # lookahead requiring one immediately after can never match there -
    # that's the whole call site this exists for. Same exclusion shape as
    # "touch" instead, widened to the two boilerplate phrasings actually
    # seen ("any other ..." / "any existing ...").
    r"do not change(?!\s+(?:any\s+)?(?:other|existing))|"
    r"should not change(?!\s+(?:any\s+)?(?:other|existing))|"
    r"leave\b.{0,40}\bunchanged|which is unrelated",
    re.IGNORECASE,
)

# NOTE ON HISTORY: expected_files() used to also require an edit-action verb
# (create/add/update/...) to precede a backtick-quoted path before counting
# it as an edit target, meant to stop a task that merely *talks about* a
# file in prose (e.g. a README task describing what `aider_loop.py` does)
# from being wrongly flagged. That allowlist-of-verbs approach turned out
# to be the more dangerous failure mode in practice: it silently missed
# real edit targets twice on two different real task phrasings ("Extend
# `file.py` with..." and "Replace the entire contents of `file.py`
# with..." - neither verb was in the list), and both times the missing
# entry meant expected_files() returned an EMPTY list, which the caller
# (see the loop in main()) treats as "nothing to check" rather than
# "check found nothing" - so a real no-op, and once actual corrupted
# content, both sailed straight through validate_syntax() (which trivially
# passes when nothing meaningful changed) to "Marking done", undetected
# until a human happened to read the file. An occasional false "needs-
# review" stop (the failure mode a verb-allowlist was protecting against)
# is a nuisance a person can immediately resolve by looking; a silent false
# "done" is not - it ships wrong content with nothing left visibly wrong.
# Given that asymmetry, the verb requirement is gone: every backtick-quoted
# path with a recognized extension counts as an expected edit target unless
# NEGATION_NEAR_RE says otherwise, full stop.


def path_matches_any(expected_path: str, touched_paths: list[str]) -> bool:
    """
    True if expected_path refers to the same file as one of touched_paths,
    allowing for the same file being written two different ways (a full
    repo-relative path like "src/App.jsx" vs. a bare filename like "App.jsx"
    used as shorthand later in the same task's text) - git diff always
    reports the full path, so a literal-string comparison would otherwise
    flag a correctly-touched file as missing. Found via a real task that did
    exactly this in its own wording.
    """
    for touched in touched_paths:
        if expected_path == touched:
            return True
        if touched.endswith("/" + expected_path) or expected_path.endswith("/" + touched):
            return True
    return False


def expected_files(item_text: str) -> list[str]:
    """Backtick-quoted file paths mentioned in the task text (outside any
    fenced code block) - the files we expect this item's commit to actually
    touch. A path counts unless a "don't touch this" phrase follows it
    nearby (NEGATION_NEAR_RE) - see the long comment above NEGATION_NEAR_RE's
    old sibling regex for why this isn't also gated on an edit-action verb
    preceding it.

    Fenced blocks are stripped before scanning: since parse_todo() started
    capturing an item's full multi-line text (see its docstring) rather
    than just the checkbox line, a task that spells out a whole target
    file's content inline - which is most of them, in this project - has
    that content's OWN file references show up in item_text too (e.g. a
    README documenting "site.yaml" and "fabric.yaml" in its own prose).
    Those aren't edit targets, they're the target file's content; only
    prose outside any fence names a real one ("In `results.html`, add...").
    extract_exact_content_specs() is what actually verifies fenced content,
    separately and more precisely than this function ever could."""
    prose_only = _FENCE_RE.sub("", item_text)
    expected = []
    for m in FILE_PATH_RE.finditer(prose_only):
        after = prose_only[m.end():m.end() + 120]
        before = prose_only[max(0, m.start() - 120):m.start()]
        if _looks_like_code_reference(m.group(1)):
            continue
        if _negates_before(before) or _negates(m.group(1), after):
            continue
        expected.append(m.group(1))
    return sorted(set(expected))


def _negates_before(before: str) -> bool:
    """True if a "don't touch this" phrase sits immediately BEFORE a path,
    with no other path in between - i.e. this path is the one it refers to.

    _negates() below only ever looked at text AFTER a path, which meant it
    caught "`bar.py`, which is unrelated" but silently missed the far more
    natural "do not touch `bar.py`" whenever that path was the last one in
    the item. Measured cost of that gap: on archaeologist's first full
    run, twelve of twenty parked items were items whose own text ended
    "...and do not touch `arch/chunk.py`." The loop dutifully required the
    commit to touch the very file the item forbade touching, and parked
    every one of them for not doing so.
    """
    last = None
    for m in NEGATION_NEAR_RE.finditer(before):
        last = m  # nearest phrase to this path wins
    if last is None:
        return False
    # Only this path is negated - if another path appears between the
    # phrase and here, that one was its target, not this one.
    return not FILE_PATH_RE.search(before[last.end():])


def _negates(path: str, after: str) -> bool:
    """True if `after` (the text following one mention of `path`) contains
    a negation phrase that is actually about `path`, not some other file
    also named nearby. NEGATION_NEAR_RE's phrases point in different
    directions - "do not touch `other.py`" names its file AFTER the
    phrase, "`other.py`, which ... should not be touched" names it
    BEFORE - so which file a match belongs to has to be resolved by
    proximity to the phrase itself, not just by finding some negation
    phrase anywhere in the window. Without this, a task like "In `foo.py`,
    add X. Do not touch `bar.py`, which is unrelated." would negate foo.py
    (an earlier, different file) purely for sharing a window with bar.py's
    own, correctly-targeted negation."""
    neg = NEGATION_NEAR_RE.search(after)
    if not neg:
        return False
    before_neg = after[:neg.start()]
    after_neg = after[neg.end():neg.end() + 60]
    # Forward-referring ("do not touch `other.py`"): whatever's named
    # right after the phrase is what's negated, if anything is.
    forward = FILE_PATH_RE.search(after_neg)
    if forward:
        return forward.group(1) == path
    # Backward-referring ("`other.py`, which ... should not be touched"):
    # only relevant if some path was actually named between this mention
    # and the phrase - if one was, that's what's negated instead of path.
    backward = None
    for pm in FILE_PATH_RE.finditer(before_neg):
        backward = pm.group(1)  # last match wins - nearest to the phrase
    if backward is not None:
        return backward == path
    # Nothing named on either side: the phrase directly follows this
    # mention with nothing else in between - the original, well-tested
    # case ("...inside `src/foo.jsx`, which is unrelated and should not
    # be touched").
    return True


# Text the model has, verbatim, written into a target file as its actual
# content instead of treating it as an instruction -- caught twice on real
# tasks (once into an HTML template, once into a from-scratch markdown
# README), both times identical down to the sentence. This reads like a
# fragment of aider's own system prompt leaking through into the model's
# output. Crucially, this corruption is invisible to every other check
# here: the file genuinely was touched (so expected_files() sees nothing
# wrong), and it's often syntactically-valid prose/HTML/whatever the target
# format is (so validate_syntax() sees nothing wrong either, and for a
# non-Python file there is no syntax check at all). Only a direct content
# check catches it.
PROMPT_LEAKAGE_MARKERS = (
    "Plan the change, then implement it",
    "ONLY EVER RETURN CODE IN A SEARCH/REPLACE BLOCK",
    "I added these files to the chat",
)


def detect_prompt_leakage(project_dir: Path, touched_paths: list[str]) -> list[str]:
    """Returns the subset of touched_paths whose current on-disk content
    contains a known prompt-leakage marker verbatim."""
    corrupted = []
    for rel in touched_paths:
        path = project_dir / rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(marker in text for marker in PROMPT_LEAKAGE_MARKERS):
            corrupted.append(rel)
    return corrupted


# Characters that are illegal in a filename on Windows/NTFS (< > : " | ? *),
# found because aider has, twice on real tasks, created a new file whose
# *name* is a fragment of its own malformed SEARCH/REPLACE reasoning (once
# a plain stray file, once four files literally named things like
# "SearchReplaceNoExactMatch: This SEARCH block failed to exactly match
# lines in..."). Past the obvious clutter, several of those names contain
# these characters -- meaning a Windows user (this project's actual target
# audience, per docs/getting-started.md) could not even check out this
# repo while they existed; git itself doesn't stop you from committing
# them on Linux. A real project file never legitimately needs any of these
# characters, so their presence alone is enough signal to flag on its own,
# with no reliance on how the file's content looks.
_WINDOWS_ILLEGAL_CHARS_RE = re.compile(r'[<>:"|?*]')


def suspicious_new_paths(touched_paths: list[str]) -> list[str]:
    """Returns the subset of touched_paths that look like a garbage
    filename rather than a real one aider meant to create: a Windows-illegal
    character, or implausibly long for a single path component (a real
    source/doc filename is never this long)."""
    suspicious = []
    for rel in touched_paths:
        basename = rel.rsplit("/", 1)[-1]
        if _WINDOWS_ILLEGAL_CHARS_RE.search(rel) or len(basename) > 100:
            suspicious.append(rel)
    return suspicious


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)\n```", re.DOTALL)


def extract_exact_content_specs(item_text: str) -> dict[str, str]:
    """Many tasks in this project's todo files spell out a file's target
    content verbatim: "...`path/to/file` with exactly this content:" or
    "...`path/to/file` with exactly:", immediately followed by a fenced
    code block. When a task is phrased that way, there's no need to guess
    whether what aider wrote matches -- it can be checked directly, byte
    for byte (mod trailing-newline normalization), rather than inferred
    from heuristics like expected_files() or detect_prompt_leakage(). This
    only covers tasks actually phrased this way; anything else still falls
    back to those.

    Returns {file_path: expected_content} for every such spec found."""
    specs: dict[str, str] = {}
    for fence in _FENCE_RE.finditer(item_text):
        before = item_text[:fence.start()].rstrip()
        if not before.endswith(":"):
            continue
        if "exactly" not in before[-80:].lower():
            continue
        path_matches = [m for m in FILE_PATH_RE.finditer(item_text[:fence.start()])
                        if not _looks_like_code_reference(m.group(1))]
        if not path_matches:
            continue
        specs[path_matches[-1].group(1)] = fence.group(1)
    return specs


def _normalize_content(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip("\n")


def content_mismatches(project_dir: Path, specs: dict[str, str]) -> list[str]:
    """Returns the subset of specs' file paths whose actual on-disk content
    doesn't match the spec (after normalizing line endings and a trailing
    newline, since a real file's own convention there is irrelevant to
    whether the meaningful content is right)."""
    mismatched = []
    for rel, expected in specs.items():
        path = project_dir / rel
        try:
            actual = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            mismatched.append(rel)  # doesn't exist at all -- also a mismatch
            continue
        if _normalize_content(actual) != _normalize_content(expected):
            mismatched.append(rel)
    return mismatched


def changed_files(project_dir: Path, pre_hash: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", pre_hash, "HEAD"],
        cwd=str(project_dir), capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# A task that names its files is also a promise about what it won't touch.
# Found on a real run: asked to create art/car-blue.txt "same shape as
# art/car-red.txt", the model also rewrote car-red.txt, then read the
# resulting test failure as car-red's fault and edited it again on every
# retry. Nothing else caught it -- the touched-files check only asks whether
# the named files changed, not whether others did. Putting unnamed files back
# before validation means the tests judge the task alone, and the retry
# feedback points at the file the model was actually asked to write.
def restore_unnamed_files(project_dir: Path, pre_hash: str, expected: list[str],
                          log_path: Path) -> list[str]:
    """Reverts every file changed since pre_hash that the task didn't name,
    commits the restoration, and returns the paths restored. Does nothing
    when the task named no files (there's no scope to enforce) or when the
    project sets [scope] restore_unnamed = false."""
    if not expected or not cfg("scope", "restore_unnamed", default=True):
        return []
    extra = [f for f in changed_files(project_dir, pre_hash)
             if not any(path_matches_any(e, [f]) for e in expected)]
    if not extra:
        return []
    for rel in extra:
        existed = subprocess.run(["git", "cat-file", "-e", f"{pre_hash}:{rel}"],
                                 cwd=str(project_dir), capture_output=True).returncode == 0
        if existed:
            subprocess.run(["git", "checkout", pre_hash, "--", rel],
                           cwd=str(project_dir), capture_output=True)
        else:
            subprocess.run(["git", "rm", "-q", "-f", "--", rel],
                           cwd=str(project_dir), capture_output=True)
    subprocess.run(["git", "add", "-A", "--", *extra], cwd=str(project_dir), capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m",
                    f"aider-loop: restore files the task didn't name: {', '.join(extra)}"],
                   cwd=str(project_dir), capture_output=True)
    log(f"Restored file(s) the task didn't name: {extra} (task named: {expected})", log_path)
    return extra


def record_run(runs_dir: Path, index: int, record: dict) -> Path:
    """Write one item's outcome to the run log directory.

    The todo file records a single character per item; that was enough
    when a run stopped at the first problem and you went straight to the
    terminal scrollback. Once the loop parks failures and keeps going, an
    overnight run ends with several parked items and no way to tell what
    happened to each without re-reading a thousand lines of interleaved
    log. This is the queue a review UI reads later: the item text, the
    branch its work is sitting on, what the checks said, and why.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"item-{index:03d}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


# A literal "TDD:" prefix opts an item into red-green-refactor instead of
# the normal single-pass flow. Not inferred from an item naming a test
# file alongside an implementation file - several real items already do
# that (see blockroad's choice-length check, done in one pass) without
# wanting strict phase separation, and that style is not wrong, just a
# different and weaker guarantee than this one buys. Opt-in keeps both
# available without one silently overriding the other.
TDD_PREFIX_RE = re.compile(r"^\s*TDD:\s*", re.IGNORECASE)


def is_tdd_item(item_text: str) -> bool:
    return bool(TDD_PREFIX_RE.match(item_text))


def classify_tdd_files(files: list[str]) -> tuple[str, str] | None:
    """Given the files a TDD item named, decides which is the test and
    which is the implementation. Requires exactly two files, exactly one
    of which looks test-shaped: a `test/` or `tests/` path component, or
    `.test.` / `_test.` / a `test_` filename prefix - covering both this
    project's own test/check.test.js convention and pytest's test_foo.py
    / foo_test.py. Returns None rather than guessing when the naming
    doesn't clearly pick a side, falling the item back to needs-review -
    a silent wrong guess here would run the wrong file through the wrong
    phase (implementing into what should have stayed a fixed target, or
    "testing" the test)."""
    if len(files) != 2:
        return None
    def looks_like_test(path: str) -> bool:
        parts = path.split("/")
        name = parts[-1]
        return (any(p in ("test", "tests") for p in parts[:-1])
                or ".test." in name or "_test." in name or name.startswith("test_"))
    test_like = [f for f in files if looks_like_test(f)]
    other = [f for f in files if not looks_like_test(f)]
    return (test_like[0], other[0]) if len(test_like) == 1 and len(other) == 1 else None


# Used by --models to route items between worker tiers - see
# estimate_difficulty()'s docstring for the reasoning, and README's
# "Parallel workers" section for how tiers map to worker slots.
DIFFICULTY_EASY = "easy"
DIFFICULTY_HARD = "hard"


def estimate_difficulty(item_text: str) -> str:
    """Classifies an item as "easy" (safe to hand to a smaller/weaker
    model) or "hard" (wants the strongest model available), for
    --models to route between worker tiers.

    This is deliberately built on the one difficulty signal this project
    actually has evidence for, not a guessed heuristic (word count,
    "sounds complicated") that would just be a second unverified guess on
    top of the first: an item with a byte-exact content spec is checked
    byte-for-byte regardless of which model writes it (see
    extract_exact_content_specs / content_mismatches), so a weaker model
    getting it wrong costs exactly one parked item, never a silent wrong
    answer - and the project's own README already names this the single
    most reliable item shape, precisely because it only asks a model to
    transcribe, not decide ("Local models transcribe well and decide
    badly; exact specs play to that."). That's "easy" here in the
    specific sense that matters for routing: low-stakes to get wrong, not
    "simple content".

    Everything else defaults to "hard", including TDD items and any item
    naming more than one file. Both need real judgment this project has
    only ever measured a 14B model make correctly - TDD's green phase
    twice needed genuine scope judgment despite ambiguous instructions
    (see run_tdd_phases' real-model test in the README), and a
    multi-file item usually exists because two files have to agree with
    each other (see "Writing items"), which is exactly the failure mode
    that cost three retries on this project's own history when a
    weaker model's coordination went wrong. There is no measurement yet
    that a smaller model handles either reliably, so both default to the
    strong tier rather than assume they do.
    """
    if is_tdd_item(item_text):
        return DIFFICULTY_HARD
    if len(expected_files(item_text)) > 1:
        return DIFFICULTY_HARD
    if extract_exact_content_specs(item_text):
        return DIFFICULTY_EASY
    return DIFFICULTY_HARD


def run_review_gate(worktree_path: Path, base: str, item_text: str, expected: list[str],
                    args, log_path: Path, record: dict, finish,
                    model: str | None = None, backend_override: str | None = None,
                    tdd: bool = False, restore_base: str | None = None):
    """Last gate before merge, after every mechanical check (including
    TDD's red/green check) has passed: a second model reads the diff for
    wrong logic the mechanical checks can't see - the "silent wrong"
    class `forge calibrate` measures, and exactly the class that let a
    TDD item merge here with `entry.amount` (the real field is `.cents`)
    before this gate was wired into the TDD path too.

    A no-op (returns None immediately) when --review wasn't given, so
    both call sites can call this unconditionally. Otherwise returns None
    to mean "approved, proceed to merge", or the (status, record) tuple
    from `finish()` if the item should be parked instead - callers must
    check for that and return it directly, the same as any other early
    exit here.

    `restore_base` is the commit a rejected review's fix attempt gets
    reverted against for any file outside `expected` - defaults to
    `base` (the diff's own start point), which is right for the normal
    path but wrong for TDD: `base` is BEFORE the red phase, so the test
    file doesn't exist there at all, and restoring "to how it was at
    base" means deleting it outright rather than putting back its
    red-phase content. Found live: a TDD review-fix that strayed into
    editing the test file got it deleted, not restored, and the item
    still parked safely (test_file_modified_since_red caught the
    resulting mismatch either way) - just more conservatively than
    necessary. The TDD call site passes the red-phase commit instead.
    """
    reviewer = getattr(args, "review", None)
    if not reviewer:
        return None
    restore_base = restore_base or base
    review_attempt = 0
    while True:
        diff = subprocess.run(["git", "-C", str(worktree_path), "diff", base, "HEAD"],
                              capture_output=True, text=True).stdout
        log(f"Asking {reviewer} to review the diff...", log_path)
        if reviewer == "claude":
            approved, feedback = claude_editor.review_diff(
                worktree_path, item_text, diff, lambda m: log(m, log_path),
                model=getattr(args, "review_model", None), tdd=tdd)
        else:
            try:
                out = subprocess.run(["agy", "--print",
                                      f"Review this diff for the task below. The VERY LAST LINE of your "
                                      f"reply must be exactly one word: APPROVE if it's correct, REJECT "
                                      f"if not (with your reasons above it).\n\nTask:\n{item_text}"
                                      f"\n\nDiff:\n{diff}"],
                                     capture_output=True, text=True, check=True).stdout.strip()
                last = out.splitlines()[-1].strip().upper() if out else ""
                approved, feedback = (last == "APPROVE" or (last != "REJECT" and out.upper().startswith("APPROVE"))), out
            except (OSError, subprocess.CalledProcessError) as e:
                approved, feedback = None, str(e)
        if approved:
            log(f"{reviewer} approved the changes.", log_path)
            record["review"] = {"reviewer": reviewer, "rounds": review_attempt + 1}
            return None
        if approved is None:
            log(f"{reviewer} review failed to run: {feedback[-500:]}. Parking needs-review.", log_path)
            return finish(STATUS_NEEDS_REVIEW, f"{reviewer} review could not run")
        review_attempt += 1
        if review_attempt > args.max_validation_retries:
            record["review_feedback"] = feedback[-4000:]
            log(f"{reviewer} still rejects after {review_attempt - 1} fix round(s). "
                f"Parking blocked: {item_text}", log_path)
            return finish(STATUS_BLOCKED, f"{reviewer} review rejected the change")
        log(f"{reviewer} rejected (round {review_attempt}/{args.max_validation_retries}):\n"
            f"{feedback[-2000:]}", log_path)
        fix_success, _, fix_usage = run_editor_on_item(worktree_path, (
            f"A reviewer rejected the change made for this task.\n\n"
            f"Original task:\n{item_text}\n\n"
            f"Reviewer feedback:\n{feedback}\n\n"
            f"Fix the problems the reviewer lists while still completing the original task."
        ), log_path, files=expected, model=model, backend_override=backend_override)
        _accumulate_usage(record, fix_usage)
        if not fix_success:
            continue
        restore_unnamed_files(worktree_path, restore_base, expected, log_path)
        wt.restore_todo(worktree_path, base, args.todo_file)
        touched = changed_files(worktree_path, base)
        record["touched_files"] = touched
        valid, validation_msg = validate_syntax(worktree_path, log_path,
                                                scope=[worktree_path / f for f in touched])
        if not valid:
            record["validation_output"] = validation_msg[-4000:]
            log(f"Review fix broke validation, parking blocked: {item_text}", log_path)
            return finish(STATUS_BLOCKED, "validation failed after review fix")


def run_tdd_phases(project_dir: Path, worktree_path: Path, task_text: str,
                   test_file: str, impl_file: str, base: str, args, log_path: Path,
                   record: dict, finish, model: str | None = None,
                   git_lock: threading.RLock | None = None, backend_override: str | None = None) -> tuple[str, dict]:
    """Red, then green: aider writes ONLY the test and that test is
    confirmed to fail for a real reason, then a separate aider call
    implements the feature without being allowed to touch the test again.

    This is the version of "the tests are the check" (see README) that
    doesn't depend on the test having been written correctly by a human
    ahead of time: the test's own honesty is verified before it's ever
    trusted to grade anything, the same way a human reviewer would want
    to see a new test fail before believing it tests the right thing.
    """
    git_lock = git_lock or threading.RLock()
    record["tdd"] = {"test_file": test_file, "impl_file": impl_file}

    # ---------------------------------------------------------------- RED
    red_prompt = (
        f"Write ONLY the test described below, in `{test_file}`. Do not create, "
        f"modify, or touch `{impl_file}` or any other file - the feature it "
        f"describes does not exist yet, so the test you write is EXPECTED to "
        f"fail. Do not write a stub or placeholder implementation anywhere to "
        f"make it pass; that defeats the point of writing the test first.\n\n{task_text}"
    )
    success, _, usage = run_editor_on_item(worktree_path, red_prompt, log_path, files=[test_file], model=model, backend_override=backend_override)
    _accumulate_usage(record, usage)
    if not success:
        return finish(STATUS_BLOCKED, "TDD red phase: aider itself failed")

    suspicious = suspicious_new_paths(changed_files(worktree_path, base))
    if suspicious:
        log(f"Garbage-looking filename(s) created in the red phase: {suspicious}. "
            f"Parking: {task_text}", log_path)
        return finish(STATUS_BLOCKED, f"TDD red phase created garbage filename(s): {suspicious}")

    restore_unnamed_files(worktree_path, base, [test_file], log_path)
    wt.restore_todo(worktree_path, base, args.todo_file)
    red_touched = changed_files(worktree_path, base)
    record["red_touched_files"] = red_touched

    if path_matches_any(impl_file, red_touched):
        log(f"Red phase touched `{impl_file}` despite being told not to. "
            f"Parking: {task_text}", log_path)
        return finish(STATUS_NEEDS_REVIEW, f"red phase touched the implementation file `{impl_file}`")
    if not path_matches_any(test_file, red_touched):
        log(f"Red phase never wrote `{test_file}`. Parking: {task_text}", log_path)
        return finish(STATUS_NEEDS_REVIEW, f"red phase never touched the test file `{test_file}`")

    corrupted = detect_prompt_leakage(worktree_path, red_touched)
    if corrupted:
        return finish(STATUS_BLOCKED, f"prompt text was written into {corrupted} during the red phase")

    # A crash isn't "a failing test" - it's no signal at all, and the
    # green phase below would have nothing real to fix. Syntax only, not
    # validate_syntax() - that also runs the project's own [validate]
    # commands, which is exactly the gate red phase is deliberately
    # failing right now; see check_syntax_only()'s docstring.
    syn_results = check_syntax_only(worktree_path, log_path, scope=[worktree_path / test_file])
    syn_ok = all(r[0] for r in syn_results)
    if not syn_ok:
        syn_msg = "\n\n".join(r[1] for r in syn_results if r[1] and r[1] != "ok")
        log(f"Red-phase test has a syntax error: {syn_msg}. Parking: {task_text}", log_path)
        return finish(STATUS_BLOCKED, "red phase test file has a syntax error")

    gate_ok, gate_msg = run_configured_commands(worktree_path, log_path)
    record["red_phase_gate_output"] = gate_msg[-2000:]
    if gate_ok:
        log(f"Red-phase test passed immediately, before any implementation was "
            f"written. Parking: {task_text}", log_path)
        return finish(STATUS_NEEDS_REVIEW,
                     "the new test passed before any implementation was written - it may "
                     "not test the described behavior, or the behavior already exists")
    log(f"Red phase confirmed: `{test_file}` fails as expected.", log_path)
    red_commit = git_head(worktree_path)

    # --------------------------------------------------------------- GREEN
    green_prompt = (
        f"The test in `{test_file}` (already written - do not modify it, or touch "
        f"any file other than `{impl_file}`) currently fails because the feature "
        f"isn't implemented yet. Implement `{impl_file}` so that test passes.\n\n{task_text}"
    )
    success, _, usage = run_editor_on_item(worktree_path, green_prompt, log_path,
                                           files=[impl_file, test_file], model=model, backend_override=backend_override)
    _accumulate_usage(record, usage)
    if not success:
        return finish(STATUS_BLOCKED, "TDD green phase: aider itself failed")

    def test_file_modified_since_red() -> bool:
        # Byte-identical to what the red phase produced, not just "not
        # reported as touched" - a whole-format rewrite that reproduces
        # the same content would still show as touched by git, and the
        # failure this actually guards against (the model editing its own
        # test to make it pass) is a real content change, which this
        # catches precisely without over-flagging a no-op rewrite.
        return path_matches_any(test_file, changed_files(worktree_path, red_commit))

    if test_file_modified_since_red():
        log(f"Green phase modified `{test_file}` despite being told not to. "
            f"Parking: {task_text}", log_path)
        return finish(STATUS_NEEDS_REVIEW, f"green phase modified the test file `{test_file}`")

    suspicious = suspicious_new_paths(changed_files(worktree_path, red_commit))
    if suspicious:
        return finish(STATUS_BLOCKED, f"TDD green phase created garbage filename(s): {suspicious}")

    restore_unnamed_files(worktree_path, base, [test_file, impl_file], log_path)
    wt.restore_todo(worktree_path, base, args.todo_file)
    if test_file_modified_since_red():
        # restore_unnamed_files only protects files the item didn't name;
        # test_file IS named (it's one of the two expected files), so an
        # edit to it survives that call and has to be caught here too.
        return finish(STATUS_NEEDS_REVIEW, f"green phase modified the test file `{test_file}`")

    touched = changed_files(worktree_path, base)
    record["touched_files"] = touched
    if not path_matches_any(impl_file, touched):
        return finish(STATUS_NEEDS_REVIEW, f"green phase never touched `{impl_file}`")

    corrupted = detect_prompt_leakage(worktree_path, touched)
    if corrupted:
        return finish(STATUS_BLOCKED, f"prompt text was written into {corrupted} during the green phase")

    scope = [worktree_path / f for f in touched]
    valid, validation_msg = validate_syntax(worktree_path, log_path, scope)

    validation_attempt = 0
    while not valid and validation_attempt < args.max_validation_retries:
        validation_attempt += 1
        log(f"TDD green phase failed validation (retry {validation_attempt}/"
            f"{args.max_validation_retries}): {task_text}", log_path)
        fix_success, _, fix_usage = run_editor_on_item(worktree_path, (
            f"The implementation in `{impl_file}` does not yet make the test in "
            f"`{test_file}` pass.\n\nOriginal task:\n{task_text}\n\n"
            f"Failure output:\n{validation_msg}\n\n"
            f"Fix `{impl_file}` so the test passes. Do not modify `{test_file}` "
            f"or touch any other file."
        ), log_path, files=[impl_file, test_file], model=model, backend_override=backend_override)
        _accumulate_usage(record, fix_usage)
        if not fix_success:
            # Deliberately not `break`. A failed fix attempt is often the
            # model regenerating byte-identical content (lite_editor
            # reports that as failure, correctly - nothing changed), and
            # breaking here spent the whole remaining retry budget on
            # one such attempt: measured on archaeologist's first TDD
            # item, retry 1 of 3 produced identical output and retries 2
            # and 3 never ran. Retries are bounded by the loop condition
            # anyway, so letting it try again costs at most the budget
            # already agreed to.
            log(f"TDD fix attempt {validation_attempt} produced nothing usable; "
                f"continuing to the next retry if any remain.", log_path)
            continue
        if test_file_modified_since_red():
            return finish(STATUS_NEEDS_REVIEW,
                         f"a validation-fix retry modified the test file `{test_file}`")
        restore_unnamed_files(worktree_path, base, [test_file, impl_file], log_path)
        wt.restore_todo(worktree_path, base, args.todo_file)
        if test_file_modified_since_red():
            return finish(STATUS_NEEDS_REVIEW,
                         f"a validation-fix retry modified the test file `{test_file}`")
        touched = changed_files(worktree_path, base)
        record["touched_files"] = touched
        valid, validation_msg = validate_syntax(worktree_path, log_path,
                                                scope=[worktree_path / f for f in touched])
    record["validation_retries"] = validation_attempt

    if not valid:
        record["validation_output"] = validation_msg[-4000:]
        return finish(STATUS_BLOCKED, "TDD green phase: validation failed")

    # `files=[impl_file]` only, not the test - a TDD review-fix isn't
    # allowed to touch the test any more than the earlier validation-fix
    # retries above were (see test_file_modified_since_red throughout this
    # function). lite_editor enforces that structurally; for the claude
    # backend the prompt still says so explicitly.
    if (gate_result := run_review_gate(worktree_path, base, task_text, [impl_file],
                                       args, log_path, record, finish, model=model,
                                       backend_override=backend_override, tdd=True,
                                       restore_base=red_commit)) is not None:
        return gate_result
    if test_file_modified_since_red():
        return finish(STATUS_NEEDS_REVIEW, f"a review-fix retry modified the test file `{test_file}`")

    with git_lock:
        merged, merge_output = wt.merge_ff(project_dir, record["branch"])
        if not merged:
            return finish(STATUS_NEEDS_REVIEW, f"passed checks but merge failed: {merge_output}")

        log(f"TDD item merged (test failed red, passed green): {task_text}", log_path)
        return finish(STATUS_DONE, "TDD: test failed before implementation, passed after")


def process_item(project_dir: Path, item: "TodoItem", index: int,
                 args, log_path: Path, model: str | None = None,
                 git_lock: threading.RLock | None = None, backend_override: str | None = None) -> tuple[str, dict]:
    """Run one item in its own worktree. Returns (status, record).

    The project checkout is only ever written to by the final
    fast-forward, and only for an item that passed every check. A failure
    is not reverted, because it was never applied -- it stays on its own
    branch and the caller moves to the next item from the same base.

    `model` overrides the aider model for this item (see
    run_aider_on_item) - used by --parallel to give each concurrent
    worker a different one. `git_lock` serializes the operations that
    touch project_dir's own git state (creating the worktree, merging,
    removing it) so concurrent workers can't race on it; every other
    step here (aider itself, every check) touches only this item's own
    worktree and needs no lock at all. Sequential runs still pass a real
    lock - just one nothing else ever contends for.
    """
    git_lock = git_lock or threading.RLock()
    base = git_head(project_dir)
    record: dict = {
        "index": index,
        "item": item.text,
        "base": base,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    with git_lock:
        worktree_path, branch = wt.create(project_dir, index, base)
    record["branch"] = branch
    actions = wt.materialize(project_dir, worktree_path)
    log(f"Worktree {worktree_path} on {branch}"
        + (f" ({', '.join(actions)})" if actions else " (nothing to materialize)"), log_path)

    def finish(status: str, reason: str) -> tuple[str, dict]:
        record["status"] = status
        record["reason"] = reason
        record["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        # A parked item's branch is the only copy of what the model wrote.
        with git_lock:
            wt.remove(project_dir, worktree_path, branch,
                      keep_branch=(status != STATUS_DONE) or args.keep_branches)
        return status, record

    # Parsed before the run, not after: these are both what aider is handed
    # as editable and what the touched-files check later holds it to.
    expected = expected_files(item.text)
    if expected:
        log(f"Handing aider the file(s) the item named: {expected}", log_path)

    if is_tdd_item(item.text):
        classification = classify_tdd_files(expected)
        if classification is None:
            log(f"TDD item but couldn't identify one test file and one implementation "
                f"file from {expected} (need exactly two named files, one test-shaped). "
                f"Parking: {item.text}", log_path)
            return finish(STATUS_NEEDS_REVIEW,
                         "TDD item did not name exactly one test file and one implementation file")
        test_file, impl_file = classification
        task_text = TDD_PREFIX_RE.sub("", item.text, count=1)
        return run_tdd_phases(project_dir, worktree_path, task_text, test_file, impl_file,
                              base, args, log_path, record, finish, model=model, git_lock=git_lock,
                              backend_override=backend_override)

    success = False
    attempt = 0
    while attempt <= args.max_retries and not success:
        attempt += 1
        if attempt > 1:
            log(f"Retry {attempt - 1}/{args.max_retries} for: {item.text}", log_path)
        success, _, usage = run_editor_on_item(worktree_path, item.text, log_path, files=expected, model=model, backend_override=backend_override)
        _accumulate_usage(record, usage)
    record["aider_attempts"] = attempt

    if not success:
        log(f"Aider failed after {attempt} attempt(s), parking blocked: {item.text}", log_path)
        return finish(STATUS_BLOCKED, f"aider itself failed after {attempt} attempt(s)")

    # Garbage filenames are looked for BEFORE anything is restored, because
    # restore_unnamed_files() would otherwise destroy the evidence: a file
    # named after the model's own output ("File Listing: stories/index.json")
    # is by definition not a file the task named, so it gets reverted as an
    # unnamed extra and `touched` is clean again by the time the check below
    # runs. Measured on blockroad: an item created `File Listing:
    # stories/index.json` and `File: stories/the-night-build.story`, and
    # parked for "validation failed" instead - burning all three validation
    # retries re-triggering the same corruption, because the failure it was
    # asked to fix was never the real one.
    suspicious = suspicious_new_paths(changed_files(worktree_path, base))
    if suspicious:
        log(f"Garbage-looking filename(s) created: {suspicious} - aider likely turned a "
            f"malformed file block into a new file named after its own output. "
            f"Parking: {item.text}", log_path)
        record["touched_files"] = suspicious
        return finish(STATUS_BLOCKED, f"garbage filename(s) created: {suspicious}")

    restore_unnamed_files(worktree_path, base, expected, log_path)
    if wt.restore_todo(worktree_path, base, args.todo_file):
        log(f"Item edited {args.todo_file}; restored it (the checklist is the "
            f"loop's bookkeeping, not the item's work).", log_path)
    touched = changed_files(worktree_path, base)
    record["expected_files"] = expected
    record["touched_files"] = touched

    # Byte-exact first: the only check here that isn't a heuristic.
    mismatched = content_mismatches(worktree_path, extract_exact_content_specs(item.text))
    if mismatched:
        log(f"Content doesn't match the task's exact spec for {mismatched}. Parking: {item.text}", log_path)
        return finish(STATUS_BLOCKED, f"content did not match the exact spec for {mismatched}")

    missing = [f for f in expected if not path_matches_any(f, touched)]
    if expected and missing:
        log(f"Expected file(s) not touched: {missing} (touched: {touched or 'nothing'}). "
            f"Parking needs-review: {item.text}", log_path)
        return finish(STATUS_NEEDS_REVIEW, f"named file(s) {missing} were never touched")

    corrupted = detect_prompt_leakage(worktree_path, touched)
    if corrupted:
        log(f"Prompt-leakage marker found in {corrupted}. Parking: {item.text}", log_path)
        return finish(STATUS_BLOCKED, f"prompt text was written into {corrupted} as content")

    scope = [worktree_path / f for f in touched]
    valid, validation_msg = validate_syntax(worktree_path, log_path, scope)

    validation_attempt = 0
    while not valid and validation_attempt < args.max_validation_retries:
        validation_attempt += 1
        log(f"Validation failed (retry {validation_attempt}/{args.max_validation_retries}), "
            f"asking aider to fix it: {item.text}", log_path)
        fix_success, _, fix_usage = run_editor_on_item(worktree_path, (
            f"The previous change for this task did not pass validation.\n\n"
            f"Original task:\n{item.text}\n\n"
            f"Validation failure output:\n{validation_msg}\n\n"
            f"Fix the failure above while still completing the original task. "
            f"Do not revert or abandon the original change; correct it."
        ), log_path, files=expected, model=model, backend_override=backend_override)
        _accumulate_usage(record, fix_usage)
        if not fix_success:
            # See the same spot in run_tdd_phases for why this continues
            # rather than breaking: a fix attempt that changed nothing
            # would otherwise consume the entire remaining retry budget.
            log(f"Fix attempt {validation_attempt} produced nothing usable during the "
                f"validation-fix retry for: {item.text}", log_path)
            continue
        # Same check, same reason, on every retry: a fix attempt can produce
        # the corruption just as easily as the first attempt, and retrying
        # into it is exactly the loop this ordering was written to stop.
        suspicious = suspicious_new_paths(changed_files(worktree_path, base))
        if suspicious:
            log(f"Garbage-looking filename(s) created during the fix retry: {suspicious}. "
                f"Parking rather than retrying into it again: {item.text}", log_path)
            record["touched_files"] = suspicious
            record["validation_retries"] = validation_attempt
            return finish(STATUS_BLOCKED, f"garbage filename(s) created on retry: {suspicious}")
        restore_unnamed_files(worktree_path, base, expected, log_path)
        wt.restore_todo(worktree_path, base, args.todo_file)
        touched = changed_files(worktree_path, base)
        record["touched_files"] = touched
        valid, validation_msg = validate_syntax(worktree_path, log_path, scope=[worktree_path / f for f in touched])
    record["validation_retries"] = validation_attempt

    if not valid:
        record["validation_output"] = validation_msg[-4000:]
        log(f"Validation failed, parking blocked: {item.text}", log_path)
        return finish(STATUS_BLOCKED, "validation failed")

    if (gate_result := run_review_gate(worktree_path, base, item.text, expected, args, log_path,
                                       record, finish, model=model,
                                       backend_override=backend_override)) is not None:
        return gate_result
    touched = record.get("touched_files", touched)

    if not touched:
        # Every check above passes vacuously when nothing changed, and the
        # merge below would be a no-op marked done -- the exact silent
        # false-'done' the touched-files check exists to prevent, reachable
        # here whenever the item named no files for it to check.
        log(f"Item changed nothing at all. Parking needs-review: {item.text}", log_path)
        return finish(STATUS_NEEDS_REVIEW, "the item produced no changes")

    # Locked from the merge attempt through finish()'s own cleanup: with
    # --parallel, another item's merge landing in between "we merged" and
    # "we recorded the resulting HEAD" would make record["merged"] wrong,
    # and two threads calling wt.merge_ff concurrently is a real git race
    # on project_dir's shared ref state, not just a bookkeeping one.
    with git_lock:
        merged, merge_output = wt.merge_ff(project_dir, branch)
        if not merged:
            log(f"Item passed but could not fast-forward the project: {merge_output}", log_path)
            return finish(STATUS_NEEDS_REVIEW, f"passed checks but merge failed: {merge_output}")

        log(f"Merged {branch} and marking done: {item.text}", log_path)
        record["merged"] = git_head(project_dir)
        return finish(STATUS_DONE, "passed every check and was merged")


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


def update_item_status(todo_path: Path, item: "TodoItem", status: str) -> None:
    """Writes one item's status into todo.md, re-reading the file fresh
    first so a concurrent worker's own status write to a DIFFERENT line
    isn't clobbered by a write built from a now-stale copy of the file.

    Safe to reuse `item.line_index` against this fresh read because line
    positions are stable across the run: parse_todo() never reorders or
    removes lines, and every status write only ever overwrites its own
    item's single line in place - so the line at that index in a freshly
    re-read file is still this same item, whatever else changed
    elsewhere in the meantime.
    """
    raw_lines, _ = parse_todo(todo_path)
    item.status = status
    raw_lines[item.line_index] = item.render()
    todo_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")


def log_run_summary(items_processed: int, parked: list[dict], todo_path: Path, log_path: Path,
                    runs_dir: Path | None = None) -> None:
    log(f"Run complete. {items_processed} item(s) processed. See {todo_path} for status.", log_path)
    if runs_dir and runs_dir.is_dir():
        # Read back every item's own record rather than threading a
        # running total through both dispatch paths - `parked` only ever
        # holds the failures, and a token total that silently excluded
        # every item that actually merged would be worse than none.
        prompt = completion = 0
        seconds = 0.0
        n = 0
        for f in runs_dir.glob("item-*.json"):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            tokens = rec.get("tokens")
            if not tokens:
                continue
            n += 1
            prompt += tokens.get("prompt_tokens", 0)
            completion += tokens.get("completion_tokens", 0)
            seconds += tokens.get("seconds", 0.0)
        if n:
            log(f"Tokens: ~{prompt:,} prompt + ~{completion:,} completion across {n} item(s)"
                + (f" ({seconds:.0f}s of measured generation time)" if seconds else "")
                + ". Local model, no bill - this is where compute went, not what it cost.", log_path)
    if parked:
        # The point of parking rather than reverting: the work still
        # exists. Name the branches here so the run ends with something
        # actionable rather than a count.
        log(f"{len(parked)} item(s) did not pass and were never applied to the project. "
            f"Each one's work is on its own branch:", log_path)
        for r in parked:
            log(f"  [{r['status']}] {r['branch']} - {r['reason']}\n"
                f"      {r['item'].splitlines()[0][:110]}", log_path)
        log(f"Review with: git log -p {parked[0]['branch']}   "
            f"(and `git branch -D` once you're done with it), or run review_server.py.", log_path)


def run_parallel(project_dir: Path, todo_path: Path, args, log_path: Path,
                 models: list[str], runs_dir: Path) -> tuple[int, list[dict]]:
    """Runs open items concurrently, one worker per model in `models`.

    Each worker is deliberately a DIFFERENT ollama model, not N copies of
    the same one. This script never raises OLLAMA_NUM_PARALLEL or assumes
    any particular ollama concurrency configuration - two workers sharing
    one model would just take turns on the GPU behind ollama's own
    default of one generation at a time, with none of the wall-clock
    benefit and all of the added complexity here. Verified directly on
    real hardware: two different models (a 14B and a 7B coder) loaded and
    generated concurrently on one 24GB GPU with real headroom left over
    (~21.8GB used, both at 100% GPU, both producing real output).

    One `git_lock` (RLock) serializes every operation that touches
    project_dir's own shared git state - creating a worktree, merging one
    in, removing it - across every worker. Nothing else needs it: aider
    itself and every check run entirely inside that item's own worktree,
    fully isolated from every other item in flight. See merge_ff()'s
    docstring for the other half of what parallel dispatch needs from
    git: a fast-forward is no longer guaranteed (two items can branch
    from the same base and only one can still be "the tip" once the
    first lands), so a real merge is the fallback, not an error.

    Item selection is claim-and-scan under the same lock: re-read the
    checklist fresh, take the next STATUS_OPEN item not already claimed
    this run, mark it claimed, release the lock, then do the actual work
    unlocked. Re-reading fresh each time (rather than working from one
    stale snapshot) is what lets a worker notice items another worker has
    already finished; the in-memory `claimed` set is what stops two
    workers claiming the same still-open item in the gap before either of
    them has written a status back.

    Which open item a worker takes first is tiered by slot, not just
    list order: slot 0 (the first model in `models`) is the strong tier
    and prefers a "hard" item; every other slot is the weak tier and
    prefers an "easy" one (see estimate_difficulty()). Either tier falls
    back to whatever's open if nothing of its preferred difficulty is
    left, so a worker never sits idle purely because the item it would
    rather have isn't available yet - the tier is a preference over the
    queue, not a partition of it.
    """
    git_lock = threading.RLock()
    parked_lock = threading.Lock()
    claimed: set[int] = set()
    counter = {"n": 0}
    parked: list[dict] = []

    def claim_next(prefer: str):
        with git_lock:
            if args.max_items is not None and counter["n"] >= args.max_items:
                return None, None
            _, items = parse_todo(todo_path)
            open_items = [(i, it) for i, it in enumerate(items)
                         if it.status == STATUS_OPEN and i not in claimed]
            if not open_items:
                return None, None
            preferred = [(i, it) for i, it in open_items if estimate_difficulty(it.text) == prefer]
            i, item = preferred[0] if preferred else open_items[0]
            claimed.add(i)
            counter["n"] += 1
            return counter["n"], item

    def worker(slot: int, model: str):
        # Slot 0 is the strong tier by convention - the first model named
        # is the one a user would put first, and this project's own
        # measurements only ever showed the 14B (not the 7B) handling the
        # judgment calls DIFFICULTY_HARD is reserved for.
        prefer = DIFFICULTY_HARD if slot == 0 else DIFFICULTY_EASY
        while True:
            index, item = claim_next(prefer)
            if item is None:
                return
            tag = f"worker {slot}:{model}"
            log(f"[{tag}] === Item {index} ({estimate_difficulty(item.text)}): "
                f"{item.text.splitlines()[0][:100]} ===", log_path)
            status, record = process_item(project_dir, item, index, args, log_path,
                                          model=model, git_lock=git_lock)
            if status != STATUS_DONE and _should_fall_back(args, model):
                log(f"[{tag}] {model} failed [{status}]. Escalating to {_fallback_label(args)}...", log_path)
                status, record = process_item(project_dir, item, index, args, log_path,
                                              model=args.fallback_model, git_lock=git_lock,
                                              backend_override=getattr(args, "fallback_backend", "aider"))
                record["fallback_used"] = _fallback_label(args)
            record["worker"] = slot
            record["model"] = model
            record["difficulty"] = estimate_difficulty(item.text)
            with git_lock:
                update_item_status(todo_path, item, status)
                record_path = record_run(runs_dir, index, record)
            if status != STATUS_DONE:
                with parked_lock:
                    parked.append(record)
                log(f"[{tag}] Parked [{status}]: {record['reason']}. Details in {record_path}.", log_path)
            else:
                log(f"[{tag}] Merged: {item.text.splitlines()[0][:100]}", log_path)

    log(f"Starting {len(models)} worker(s) on {project_dir}: {models} (run log: {runs_dir})", log_path)
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = [pool.submit(worker, i, m) for i, m in enumerate(models)]
        for f in futures:
            f.result()  # re-raise a worker's exception instead of swallowing it

    return counter["n"], parked


def main():
    parser = argparse.ArgumentParser(description="Unattended Aider loop runner")
    parser.add_argument("--project-dir", required=True, type=str,
                         help="Path to the project containing todo.md and .aider.conf.yml")
    parser.add_argument("--todo-file", default="todo.md", type=str,
                         help="Name of the checklist file inside project-dir (default: todo.md)")
    parser.add_argument("--max-items", default=DEFAULT_MAX_ITEMS, type=int,
                         help="Stop after this many items (default: run until list is empty)")
    parser.add_argument("--max-retries", default=DEFAULT_MAX_RETRIES, type=int,
                         help="Retries per item if Aider itself errors out (not validation failures)")
    parser.add_argument("--max-validation-retries", default=DEFAULT_MAX_VALIDATION_RETRIES, type=int,
                         help="Retries when validation (build/test) fails, not when aider itself "
                              "crashes (that's --max-retries). Each retry re-invokes aider with "
                              "the original task PLUS the actual validation failure output, asking "
                              "it to fix the failure without abandoning the task. Only after these "
                              "are exhausted does the item get marked blocked and reverted.")
    parser.add_argument("--sleep-between", default=DEFAULT_SLEEP_BETWEEN_ITEMS, type=int,
                         help="Seconds to pause between items (Ctrl+C window)")
    parser.add_argument("--stop-on-problem", action="store_true",
                         help="Stop the run at the first item that doesn't pass, the way the "
                              "loop behaved before per-item worktrees. Now off by default: a "
                              "failed item is never applied to the project, so the next item "
                              "cannot build on top of it and there is nothing to stop for.")
    parser.add_argument("--keep-branches", action="store_true",
                         help="Keep the per-item branch even for items that passed and merged "
                              "(a failed item's branch is always kept - it's the only copy of "
                              "what the model produced).")
    parser.add_argument("--backend", default=None, choices=["aider", "lite"],
                         help="Which editing backend to use, overriding [model] backend in "
                              ".aiderloop.toml. 'aider' (the default) shells out to aider; "
                              "'lite' uses lite_editor.py, built for exactly this loop's usage - "
                              "it decides the file set in Python before calling the model and asks "
                              "for whole-file content rather than a diff, so SEARCH/REPLACE "
                              "mismatches, file-selection reflection loops and invented filenames "
                              "are all structurally impossible. See its module docstring.")
    parser.add_argument("--models", default=None, type=str,
                         help="Comma-separated ollama model name(s), overriding .aider.conf.yml's "
                              "model: key. One name runs sequentially as before, just with that "
                              "model. Two or more run that many items concurrently, one worker per "
                              "model - see README's parallel-workers section for why each worker "
                              "is deliberately a DIFFERENT model rather than the same one twice.")
    parser.add_argument("--fallback-model", default=None, type=str,
                         help="Model to escalate to if the primary model fails validation/review on an item. "
                              "Particularly useful when a fast coder model repeatedly chokes on syntax errors "
                              "and a heavier reasoning model is needed to unblock it.")
    parser.add_argument("--fallback-backend", default="aider", choices=["aider", "lite", "claude"],
                         help="Editing backend for the fallback (default: aider). `claude` hands a parked "
                              "item to the Claude Code CLI (`claude -p`, billed to your subscription, "
                              "never the API) - --fallback-model is then optional and names a Claude "
                              "model such as sonnet or opus.")
    parser.add_argument("--review", choices=["claude", "agy"], default=None,
                         help="Have a second model review each passing diff before it merges; a "
                              "rejection is fed back to the editor for up to --max-validation-retries "
                              "rounds.")
    parser.add_argument("--review-model", default=None,
                         help="Claude model for --review claude (default: the CLI's own default).")
    parser.add_argument("--agy-review", dest="review", action="store_const", const="agy",
                         help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true",
                         help="Enable verbose output")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    global CONFIG
    CONFIG = load_config(project_dir)
    if args.backend:
        # CLI wins over the project's own [model] backend, the same way
        # --models wins over [worker] models. Written into CONFIG rather
        # than threaded through every call site, since run_editor_on_item
        # reads it from there anyway.
        CONFIG.setdefault("model", {})["backend"] = args.backend
    todo_path = project_dir / args.todo_file
    log_path = project_dir / "aider_loop.log"

    if not project_dir.is_dir():
        print(f"Project dir does not exist: {project_dir}")
        sys.exit(1)
    if not todo_path.is_file():
        print(f"No {args.todo_file} found in {project_dir}")
        sys.exit(1)
    backend = cfg("model", "backend", default="aider")
    log(f"Editing backend: {backend}", log_path)
    if backend != "lite" and not (project_dir / ".aider.conf.yml").is_file():
        log("Warning: no .aider.conf.yml found in project dir - aider will use "
            "its own defaults, which may not be your local Qwen setup.", log_path)

    # Now a hard requirement rather than a warning: every item runs in a
    # `git worktree` branched from HEAD, which is also the only thing
    # keeping a failed item away from the project checkout. Without git
    # there is no isolation to degrade to -- the old non-git path ran items
    # directly in the tree and left broken changes in place for the next
    # item to build on, which is exactly what this replaced.
    if not is_git_repo(project_dir):
        print(f"{project_dir} is not a git repo; per-item worktrees need one.")
        sys.exit(1)

    # Before the cleanliness check, not after: a worktree abandoned by a
    # killed run doesn't dirty the checkout, but it does hold a branch and
    # it will never be collected otherwise.
    wt.prune_stale(project_dir, log=lambda m: log(m, log_path))

    # The checklist and the loop's own log are excused - see is_clean().
    clean, dirty = wt.is_clean(project_dir, exempt=(args.todo_file, "aider_loop.log"))
    if not clean:
        print("The project checkout has uncommitted changes to tracked files:\n"
              f"{dirty}\n\n"
              "A passing item is merged with `git merge --ff-only`, which refuses to run "
              "over local modifications. Commit or stash first - finding this out after an "
              "item has already spent half an hour in the model is the wrong time.")
        sys.exit(1)

    if not (project_dir / CONFIG_FILENAME).is_file():
        log(f"Note: no {CONFIG_FILENAME} in project dir - using built-in defaults. "
            f"See the README for what it configures (author model, validate "
            f"commands, preflight limits).", log_path)

    preflight_todo_rules(todo_path, log_path)
    preflight_repo_size(project_dir, log_path)

    runs_dir = wt.runs_root(project_dir) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    models = resolve_models(args)
    if len(models) > 1:
        items_processed, parked = run_parallel(project_dir, todo_path, args, log_path, models, runs_dir)
        log_run_summary(items_processed, parked, todo_path, log_path, runs_dir)
        return

    # One name (or none) runs the original sequential path, just with
    # that model overriding .aider.conf.yml when given. This is the exact
    # code that existed before --models did - untouched except for
    # threading that one optional override through, so the default case
    # (no --models at all) is byte-for-byte the same run it always was.
    single_model = models[0] if models else None

    log(f"Starting aider_loop on {project_dir} (run log: {runs_dir})", log_path)

    items_processed = 0
    parked: list[dict] = []

    while True:
        if args.max_items is not None and items_processed >= args.max_items:
            log(f"Reached max-items limit ({args.max_items}), stopping.", log_path)
            break

        raw_lines, items = parse_todo(todo_path)
        item = next_open_item(items)

        if item is None:
            log("No open items remain in todo.md. Done.", log_path)
            break

        index = items_processed + 1
        log(f"=== Item {index}: {item.text} ===", log_path)

        status, record = process_item(project_dir, item, index, args, log_path, model=single_model)

        if status != STATUS_DONE and _should_fall_back(args, single_model):
            log(f"Model {single_model} failed [{status}]. Escalating to {_fallback_label(args)}...", log_path)
            status, record = process_item(project_dir, item, index, args, log_path, model=args.fallback_model, backend_override=getattr(args, "fallback_backend", "aider"))
            record["fallback_used"] = _fallback_label(args)

        item.status = status
        write_todo(todo_path, raw_lines, items)
        record_path = record_run(runs_dir, index, record)
        items_processed += 1

        if status != STATUS_DONE:
            parked.append(record)
            log(f"Parked [{status}] on branch {record['branch']}: {record['reason']}. "
                f"The project checkout is untouched; details in {record_path}.", log_path)
            if args.stop_on_problem:
                log("Stopping at the first problem (--stop-on-problem).", log_path)
                break

        log(f"=== Finished item ({items_processed} total this run) ===\n", log_path)
        time.sleep(args.sleep_between)
        continue

    log_run_summary(items_processed, parked, todo_path, log_path, runs_dir)


if __name__ == "__main__":
    main()
