"""Orchestrator and execution runner for forge."""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import agy_editor, claude_editor, config, lite_editor
from . import worktree as wt
from .config import (
    CONFIG_FILENAME,
    DEFAULT_MAX_ITEMS,
    DEFAULT_HARD_VALIDATION_RETRIES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_VALIDATION_RETRIES,
    DEFAULT_SLEEP_BETWEEN_ITEMS,
    cfg,
    check_ollama_context,
    load_config,
    log,
    resolve_models,
)
from .tdd import (
    TDD_PREFIX_RE,
    classify_tdd_files,
    is_tdd_item,
    run_tdd_phases,
)
from .todo import (
    STATUS_BLOCKED,
    STATUS_DONE,
    STATUS_NEEDS_REVIEW,
    STATUS_OPEN,
    TodoItem,
    next_open_item,
    parse_todo,
    preflight_todo_rules,
    update_item_status,
    write_todo,
)
from .validator import (
    changed_files,
    content_mismatches,
    detect_prompt_leakage,
    expected_files,
    extract_exact_content_specs,
    path_matches_any,
    preflight_repo_size,
    restore_unnamed_files,
    suspicious_new_paths,
    validate_syntax,
)

# Used by --models to route items between worker tiers - see
# estimate_difficulty()'s docstring for the reasoning, and README's
# "Parallel workers" section for how tiers map to worker slots.
DIFFICULTY_EASY = "easy"
# Backends that hand the item to a hosted model's CLI rather than a local one.
CLOUD_BACKENDS = ("claude", "agy")
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


def _truncated_without_output(output: str, usage: dict | None) -> bool:
    """True when the editor hit its output ceiling without ever emitting a
    usable file block - a diagnosable condition with a known remedy, not a
    reason to repeat the same call."""
    if usage is not None and usage.get("done_reason") == "length":
        return lite_editor.NO_OUTPUT_MARKER in (output or "")
    return lite_editor.TRUNCATED_MARKER in (output or "")


def validation_retries_for(args, item_text: str) -> int:
    """The validation-fix budget for one item.

    Flat --max-validation-retries unless --hard-validation-retries is set,
    in which case items estimate_difficulty() calls "hard" get the larger
    budget and everything else is left alone. The asymmetry is the point:
    an easy item here is one with a byte-exact spec, which is already
    checked byte-for-byte whether the model retries or not, so extra
    retries on it cost model time and buy no additional certainty. A hard
    item is the opposite - its only check is whether the tests actually
    pass, and that is exactly the check a second look can flip.
    """
    budget = getattr(args, "hard_validation_retries", None)
    if budget is None or estimate_difficulty(item_text) != DIFFICULTY_HARD:
        return args.max_validation_retries
    return max(budget, args.max_validation_retries)


def is_git_repo(project_dir: Path) -> bool:
    return (project_dir / ".git").exists()


def repair_prompt(item_text: str, note: str) -> str:
    """The editor prompt for --repair-from: the original task, why the
    earlier attempt (already committed in the worktree) was rejected, and
    an instruction to fix rather than restart."""
    note = note.strip() or "(no failure details were recorded)"
    return (
        f"{item_text}\n\n"
        f"A previous attempt at this task is already committed in this repository, "
        f"but it was rejected. Why it was rejected:\n\n{note}\n\n"
        f"Fix the problems above so the task is fully and correctly done. Keep the "
        f"parts of the previous attempt that are correct; only start over if it is "
        f"unsalvageable."
    )


def git_head(project_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(project_dir),
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_aider_on_item(project_dir: Path, item_text: str, log_path: Path,
                      files: list[str] | None = None, model: str | None = None,
                      backend_override: str | None = None) -> tuple[bool, str]:
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
    return f"{backend} ({name})" if backend in CLOUD_BACKENDS else f"fallback model {name}"


def _should_fall_back(args, model: str | None) -> bool:
    if getattr(args, "fallback_backend", None) in CLOUD_BACKENDS:
        return True
    return bool(getattr(args, "fallback_model", None)) and args.fallback_model != model


# Set once by main() from --doer-preamble and --save-editor-output. Module
# state rather than an argument because run_editor_on_item has no `args`
# and tdd.py calls it too; both are read-only once the run has started.
DOER_PREAMBLE = ""
EDITOR_OUTPUT_DIR: Path | None = None
_editor_output_lock = threading.Lock()


def run_editor_on_item(project_dir: Path, item_text: str, log_path: Path,
                       files: list[str] | None = None, model: str | None = None,
                       backend_override: str | None = None,
                       hard_item: bool | None = None,
                       force_think_off: bool = False) -> tuple[bool, str, dict]:
    """Every editor call goes through here, so this is the one place
    --doer-preamble is applied: prepended to the prompt text the editor
    model receives, and nothing else. Every check (expected files, exact
    content, TDD classification, difficulty) reads the item's own text,
    never this, and difficulty is decided here from the text *without* the
    preamble - so with no preamble the call is exactly what it was before.
    It exists for Leash's sabotage experiment: a side instruction only the
    doer sees, which the checks are not allowed to know about."""
    if hard_item is None:
        hard_item = estimate_difficulty(item_text) == DIFFICULTY_HARD
    editor_text = f"{DOER_PREAMBLE}\n\n{item_text}" if DOER_PREAMBLE else item_text
    success, output, usage = _dispatch_editor(project_dir, editor_text, log_path, files=files,
                                              model=model, backend_override=backend_override,
                                              hard_item=hard_item, force_think_off=force_think_off)
    if EDITOR_OUTPUT_DIR is not None:
        _save_editor_output(success, output, usage, model, backend_override)
    return success, output, usage


def _save_editor_output(success: bool, output: str, usage: dict, model: str | None,
                        backend_override: str | None) -> None:
    """Appends one editor call's raw response to <dir>/editor-calls.jsonl.
    What the model said outside its file blocks, and its reasoning trace,
    are otherwise discarded; a benchmark needs them to tell a refusal from
    a failure. The prompt is deliberately not written."""
    backend = backend_override or cfg("model", "backend", default="aider")
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "backend": backend,
           "model": model, "success": success, "output": output,
           "thinking": lite_editor.last_thinking() if backend == "lite" else "",
           "usage": usage}
    with _editor_output_lock:
        EDITOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(EDITOR_OUTPUT_DIR / "editor-calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _dispatch_editor(project_dir: Path, item_text: str, log_path: Path,
                     files: list[str] | None = None, model: str | None = None,
                     backend_override: str | None = None,
                     hard_item: bool | None = None,
                     force_think_off: bool = False) -> tuple[bool, str, dict]:
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
    if backend == "agy":
        success, output = agy_editor.run_agy_on_item(
            project_dir, item_text, lambda m: log(m, log_path), files=files, model=model)
        return success, output, agy_editor.last_usage()
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
        # Hard-item overrides. Both exist because of one measured failure
        # (see lite_editor.OUTPUT_BUDGET_DIRECTIVE): on an open-ended
        # algorithmic item the model can spend the whole budget reasoning
        # and emit no file block at all. They are deliberately scoped to
        # hard items - an exact-spec item transcribes rather than decides,
        # which is the case this harness is already reliable on, and there
        # is no reason to spend more tokens or change its prompt.
        if hard_item is None:
            hard_item = estimate_difficulty(item_text) == DIFFICULTY_HARD
        if hard_item:
            hard_np = cfg("model", "lite_hard_num_predict", default=None)
            if hard_np:
                kwargs["num_predict"] = hard_np
            if cfg("model", "lite_hard_budget_directive", default=False):
                kwargs["budget_directive"] = True
            if cfg("model", "lite_hard_think_off", default=False):
                kwargs["think"] = False
        # Set by the retry loop after an attempt was truncated mid-reasoning
        # without producing a file block. Applies whatever the item's
        # difficulty, because at this point it is not a guess about the item
        # - it is a measured fact about what the last attempt did.
        if force_think_off:
            kwargs["think"] = False
        success, output = lite_editor.run_lite_on_item(
            project_dir, item_text, log_path, files=files,
            model=model or cfg("model", "author"), **kwargs)
        return success, output, lite_editor.last_usage()
    success, output = run_aider_on_item(project_dir, item_text, log_path, files=files, model=model)
    return success, output, parse_aider_token_line(output)


def _accumulate_usage(record: dict, usage: dict, model: str | None = None) -> None:
    """Adds one call's usage into an item's running total. Called after
    every editor call, including ones that end in failure - a parked
    item's tokens were still real compute spent, and are worth knowing
    about (an item that burned three retries' worth of tokens and still
    parked is a different kind of expensive than one that failed fast)."""
    totals = record.setdefault("tokens", {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})
    p = usage.get("prompt_tokens", 0)
    c = usage.get("completion_tokens", 0)
    s = usage.get("seconds", 0.0)
    totals["prompt_tokens"] += p
    totals["completion_tokens"] += c
    totals["seconds"] += s
    m = model or record.get("model") or usage.get("model") or cfg("model", "author") or "default"
    config.emit_tokens(p, c, s, model=m)


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
        elif reviewer == "ollama":
            from . import spec_compiler
            model_name = getattr(args, "review_model", None) or "qwen3-coder:30b"
            prompt = f"Review this diff for the task below. The VERY LAST LINE of your reply must be exactly one word: APPROVE if it's correct, REJECT if not (with your reasons above it).\n\nTask:\n{item_text}\n\nDiff:\n{diff}"
            try:
                out = spec_compiler.call_ollama(model=model_name, prompt=prompt, url=spec_compiler.DEFAULT_OLLAMA_URL, num_ctx=4096, num_predict=2048, timeout=90.0).strip()
                last = out.splitlines()[-1].strip().upper() if out else ""
                approved, feedback = (last == "APPROVE" or (last != "REJECT" and out.upper().startswith("APPROVE"))), out
            except Exception as e:
                approved, feedback = None, str(e)
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
        _accumulate_usage(record, fix_usage, model=model)
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


def process_item(project_dir: Path, item: TodoItem, index: int,
                 args, log_path: Path, model: str | None = None,
                 git_lock: threading.RLock | None = None,
                 backend_override: str | None = None) -> tuple[str, dict]:
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
        "model": model or cfg("model", "author") or "default",
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
        config.emit_stage(index, item.text, "merged" if status == STATUS_DONE else "parked",
                          record.get("touched_files", []))
        # A parked item's branch is the only copy of what the model wrote.
        with git_lock:
            wt.remove(project_dir, worktree_path, branch,
                      keep_branch=(status != STATUS_DONE) or args.keep_branches)
        return status, record

    # Parsed before the run, not after: these are both what aider is handed
    # as editable and what the touched-files check later holds it to.
    expected = expected_files(item.text)
    config.emit_stage(index, item.text, "worktree", expected)
    if expected:
        log(f"Handing aider the file(s) the item named: {expected}", log_path)

    # Repair mode (--repair-from): start from an earlier, parked attempt and
    # ask the editor to fix it. `base` deliberately stays the original base,
    # so every check below still judges the whole change, not just the fix.
    editor_task = item.text
    repair_from = getattr(args, "repair_from", None)
    if repair_from:
        if not wt.start_from(worktree_path, repair_from):
            log(f"--repair-from {repair_from} is not a commit in this repo. Parking: {item.text}",
                log_path)
            return finish(STATUS_BLOCKED, f"repair source {repair_from} not found")
        record["mode"] = "repair"
        record["repair_from"] = repair_from
        editor_task = repair_prompt(item.text, getattr(args, "repair_note_text", ""))
        log(f"Repair mode: starting from {repair_from}", log_path)

    val_budget = validation_retries_for(args, item.text)
    record["validation_retry_budget"] = val_budget

    # A TDD item's red/green phases can't be replayed on top of an attempt
    # that already wrote the test, so repair takes the single-pass path with
    # both files named; the project's validate commands still run the tests.
    if is_tdd_item(item.text) and not repair_from:
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
                              backend_override=backend_override,
                              validation_retries=val_budget)

    success = False
    attempt = 0
    last_output = ""
    # Set once an attempt has been truncated mid-reasoning without producing
    # a file block. Retrying that identically is pure waste - measured on
    # book-store: three attempts, 182s and the full 8000-token budget each,
    # no file written any time. The remedy is known (stop spending the
    # budget on a reasoning trace), so the next attempt applies it instead
    # of repeating the same call and hoping.
    think_off_next = False
    while attempt <= args.max_retries and not success:
        attempt += 1
        if attempt > 1:
            why = " (previous attempt was truncated before it wrote a file; " \
                  "disabling the model's thinking mode for this one)" if think_off_next else ""
            log(f"Retry {attempt - 1}/{args.max_retries}{why} for: {item.text}", log_path)
        config.emit_stage(index, item.text, "coding", expected)
        if think_off_next:
            # Recorded, not just logged: the log line lives at the head of a
            # long item log and any consumer keeping a bounded tail of it
            # (Leash keeps 4000 chars) loses exactly the evidence that this
            # adaptation happened at all. An adaptation nobody can observe
            # after the fact is indistinguishable from one that never fired.
            record["adaptive_think_off_retries"] = \
                record.get("adaptive_think_off_retries", 0) + 1
        success, last_output, usage = run_editor_on_item(
            worktree_path, editor_task, log_path, files=expected, model=model,
            backend_override=backend_override, force_think_off=think_off_next)
        _accumulate_usage(record, usage, model=model)
        if not success and _truncated_without_output(last_output, usage):
            record["truncated_attempts"] = record.get("truncated_attempts", 0) + 1
            # Only worth trying if the model actually has a reasoning mode
            # to turn off; otherwise the next attempt is identical anyway.
            think_off_next = lite_editor.is_thinking_model(
                model or cfg("model", "author") or "")
    record["aider_attempts"] = attempt

    if success and getattr(args, "snapshot_first_attempt", False):
        record["first_attempt_commit"] = wt.snapshot_first_attempt(worktree_path, branch)

    if not success:
        # Two different things end up here and they must not read the same.
        # "The editor produced nothing parseable" is THIS HARNESS failing to
        # get output it could use - it says nothing about whether the model
        # was right, careful or uncertain. Reporting it with the same words
        # as a genuine park makes a harness bug look like the safety net
        # working, which is exactly how 34 of 120 trials on this project's
        # own benchmark were misread.
        if _truncated_without_output(last_output, None) or lite_editor.NO_OUTPUT_MARKER in last_output:
            record["editor_failure"] = "no_parseable_output"
            log(f"Editor produced no parseable output after {attempt} attempt(s) - this is a "
                f"harness failure, not a judgment about the item. Parking: {item.text}", log_path)
            return finish(STATUS_BLOCKED,
                          f"editor produced no parseable output after {attempt} attempt(s) "
                          f"({lite_editor.NO_OUTPUT_MARKER})")
        record["editor_failure"] = "editor_error"
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
    config.emit_stage(index, item.text, "validating", touched)
    valid, validation_msg = validate_syntax(worktree_path, log_path, scope)

    validation_attempt = 0
    while not valid and validation_attempt < val_budget:
        validation_attempt += 1
        log(f"Validation failed (retry {validation_attempt}/{val_budget}), "
            f"asking aider to fix it: {item.text}", log_path)
        fix_success, _, fix_usage = run_editor_on_item(worktree_path, (
            f"The previous change for this task did not pass validation.\n\n"
            f"Original task:\n{item.text}\n\n"
            f"Validation failure output:\n{validation_msg}\n\n"
            f"Fix the failure above while still completing the original task. "
            f"Do not revert or abandon the original change; correct it."
        ), log_path, files=expected, model=model, backend_override=backend_override)
        _accumulate_usage(record, fix_usage, model=model)
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


def run_parallel(project_dir: Path, todo_path: Path, args, log_path: Path,
                 models: list[str], runs_dir: Path) -> tuple[int, list[dict]]:
    """Runs open items concurrently, one worker per model in `models`.

    Each worker is deliberately a DIFFERENT ollama model, not N copies of
    the same one. This script never raises OLLAMA_NUM_PARALLEL or assumes
    any particular ollama concurrency configuration - two workers sharing
    one model would just queue behind each other's GPU time behind ollama's
    own default of one generation at a time, with none of the wall-clock
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
            if config.is_stop_requested():
                return None, None
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


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--hard-validation-retries", default=DEFAULT_HARD_VALIDATION_RETRIES,
                         type=int,
                         help="Validation-retry budget for items estimate_difficulty() rates "
                              "'hard' (TDD items, multi-file items, anything without a byte-exact "
                              "content spec). Leave unset to use the flat --max-validation-retries "
                              "for every item. An exact-spec item is checked byte-for-byte whether "
                              "it retries or not, so this deliberately spends the extra attempts "
                              "only where the tests passing is the only signal there is.")
    parser.add_argument("--lite-hard-num-predict", default=None, type=int,
                         help="lite backend only: num_predict for items estimate_difficulty() "
                              "rates 'hard', overriding [model] lite_num_predict for those items. "
                              "An open-ended item can spend the whole default budget reasoning and "
                              "never emit a file block at all; this buys it room to finish.")
    parser.add_argument("--lite-hard-think-off", action="store_true",
                         help="lite backend only: disable the model's thinking mode on hard items. "
                              "On a thinking-capable model the reasoning trace is spent from the "
                              "same num_predict budget as the answer, so an open-ended item can "
                              "burn the whole budget reasoning and emit no file block at all.")
    parser.add_argument("--lite-hard-budget-directive", action="store_true",
                         help="lite backend only: prepend an output-budget directive to hard items, "
                              "telling the model to emit the file block first and not to weigh "
                              "alternatives in its output. Counterpart to --lite-hard-num-predict: "
                              "that one raises the ceiling, this one lowers the demand.")
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
    parser.add_argument("--backend", default=None, choices=["aider", "lite", "claude", "agy"],
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
    parser.add_argument("--fallback-backend", default="aider", choices=["aider", "lite", "claude", "agy"],
                         help="Editing backend for the fallback (default: aider). `claude` hands a parked "
                              "item to the Claude Code CLI (`claude -p`, billed to your subscription, "
                              "never the API) - --fallback-model is then optional and names a Claude "
                              "model such as sonnet or opus. `agy` does the same through the agy CLI, "
                              "with --fallback-model naming one of `agy models`, e.g. gemini-3.8-flash-high.")
    parser.add_argument("--review", choices=["claude", "agy", "ollama"], default=None,
                         help="Have a second model review each passing diff before it merges; a "
                              "rejection is fed back to the editor for up to --max-validation-retries "
                              "rounds.")
    parser.add_argument("--review-model", default=None,
                         help="Claude model for --review claude (default: the CLI's own default).")
    parser.add_argument("--agy-review", dest="review", action="store_const", const="agy",
                         help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-first-attempt", action="store_true",
                         help="Pin each item's first successful editor attempt, before any check, "
                              "restore or retry changes it, at refs/leash/first-attempt/<branch> "
                              "and record it as first_attempt_commit. For benchmarks that grade "
                              "the unchecked output of the same attempt the checks judged.")
    parser.add_argument("--repair-from", default=None, metavar="REV",
                         help="Start the item from REV (typically a parked item's branch) instead of "
                              "HEAD and ask the editor to fix that attempt. Checks still diff against "
                              "the original base. TDD items take the single-pass path in this mode.")
    parser.add_argument("--repair-note", default=None, metavar="FILE",
                         help="With --repair-from: a file explaining why the earlier attempt was "
                              "rejected (the parked record's reason and check output).")
    parser.add_argument("--doer-preamble", default=None, metavar="FILE",
                         help="Prepend this file's text to every editor prompt, and to nothing "
                              "else: no check, log line or record sees it. For sabotage "
                              "experiments, where the doer gets a side instruction the checks "
                              "must not know about. A missing or empty file changes nothing.")
    parser.add_argument("--save-editor-output", default=None, metavar="DIR",
                         help="Append each editor call's raw response (and, for lite, the "
                              "model's reasoning trace) to DIR/editor-calls.jsonl. Observational "
                              "only; the prompt is not written.")
    parser.add_argument("--verbose", action="store_true",
                         help="Enable verbose output")
    args = parser.parse_args(argv)
    global DOER_PREAMBLE, EDITOR_OUTPUT_DIR
    args.doer_preamble_text = ""
    if args.doer_preamble:
        try:
            args.doer_preamble_text = Path(args.doer_preamble).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            pass
        except OSError as e:
            parser.error(f"cannot read --doer-preamble: {e}")
    DOER_PREAMBLE = args.doer_preamble_text
    EDITOR_OUTPUT_DIR = Path(args.save_editor_output).expanduser().resolve() \
        if args.save_editor_output else None
    args.repair_note_text = ""
    if args.repair_note:
        if not args.repair_from:
            parser.error("--repair-note requires --repair-from")
        try:
            args.repair_note_text = Path(args.repair_note).read_text(encoding="utf-8")
        except OSError as e:
            parser.error(f"cannot read --repair-note: {e}")

    project_dir = Path(args.project_dir).expanduser().resolve()
    from . import config
    config.reset_stop()
    config.REDACT = (DOER_PREAMBLE,) if DOER_PREAMBLE else ()
    config.CONFIG = load_config(project_dir)
    if args.backend:
        # CLI wins over the project's own [model] backend, the same way
        # --models wins over [worker] models. Written into CONFIG rather
        # than threaded through every call site, since run_editor_on_item
        # reads it from there anyway.
        config.CONFIG.setdefault("model", {})["backend"] = args.backend
    if args.lite_hard_num_predict:
        config.CONFIG.setdefault("model", {})["lite_hard_num_predict"] = args.lite_hard_num_predict
    if args.lite_hard_budget_directive:
        config.CONFIG.setdefault("model", {})["lite_hard_budget_directive"] = True
    if args.lite_hard_think_off:
        config.CONFIG.setdefault("model", {})["lite_hard_think_off"] = True
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
    if backend == "aider" and not (project_dir / ".aider.conf.yml").is_file():
        log("Warning: no .aider.conf.yml found in project dir - aider will use "
            "its own defaults, which may not be your local Qwen setup.", log_path)

    # A thinking-capable model spends its reasoning trace from the same
    # num_predict budget as its answer, so on an open-ended item it can
    # exhaust the budget mid-thought and never emit a file block at all -
    # which this loop can only report as a parked item. Measured on
    # qwen3.8:27b: 34 of 120 trials on an external benchmark, every one of
    # them looking like the model being careful rather than the harness
    # losing the output. Nobody should have to know this about their model
    # to get a usable run, so say it up front. Warn-only: the loop now
    # detects the truncation and retries with thinking disabled by itself,
    # and --lite-hard-think-off skips the wasted first attempt entirely.
    if backend == "lite":
        author = (args.models.split(",")[0].strip() if getattr(args, "models", None)
                  else cfg("model", "author"))
        if author and lite_editor.is_thinking_model(author) \
                and not cfg("model", "lite_hard_think_off", default=False):
            log(f"Note: {author} is a thinking-capable model. Its reasoning is spent from the "
                f"same output budget as the file it writes, so a long-reasoning item can be "
                f"truncated before any file block is emitted. This loop retries such an attempt "
                f"with thinking disabled; pass --lite-hard-think-off to skip the wasted attempt.",
                log_path)

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
        return 0

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
        if config.is_stop_requested():
            log("Stop requested by user. Terminating loop cleanly.", log_path)
            break
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

        update_item_status(todo_path, item, status)
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
        if args.sleep_between > 0:
            for _ in range(args.sleep_between):
                if config.is_stop_requested():
                    break
                time.sleep(1)
        continue

    log_run_summary(items_processed, parked, todo_path, log_path, runs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
