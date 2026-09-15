"""Test-driven development (TDD) workflow for forge."""

import re
import threading
from pathlib import Path

from . import worktree as wt
from .config import log
from .todo import STATUS_BLOCKED, STATUS_DONE, STATUS_NEEDS_REVIEW
from .validator import (
    changed_files,
    check_syntax_only,
    detect_prompt_leakage,
    path_matches_any,
    restore_unnamed_files,
    run_configured_commands,
    suspicious_new_paths,
    validate_syntax,
)

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


def run_tdd_phases(project_dir: Path, worktree_path: Path, task_text: str,
                   test_file: str, impl_file: str, base: str, args, log_path: Path,
                   record: dict, finish, model: str | None = None,
                   git_lock: threading.RLock | None = None,
                   backend_override: str | None = None) -> tuple[str, dict]:
    """Red, then green: aider writes ONLY the test and that test is
    confirmed to fail for a real reason, then a separate aider call
    implements the feature without being allowed to touch the test again.

    This is the version of "the tests are the check" (see README) that
    doesn't depend on the test having been written correctly by a human
    ahead of time: the test's own honesty is verified before it's ever
    trusted to grade anything, the same way a human reviewer would want
    to see a new test fail before believing it tests the right thing.
    """
    from .runner import (
        _accumulate_usage,
        git_head,
        run_editor_on_item,
        run_review_gate,
    )

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
    success, _, usage = run_editor_on_item(worktree_path, red_prompt, log_path,
                                           files=[test_file], model=model,
                                           backend_override=backend_override)
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
                                           files=[impl_file, test_file], model=model,
                                           backend_override=backend_override)
    _accumulate_usage(record, usage)
    if not success:
        return finish(STATUS_BLOCKED, "TDD green phase: aider itself failed")
    if getattr(args, "snapshot_first_attempt", False):
        # Red test + implementation, before the green-phase checks judge it.
        record["first_attempt_commit"] = wt.snapshot_first_attempt(worktree_path, record["branch"])

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
