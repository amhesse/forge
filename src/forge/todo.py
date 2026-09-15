"""Checklist parsing, status management, and preflight checks for forge."""

import re
import sys
from pathlib import Path

from .config import log

STATUS_OPEN = " "
STATUS_DONE = "x"
STATUS_BLOCKED = "!"
STATUS_NEEDS_REVIEW = "?"  # aider produced *something* and it built, but an
# automated check couldn't confirm it actually satisfies the task - treated
# like done/blocked in that next_open_item() won't retry it, but the run
# stops here instead of continuing, so a human/Claude looks at it before
# anything else gets built on top of an unconfirmed change.

CHECKBOX_RE = re.compile(r"^(?P<indent>\s*)-\s\[(?P<mark>[ xX!?])\]\s(?P<text>.+)$")

# A checkpoint marker (an HTML comment, not a checkbox) ends the preceding
# item's continuation-line collection in parse_todo() the same way the next
# checkbox does.
_CHECKPOINT_RE = re.compile(r"^\s*<!--")


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


def update_item_status(todo_path: Path, item: TodoItem, status: str) -> None:
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
        sys.exit(1)
    elif any_problems:
        log("(the warning(s) above are advisory - the run is proceeding)", log_path)
