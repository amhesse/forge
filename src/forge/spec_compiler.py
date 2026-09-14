#!/usr/bin/env python3
"""
spec_compiler.py

Turns a vague goal into checklist items shaped the way aider_loop.py's
own README says are reliable: one file per item, byte-exact content
where the target is data, an exact anchor line named for an insertion
into existing code. Everything in this file's system prompt is a rule
this project's real runs broke first and paid for:

  - Splitting a change across a consistency constraint fails no matter
    how good the work is. Asked to create `stories/the-night-build.story`
    and register it in `stories/index.json` as two separate items, the
    model wrote a structurally perfect story and it was rejected anyway,
    solely for not being in the index yet - the checker requires each
    file to agree with the other, so neither is valid alone. Recombined
    into one item, it merged on the first pass.
  - An edit into existing code needs a named anchor. "Add a check for X"
    got pasted over an unrelated check and deleted it, twice, on a 14B
    model that otherwise wrote working code quickly. Naming the exact
    line to insert before, and saying in as many words that this is a
    pure insertion, fixed it in one attempt, 18 seconds, zero retries.
  - A vague quantity ("at least 7 scenes, 2 endings, 2 locks") produces
    a model that satisfies them one at a time across retries, breaking
    an earlier one on each pass, and never converges. Enumerating the
    actual structure - which scenes, in which order, containing which
    exact lines - let the model fill in only the prose, and it passed
    first try.

This is a spec WRITER, not a task runner: it calls a reasoning-capable
model to draft items, validates their *shape* against aider_loop's own
parsers (a real backtick path, a real exact-content fence, one primary
file per item) and appends them to the checklist. It never touches
project files and never invokes aider - a human reads the draft before
aider_loop ever sees it.

Usage:
    python spec_compiler.py --project-dir ~/projects/blockroad \\
        --goal 'Add a story about a lost puppy finding its way home'

    python spec_compiler.py --project-dir ~/projects/blockroad \\
        --goal 'Add a check that art files are square' --dry-run

Requires: Ollama running locally with a reasoning-capable model (default:
qwen3.8-aider - the 27B model that writes bad code but reasons well
about structure; see README's note on why it's poor at driving aider
directly).
"""

import argparse
import json
import re
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

from . import aider_loop as al

# qwen3.8-aider (27B) was the obvious first choice for a planning role -
# it's the model this project has that's meant to reason, not just write
# code. It failed at drafting exactly the way it failed at driving aider
# (see README): asked only to draft checklist items, it spent 405 lines
# and its full num_predict budget deliberating about JavaScript edge
# cases in plain prose - never emitting a single "- [ ]" line. This is
# the same failure measured twice now in two different tasks, and it
# looks structural: aider's own config sets reasoning_tag: think for this
# model, but the Modelfile's TEMPLATE is bare `{{ .Prompt }}` with no
# chat formatting, so whatever training taught it to wrap deliberation in
# <think> tags is never actually triggered - the prose above reads like
# unwrapped thinking, not a considered answer. Fixing that is a Modelfile
# change, not this script's problem to route around.
#
# qwen25-coder-aider (14B) is the default instead. It has no reasoning
# framing at all, and in exchange it just answers: it wrote a
# structurally correct 8-scene story in 43 seconds and a correct,
# minimal src/check.js insertion in 18, both on the first attempt once
# given a well-shaped task. That's exactly the skill this script needs -
# turning a shaped task into text - even though a smaller model is
# presumably weaker at the *shaping* judgment a planner is supposed to
# bring. Measured behavior over presumed capability.
DEFAULT_MODEL = "qwen25-coder-aider"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_NUM_CTX = 32768
DEFAULT_TIMEOUT = 900  # seconds - a planning call, not a coding one, but a
                        # 27B model on a shared GPU can still take minutes,
                        # and this model has a measured habit of spending
                        # most of its budget on plain-prose deliberation
                        # before ever emitting an answer (see README: it
                        # burned 22k tokens explaining which files it
                        # would edit and wrote none of them). Streaming
                        # below is what makes that visible instead of a
                        # silent hang; this timeout is the last resort.
DEFAULT_NUM_PREDICT = 6000  # bounds exactly that failure mode. A drafted
                        # checklist is at most a few hundred lines; there
                        # is no legitimate reason for this call to need
                        # more than a few thousand tokens of output, and
                        # capping it turns "ran forever" into "produced a
                        # truncated draft, told you so, exit 1" - the
                        # latter costs seconds, not the whole timeout.

SYSTEM_PROMPT = textwrap.dedent("""\
    You write checklist items for an unattended coding loop. A small local
    model (not you) will read each item alone, with no memory of any other
    item, and try to satisfy it in one pass. Every rule below exists
    because a real run failed without it - follow all of them.

    1. ONE ITEM, ONE CHANGE THAT STANDS ALONE. An item must leave the
       project's tests passing by itself, with no other item applied
       either before or after it. If a change is only valid together with
       another file's change (a new file that must also be registered in
       an index, a function that must also be exported, a schema change
       that must also update every reader), put BOTH edits in the SAME
       item. Splitting them produces two items that are each rejected
       alone, even when both are written correctly - this failed on a
       real run and cost three full retries before it was diagnosed.

    2. NAME EVERY FILE THE ITEM SHOULD TOUCH, IN BACKTICKS, and no file it
       should not touch. Say "do not touch any other file" explicitly.
       Prefer one primary file per item; only add a second file per rule 1.

    3. FOR AN EDIT INTO EXISTING CODE, NAME THE EXACT ANCHOR - and only
       from a file whose actual current content appears below under
       "### <path> (current content)". If the file the goal describes
       editing is NOT shown there, do not guess or invent its contents:
       write the item as best you can without a specific anchor and add
       an HTML comment noting that the exact line needs a human to fill
       in, or skip it. An invented anchor line that doesn't exist in the
       real file is worse than an item with no anchor at all - it reads
       as precise and confident and is simply wrong, and nothing
       downstream can catch that before a commit is made against it. When
       the source IS shown below, copy the anchor line character-for-
       character from it - do not paraphrase - and say explicitly whether
       this is a pure insertion (no existing line moves, changes, or is
       deleted) or a replacement (name exactly what is being replaced). A
       vague instruction like "add a check for X" gets pasted over
       unrelated code and deletes it - this happened twice on a real run
       before the anchor was named.

    4. FOR A NEW FILE WITH STRUCTURE THAT MUST SATISFY RULES (a config, a
       data file, a document with required sections), ENUMERATE THE
       STRUCTURE THE MODEL MUST PRODUCE explicitly - the parts, their
       order, their names, and any required literal content - rather than
       describing it in prose ("at least N of these", "should include
       roughly that"). A vague quantity gets satisfied one constraint at a
       time across retries while breaking earlier ones, and never
       converges - this happened on a real run with a 5-way constraint.
       Only leave freeform prose room for content that genuinely has no
       required structure (flavor text, comments).

    5. WHERE THE TARGET IS DATA OR HAS ONE OBVIOUSLY CORRECT FORM (a
       config file, a JSON list, a fixed string), SPELL OUT THE EXACT
       CONTENT and introduce it with the word "exactly" immediately before
       a fenced code block. This is checked byte-for-byte, not inferred,
       and is the single most reliable item shape available.

    6. WRITE EACH ITEM AS ONE MARKDOWN CHECKLIST LINE: `- [ ] ...`. Only
       the first line carries the checkbox; everything after it (numbered
       sub-parts, fenced code blocks) is that same item's continuation and
       must NOT start with `- [ ]` itself, or the loop will parse it as a
       separate item.

    Output ONLY the checklist items, nothing else - no preamble, no
    explanation, no "Here are the items:". If you cannot write a
    responsible item for part of the goal (missing information, content
    that needs a human's judgment - like hand-drawn art or a subjective
    design call), skip that part rather than inventing detail, and note
    what's missing as an HTML comment (`<!-- ... -->`) instead of an item.
    """)


def build_prompt(goal: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"## Project context\n\n{context}\n\n"
        f"## Goal\n\n{goal}\n\n"
        f"## Checklist items\n\n"
    )


# A path-like token in the goal text, with or without backticks - looser
# than aider_loop's own FILE_PATH_RE (which requires backticks) because a
# goal is a human's one-line request, not a checklist item, and asking
# for backticks there is an easy rule to forget.
_GOAL_PATH_RE = re.compile(r"([\w./-]*[\w-]\.[A-Za-z][A-Za-z0-9]{0,7})")
# 16000 was the original guess here and turned out too conservative on
# real contact with a real project: voltagedrop's index.html - a single
# monolithic page, exactly the shape a small single-file project
# legitimately has - is 25.6KB, comfortably over that cap, and got
# refused inclusion for no real reason. The drafting model has a 32768
# token context window; even a dense 40KB of source is a fraction of
# that budget once the rest of the prompt (system rules, checklist,
# goal) is accounted for. 40000 leaves real headroom while still
# catching the case this exists for - a file that would actually
# dominate the prompt, like the 3MB CSV fixtures aider_loop's own
# preflight check warns about (see its DEFAULT_MAX_FILE_BYTES, which
# targets a much larger problem: files that blow the CODING model's
# context during editing, not the drafting model's context here).
_MAX_SOURCE_FILE_BYTES = 40000


def gather_context(project_dir: Path, todo_path: Path, goal: str,
                   max_chars: int = 24000) -> str:
    """Assembles what the planner needs to write grounded items: the
    existing checklist (so new items match its style and don't repeat
    finished work), any project docs that describe the format being
    worked in, and - critically - the actual current content of any file
    the goal names.

    That last part isn't an optimization, it's load-bearing: rule 3 asks
    the model to quote an anchor line "character-for-character from the
    source you were shown". Without the source actually being shown, a
    model that has never seen the real file invents a plausible-looking
    line instead - measured directly, a first version of this function
    omitted source files, and the model confidently quoted an anchor line
    that does not exist anywhere in the real file. That's not a smaller
    version of the failure this project already fixed once (aider pasting
    a new block over unrelated code); it's the same failure moved one
    step earlier, into the spec itself, where aider_loop's checks can't
    see it at all - a plausible commit against a nonexistent anchor either
    does nothing or corrupts the file, and nothing here would catch it
    before that.
    """
    parts = []

    if todo_path.is_file():
        parts.append(f"### Current checklist ({todo_path.name})\n\n{todo_path.read_text(encoding='utf-8')}")

    # Common names for "how this project's content is structured" docs.
    # Not a repo walk - only the project's own explicit ones, since a
    # matched-by-guess file is exactly the kind of thing that has burned
    # this project before (see aider_loop's read: block on .aider.conf.yml
    # for the same lesson in the other direction).
    for name in ("STORY_FORMAT.md", "FORMAT.md", "CONTRIBUTING.md", "docs/FORMAT.md"):
        p = project_dir / name
        if p.is_file():
            parts.append(f"### {name}\n\n{p.read_text(encoding='utf-8')}")

    # Any file the goal names, in full - see the docstring above for why
    # this is the important part, not an extra.
    seen = set()
    for m in _GOAL_PATH_RE.finditer(goal):
        rel = m.group(1)
        if rel in seen:
            continue
        p = project_dir / rel
        if not p.is_file():
            continue
        seen.add(rel)
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > _MAX_SOURCE_FILE_BYTES:
            parts.append(f"### {rel} (current content)\n\n"
                         f"(file is {size} bytes, too large to include in full - "
                         f"any anchor-line item for this file needs a human to "
                         f"supply the exact line instead)")
            continue
        parts.append(f"### {rel} (current content - quote anchor lines from here verbatim)\n\n"
                     f"```\n{p.read_text(encoding='utf-8')}\n```")

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... (truncated)"
    return text or "(no existing checklist, format docs, or named source files found)"


def call_ollama(model: str, prompt: str, url: str, num_ctx: int, num_predict: int,
                timeout: int) -> str:
    """Streamed, not a single blocking call: a non-streaming request to a
    model that reasons in plain prose (see DEFAULT_TIMEOUT's comment)
    produces zero output on the wire until it finishes or the socket
    times out - measured directly, a 300s non-streaming call to this
    model returned nothing at all, indistinguishable from a hang. With
    streaming, tokens print as they arrive, so a run that's burning its
    budget on deliberation is visible instead of silent, and num_predict
    below is the actual backstop.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                piece = obj.get("response", "")
                chunks.append(piece)
                print(piece, end="", flush=True, file=sys.stderr)
                if obj.get("done"):
                    if obj.get("done_reason") == "length":
                        print(f"\n\n[stopped: hit num_predict={num_predict} - draft is "
                              f"likely truncated; --dry-run and check, or raise "
                              f"--num-predict]", file=sys.stderr)
                    break
    except urllib.error.URLError as e:
        print(f"\nCould not reach ollama at {url}: {e}\n"
              f"Is `ollama serve` running, and is '{model}' pulled?", file=sys.stderr)
        sys.exit(1)
    print(file=sys.stderr)
    return "".join(chunks)


# A fenced fix-up: models asked for "output only the items" still sometimes
# wrap the whole answer in a single ```markdown ... ``` block. Strip only an
# outermost fence that wraps the ENTIRE response - never an inner one (an
# item's own "exactly:" spec block must survive untouched).
_OUTER_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```\s*$", re.DOTALL)


def strip_outer_fence(text: str) -> str:
    m = _OUTER_FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def parse_draft(raw: str) -> list[al.TodoItem]:
    """Reuses aider_loop's own checklist parser on the model's raw output,
    by writing it to the shape parse_todo() expects. This is the point of
    validating shape rather than trusting the model's claim to have
    followed the rules: the same regex that will govern how aider_loop
    reads these items back governs whether they parsed as items at all
    here."""
    lines = strip_outer_fence(raw).splitlines()
    return al.parse_todo_lines(lines)


# A line the item claims already exists in the file, following the
# phrase "existing line" - as either an inline `` `x` `` right after it,
# or (the shape models actually produced in practice) a fenced block a
# little further down, whose first non-blank line is the claimed anchor.
# Deliberately narrow to that specific phrase - false negatives here just
# mean a hallucinated anchor isn't caught, which is the status quo; false
# positives would block good items on a phrasing accident.
_EXISTING_LINE_RE = re.compile(r"existing line", re.IGNORECASE)
_ANCHOR_QUOTE_RE = re.compile(r"`([^`\n]+)`|```[a-zA-Z]*\n(.*?)```", re.DOTALL)


def validate_item(item: al.TodoItem, project_dir: Path) -> list[str]:
    """Deterministic shape checks - not a judgment on whether the item is
    a *good* task, only whether it has the shape the rules above asked
    for and that aider_loop can actually enforce. A missing shape is
    reported, not silently fixed: the human reviewing the draft should
    see exactly what the model skipped.

    Returns plain messages, for callers (like `forge draft`) that only
    ever warn. A caller that needs to tell "a real check is disabled"
    from "a heuristic nudge" - preflight_todo_rules, which can abort a
    run over this - should call fatal_problems() instead; see its
    docstring for why the two are not the same severity."""
    return [msg for msg, _fatal in _validate_item_typed(item, project_dir)]


def fatal_problems(item: al.TodoItem, project_dir: Path) -> list[str]:
    """The subset of validate_item()'s problems worth aborting a run
    over: only ones where a load-bearing check (touched-files, or the
    byte-exact check) is actually disabled, or where the item is
    concretely wrong (a hallucinated anchor line). Found by calibration:
    preflight_todo_rules used to treat every validate_item() problem as
    fatal, which meant a real, correctly 3-file item (rename_across_files,
    add_currency_field) was rejected before any model ever ran, on every
    single trial, for both models - the 3-or-more-files check is
    explicitly worded as "check this isn't just..." (a nudge, not a
    verdict) and forge draft's own use of the same function already only
    warns on it; preflight silently held it to a stricter standard than
    the function's own author did."""
    return [msg for msg, fatal in _validate_item_typed(item, project_dir) if fatal]


def _validate_item_typed(item: al.TodoItem, project_dir: Path) -> list[tuple[str, bool]]:
    problems: list[tuple[str, bool]] = []
    files = al.expected_files(item.text)
    if not files:
        problems.append(("names no file in backticks - the touched-files check "
                         "will have nothing to verify", True))
    specs = al.extract_exact_content_specs(item.text)
    has_fence = "```" in item.text
    if has_fence and not specs and "exactly" not in item.text.lower():
        problems.append(("has a fenced code block but doesn't say \"exactly\" before it - "
                         "the byte-exact check won't fire, so this block is unverified prose", True))
    if len(files) > 2:
        problems.append((f"names {len(files)} files - rule 1 allows two only when they must "
                         f"agree with each other; check this isn't just an under-scoped item", False))

    # The one check that would have caught this session's own real
    # near-miss: a drafted item claimed an anchor line that does not exist
    # anywhere in the file it names. Checked against the file as it
    # actually is on disk right now, not against whatever the model was
    # shown - the ground truth that matters is what aider will see.
    for phrase_match in _EXISTING_LINE_RE.finditer(item.text):
        # Look for the first quoted line/block within 200 chars after the
        # phrase - close enough to be "the line just mentioned", far
        # enough to survive "which appears once:\n  ```js\n".
        window = item.text[phrase_match.end():phrase_match.end() + 200]
        quote_match = _ANCHOR_QUOTE_RE.search(window)
        if not quote_match:
            continue
        block = quote_match.group(1) if quote_match.group(1) is not None else quote_match.group(2)
        # A fenced match may have grabbed the whole block; only the first
        # non-blank line is the claimed anchor, the rest is new code.
        anchor = next((ln.strip() for ln in block.splitlines() if ln.strip()), "")
        if not anchor:
            continue
        found_in_any = False
        for f in files:
            p = project_dir / f
            if p.is_file() and anchor in p.read_text(encoding="utf-8", errors="ignore"):
                found_in_any = True
                break
        if not found_in_any and files:
            problems.append((f"claims an existing line {anchor!r}, but it doesn't appear "
                             f"in {files} as they exist right now - this may be a "
                             f"hallucinated anchor (measured once for real: it produces "
                             f"a confident, wrong item with no anchor that actually exists)", True))
    return problems


def main():
    parser = argparse.ArgumentParser(description="Draft aider_loop checklist items from a goal")
    parser.add_argument("--project-dir", required=True, type=str)
    parser.add_argument("--todo-file", default="TODO.md", type=str)
    parser.add_argument("--goal", required=True, type=str,
                         help="What to accomplish, in plain language")
    parser.add_argument("--model", default=DEFAULT_MODEL, type=str,
                         help=f"Ollama model to draft with (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, type=str)
    parser.add_argument("--num-ctx", default=DEFAULT_NUM_CTX, type=int)
    parser.add_argument("--num-predict", default=DEFAULT_NUM_PREDICT, type=int,
                         help="Hard cap on drafting output length, so a model that "
                              "reasons in plain prose (see README) can't run unbounded")
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT, type=int)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the drafted items and their validation instead of "
                              "appending them to the checklist")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    todo_path = project_dir / args.todo_file
    if not project_dir.is_dir():
        print(f"Project dir does not exist: {project_dir}", file=sys.stderr)
        sys.exit(1)

    context = gather_context(project_dir, todo_path, args.goal)
    prompt = build_prompt(args.goal, context)

    print(f"Drafting with {args.model} ({len(prompt)} chars of prompt)...\n", file=sys.stderr)
    raw = call_ollama(args.model, prompt, args.ollama_url, args.num_ctx,
                      args.num_predict, args.timeout)
    if not raw.strip():
        print("Model returned nothing. Is the model name right, and did it load within "
              "--timeout?", file=sys.stderr)
        sys.exit(1)

    items = parse_draft(raw)
    if not items:
        print("Model output didn't parse as any checklist items. Raw output:\n", file=sys.stderr)
        print(raw, file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(items)} item(s) drafted:\n")
    any_problems = False
    for i, item in enumerate(items, 1):
        print(f"{i}. {item.render()}")
        problems = validate_item(item, project_dir)
        if problems:
            any_problems = True
            for p in problems:
                print(f"   ⚠ {p}")
        print()

    if args.dry_run:
        print("(--dry-run: nothing written)", file=sys.stderr)
        return

    if any_problems:
        print("Some items have shape warnings above. They are still appended - a warning "
              "names a rule the model didn't follow, not a certainty the item will fail; "
              "read the flagged item(s) before running the loop.", file=sys.stderr)

    with todo_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n\n".join(f"{item.indent}- [{item.status}] {item.text}" for item in items) + "\n")
    print(f"Appended {len(items)} item(s) to {todo_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
