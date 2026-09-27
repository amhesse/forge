#!/usr/bin/env python3
"""
lite_editor.py

A minimal, purpose-built replacement for calling aider from aider_loop.py.
Same job - hand a model a task and named files, get real edits back - but
built around three specific failures aider produced on this project's own
real runs, rather than aider's general-purpose chat/architect/repo-map
machinery, none of which this loop ever uses (every item runs headless,
one message, no chat history, no multi-turn negotiation).

Each failure below is why this exists, not a guess at what might go wrong:

  1. SEARCH/REPLACE brittleness. aider's default `diff` edit format asks
     the model to emit a block that must match the current file
     character-for-character, including whitespace. Measured on a real
     item: the model's SEARCH block used spaces, the real file used tabs,
     and the whole edit was silently discarded ("SearchReplaceNoExactMatch")
     with three reflections burned finding that out.
  2. The "which files do you want me to add?" reflection loop. Without a
     file already open in aider's chat, a model - especially one that
     reasons in plain prose - answers that question instead of editing,
     sometimes for its entire token budget (qwen3.8:27b: 12k tokens sent,
     1.6k received, zero edits, three reflections spent explaining which
     files it *would* open).
  3. Garbage filenames from a malformed edit block. aider infers a target
     file's name from surrounding text when a SEARCH/REPLACE block doesn't
     parse; a model's own reasoning text ("File Listing: stories/index.json")
     has twice become an actual filename this way.

The fix for all three turns out to be the same idea: never ask the model
to describe *where* to write something, and never ask it to produce a
*diff*. Every file this harness will touch is decided in Python before
the model is ever called (exactly like aider_loop's own expected_files()
parsing already does for --file) - so there is no "which files" question
to answer. And the model is asked for the file's ENTIRE new content, not
a diff against it - the one thing this whole project has repeatedly
measured local models being reliably good at ("Local models transcribe
well and decide badly," per aider_loop's README, is precisely why
byte-exact items are the most reliable shape available). Diffing is
compute Python already does exactly correctly; there's no reason to ask
a model to attempt it in natural-language-adjacent SEARCH/REPLACE syntax
when the alternative is "rewrite the file, we'll diff it".

Output contract, deliberately not natural language: a model can still
write a paragraph of reasoning outside the markers (harmless, ignored),
but the actual content of each file must appear inside a sentinel block
this parses with a fixed regex, not inferred from context the way aider's
"the filename must be on the line above the fence" convention is:

    ===FILE: path/relative/to/project===
    <the file's complete new content, nothing else>
    ===END===

One block per file, any number of files, in any order, anywhere in the
response. A block naming a file NOT in the `files` this call was given is
never written - not sanitized, not repaired, just ignored - so an
invented or hallucinated path structurally cannot reach disk. That
removes the class of bug restore_unnamed_files() exists to clean up
after; there is no "after" to clean up, because nothing outside the
agreed file set can ever be considered.
"""

import ast
import datetime
import json
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

# Ollama's own token counts for the most recent call THIS THREAD made -
# not a global, because --models runs one worker per thread and a shared
# mutable dict would race between them. Read via last_usage() right after
# a call_ollama() completes in the same thread; there's no queue or
# history, just "what did the call I just made cost".
_usage_local = threading.local()


def last_usage() -> dict:
    """{'prompt_tokens', 'completion_tokens', 'seconds'} for the most
    recent call_ollama() on this thread, or zeros if none yet / the call
    failed before Ollama returned its final chunk. Real usage from
    Ollama's own response, not an estimate - `prompt_eval_count` and
    `eval_count` on the final streamed chunk."""
    return getattr(_usage_local, "usage", {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_NUM_CTX = 32768
DEFAULT_NUM_PREDICT = 8000  # a rewritten file plus reasoning; bounded for
                             # the same reason spec_compiler.py bounds its
                             # own drafting calls - a model that reasons in
                             # plain prose before answering (see README)
                             # can otherwise spend its entire budget doing
                             # that and never reach a single ===FILE===
                             # block. Capped, not silently allowed to run.
DEFAULT_TIMEOUT = 600
VERBOSE = False

# A block's content ends at whichever comes first: a real ===END===, the
# start of the NEXT ===FILE:=== block (matched via lookahead, so it stays
# unconsumed for finditer's own next match), or the end of the response.
# That last branch matters for real, not hypothetically: a long-form
# generation (a short story, not a config file) reliably finished the
# actual content and then simply never emitted ===END=== at all -
# measured directly, num_predict was nowhere near hit (no truncation
# warning), the model just had no natural "I'm done, close the tag" cue
# the way code's own closing brace gives it. Without this fallback, a
# complete, correct story was discarded as unparseable and retried into
# producing the exact same gap again.
_FILE_BLOCK_RE = re.compile(
    r"===\s*FILE:\s*(?P<path>[^\n=]+?)\s*===\r?\n"
    r"(?P<content>.*?)"
    r"(?:\r?\n===\s*END\s*===|(?=\r?\n===\s*FILE:)|\Z)",
    re.DOTALL,
)

SYSTEM_PREAMBLE = """\
You are editing files in a project. You will be told exactly which
file(s) to write and given their current content, if they already exist.

Output the COMPLETE new content of every file you write, wrapped exactly
like this - nothing else may appear on the marker lines themselves:

===FILE: relative/path/to/file.ext===
<the file's entire content, from the first line to the last>
===END===

Rules:
- Write out the WHOLE file, not a diff or a snippet - every line that
  should exist in the final file, including lines you are not changing.
- NEVER use placeholders or abbreviation comments like "... existing content ...",
  "/* ... existing code ... */", or "// ... rest of code unchanged ...".
  Every single line of the file must be present in your output.
- One ===FILE:===...===END=== block per file. Only write files you were
  told to write; do not create, rename, or delete any other file.
- The path after "FILE:" must be EXACTLY one of the paths you were given
  below - not a new name, not a description of the file.
- You may write reasoning or explanation outside the blocks if you find
  it helpful, but the actual file content must be inside a block - text
  outside a block is never read as file content.
- Do NOT wrap the content in markdown fences. The ===FILE:===/===END===
  markers already delimit it; a ``` line would be written into the file
  as a literal line and break it.
"""


# Appended to the prompt only for items the caller flags as hard, and
# only when a project opts in. Measured on the polyglot-subset benchmark
# (Leash Phase 4, qwen3.8:27b): on book-store the model spent all 8000
# num_predict tokens reasoning through candidate algorithms out loud and
# never emitted a ===FILE:=== marker at all - 15 trials out of 15,
# done_reason="length" every time, so run_lite_on_item saw "no parseable
# block" and the item parked without a single line of code ever being
# written. The directive is deliberately the OPPOSITE of "think harder":
# the failure is not too little reasoning, it is reasoning that never
# terminates into output.
OUTPUT_BUDGET_DIRECTIVE = """\
### Output budget - read this before you begin

Your output length is capped. Reasoning that runs past the cap means the
file is never written and the entire attempt is discarded - not graded
as wrong, discarded, as if you had said nothing.

- Write the ===FILE:=== block FIRST, before any discussion.
- Do not compare alternative approaches in your output. Pick the one you
  are most confident in and implement it.
- If something needs qualifying, put it in a code comment inside the
  file, not in prose outside the block.
"""


def _log(msg: str, log_path: Path | None) -> None:
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_path:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


# Bound on how much imported-module context one call carries, in files
# and in bytes-per-file. This is read-only orientation, not editable
# scope (see find_import_context's docstring for why it's needed at
# all) - it should be enough for the model to see a class's real
# attributes, not enough to dominate the context window or make a
# rewrite-length item's num_predict budget too tight.
MAX_IMPORT_CONTEXT_FILES = 4
MAX_IMPORT_CONTEXT_BYTES = 4000


def _resolve_import(project_dir: Path, dotted: str) -> Path | None:
    """dotted module name -> the project file it names, or None if it
    isn't one (stdlib, a third-party package, anything not on disk here).
    Tried against the project root and against `src/`, the two layouts
    this project's own items and this project itself use."""
    rel = Path(*dotted.split("."))
    for root in (project_dir, project_dir / "src"):
        candidate = root / rel.with_suffix(".py")
        if candidate.is_file():
            return candidate
        candidate = root / rel / "__init__.py"
        if candidate.is_file():
            return candidate
    return None


# A backtick-quoted dotted path: `ledger.store.Ledger`, `ledger.store.Ledger.add`.
# Matched even though the trailing segment is usually a class or method
# name, not a module - _text_dotted_modules() below strips segments from
# the right until something resolves, so "Ledger" or "Ledger.add" falls
# away and "ledger.store" (the actual module) is what's tried.
_DOTTED_REF_RE = re.compile(r"`((?:[A-Za-z_][A-Za-z0-9_]*\.){1,}[A-Za-z_][A-Za-z0-9_]*)")


def find_import_context(project_dir: Path, files: list[str], item_text: str = "") -> dict[str, str]:
    """{path: content} for project files a file in `files` imports, or
    that the item text names by dotted reference - read-only and
    separate from the writable `files` set.

    Exists because lite_editor's whole-file-rewrite contract (see the
    module docstring) only ever shows the model the file(s) it was told
    to write - it never sees a file a TDD test imports, or a class it
    calls into. Measured directly: every TDD item failed (0/6 across two
    models) because the model wrote `entry.amount` for a field that's
    actually `entry.cents` - it had never been shown the `Entry`
    dataclass it was writing against, only told it existed by name in the
    task text.

    Two sources, because one alone misses real cases:
    - AST-parsing `files`' own imports catches an *existing* file editing
      into a dependency it already names in code.
    - Scanning `item_text` for backtick-quoted dotted references (e.g.
      `ledger.store.Ledger`) is what actually covers TDD: both files
      being written are new, so there is no on-disk import statement to
      parse yet - the class name only ever appears in the task's prose.

    Bugfix and multi-file items hit the same gap on the AST side, just
    less reliably - so both sources run for every call, not only TDD.

    Best-effort and Python-only: a reference or import this can't resolve
    to an on-disk project file (stdlib, third-party, a plain method name
    with no matching module, a syntax error in the source being scanned)
    is silently skipped rather than guessed at - the read-only-context
    idea only helps if what it shows is real.
    """
    context: dict[str, str] = {}
    seen: set[str] = set()

    def consider(mod: str) -> bool:
        """Returns True if `mod` names a real project file, whether or
        not it ended up added (already in `files`/context still counts,
        so a text reference's shorter fallback prefixes aren't tried
        once the real module is found)."""
        if mod in seen:
            return False
        seen.add(mod)
        if len(context) >= MAX_IMPORT_CONTEXT_FILES:
            return False
        resolved = _resolve_import(project_dir, mod)
        if resolved is None:
            return False
        rel_path = resolved.relative_to(project_dir).as_posix()
        if rel_path in files or rel_path in context:
            return True
        content = resolved.read_text(encoding="utf-8", errors="replace")
        context[rel_path] = content[:MAX_IMPORT_CONTEXT_BYTES]
        return True

    for f in files:
        p = project_dir / f
        if p.suffix != ".py" or not p.is_file():
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    consider(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 0:
                    consider(node.module)
                else:
                    # `from .foo import Bar` etc - resolve relative to
                    # this file's own package, not the project root.
                    pkg = Path(f).parent.parts
                    up = node.level - 1
                    base = pkg[: len(pkg) - up] if up else pkg
                    consider(".".join((*base, *node.module.split("."))))

    for ref in _DOTTED_REF_RE.findall(item_text):
        if len(context) >= MAX_IMPORT_CONTEXT_FILES:
            break
        segments = ref.split(".")
        for i in range(len(segments), 0, -1):
            if consider(".".join(segments[:i])):
                break

    return context


def build_prompt(project_dir: Path, item_text: str, files: list[str],
                 context: dict[str, str] | None = None,
                 budget_directive: bool = False) -> str:
    parts = [SYSTEM_PREAMBLE]
    if budget_directive:
        parts.append(OUTPUT_BUDGET_DIRECTIVE)
    for f in files:
        p = project_dir / f
        if p.is_file():
            parts.append(f"### Current content of {f}\n\n{p.read_text(encoding='utf-8', errors='replace')}")
        else:
            parts.append(f"### {f} does not exist yet - you are creating it.")
    for f, content in (context or {}).items():
        parts.append(f"### Read-only context: {f} (imported by a file above - "
                     f"for reference only, do NOT write a FILE: block for it)\n\n{content}")
    parts.append(f"### Task\n\n{item_text}")
    # Always non-empty: run_lite_on_item refuses a call with no named
    # files rather than letting the model choose its own scope, which is
    # the entire point of this harness (see the module docstring).
    parts.append("### Files to write, and the only paths allowed in a FILE: marker:\n"
                 + "\n".join(f"- {f}" for f in files))
    return "\n\n".join(parts)


def call_ollama(model: str, prompt: str, url: str, num_ctx: int, num_predict: int,
                timeout: int, log_path: Path | None, think: bool | None = None) -> str:
    """Streamed for the same reason spec_compiler.py streams: a
    non-streaming call to a model that reasons in plain prose produces
    zero visible output until it finishes or the socket times out,
    indistinguishable from a hang."""
    body = {
        "model": model, "prompt": prompt, "stream": True,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict},
    }
    # On a thinking-capable model the reasoning trace is spent from the
    # same num_predict budget as the answer, so an open-ended item can
    # exhaust the budget mid-thought and emit no ===FILE:=== block at
    # all. Measured on book-store (polyglot-subset, qwen3.8:27b): 8000
    # tokens and 16000 tokens both ended done_reason="length" with no
    # marker written, so this is not a budget that can simply be raised.
    # Left as None (the model's own default) unless a caller asks.
    if think is not None:
        body["think"] = think
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    chunks = []
    _usage_local.usage = {"prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                piece = obj.get("response", "")
                chunks.append(piece)
                if VERBOSE:
                    print(piece, end="", flush=True)
                if obj.get("done"):
                    if obj.get("done_reason") == "length":
                        _log(f"[lite_editor] hit num_predict={num_predict} - output may be truncated", log_path)
                    # Real counts from Ollama's own final chunk, not an
                    # estimate - present on every "done" response this
                    # server version sends, absent (and left at 0) only
                    # if the connection died before one arrived.
                    _usage_local.usage = {
                        "prompt_tokens": obj.get("prompt_eval_count", 0),
                        "completion_tokens": obj.get("eval_count", 0),
                        "seconds": obj.get("total_duration", 0) / 1e9,
                    }
                    break
    except urllib.error.URLError as e:
        _log(f"[lite_editor] could not reach ollama at {url}: {e}", log_path)
        return ""
    print()
    return "".join(chunks)


# A markdown fence wrapping an ENTIRE block's content. Models habitually
# fence code even when the surrounding protocol already delimits it, so
# without this the literal line ```python lands as the first line of the
# file. Measured on the first real run of this harness: every file came
# out fenced, which a byte-exact item caught immediately and a Python
# item caught one step later as a SyntaxError.
_WRAPPING_FENCE_RE = re.compile(r"\A```[a-zA-Z0-9_+-]*[ \t]*\r?\n(?P<inner>.*)\r?\n```[ \t]*\Z", re.DOTALL)


def strip_wrapping_fence(content: str) -> str:
    """Remove a markdown fence that wraps the whole content.

    Only an outermost fence that opens on the very first line and closes
    on the very last one is removed, so a file that legitimately
    CONTAINS fenced blocks (a README, this project's own TODO.md) keeps
    them - the fences inside it don't start at position zero. The one
    case this gets wrong is a markdown file whose entire content is a
    single fenced block and nothing else; that's rare enough, and a
    byte-exact item would catch it, which is more than could be said for
    leaving every file fenced.
    """
    m = _WRAPPING_FENCE_RE.match(content.strip("\n"))
    return m.group("inner") if m else content


def parse_file_blocks(raw: str) -> dict[str, str]:
    """Returns {path: content}. A path is used exactly as written on the
    FILE: line - callers are responsible for restricting to the agreed
    file set (see run_lite_on_item) rather than trusting every block."""
    return {m.group("path").strip(): strip_wrapping_fence(m.group("content"))
            for m in _FILE_BLOCK_RE.finditer(raw)}


def run_lite_on_item(project_dir: Path, item_text: str, log_path: Path,
                     files: list[str] | None = None, model: str | None = None,
                     ollama_url: str = DEFAULT_OLLAMA_URL, num_ctx: int = DEFAULT_NUM_CTX,
                     num_predict: int = DEFAULT_NUM_PREDICT, timeout: int = DEFAULT_TIMEOUT,
                     budget_directive: bool = False,
                     think: bool | None = None) -> tuple[bool, str]:
    """Drop-in alternative to aider_loop.run_aider_on_item - same
    signature, same (success, output) return shape, so aider_loop's
    process_item can call either one interchangeably based on config.

    `success` means the model produced at least one usable, in-scope
    file block and it was written and committed - not that the change is
    correct (same distinction run_aider_on_item makes; that's still
    aider_loop's validation step's job). A block naming a file outside
    `files` is silently dropped, not written and not treated as an
    error - see the module docstring for why that's a feature (an
    invented path structurally cannot reach disk) rather than something
    to warn about on every call.
    """
    if not files:
        _log("[lite_editor] called with no target files - nothing this harness can scope "
             "itself to; refusing rather than guessing", log_path)
        return False, "lite_editor requires at least one named file"
    if not model:
        _log("[lite_editor] called with no model - this harness always needs one explicitly "
             "(unlike aider, it has no .aider.conf.yml to fall back to)", log_path)
        return False, "lite_editor requires an explicit model"

    context = find_import_context(project_dir, files, item_text=item_text)
    prompt = build_prompt(project_dir, item_text, files, context=context,
                          budget_directive=budget_directive)
    if context:
        _log(f"[lite_editor] read-only context from imports: {list(context)}", log_path)
    _log(f"[lite_editor] requesting {files} from {model}", log_path)
    raw = call_ollama(model, prompt, ollama_url, num_ctx, num_predict, timeout, log_path,
                      think=think)
    if not raw.strip():
        return False, "model returned nothing"

    blocks = parse_file_blocks(raw)
    in_scope = {path: content for path, content in blocks.items() if path in files}
    out_of_scope = [p for p in blocks if p not in files]
    if out_of_scope:
        _log(f"[lite_editor] ignored block(s) for file(s) not in scope: {out_of_scope}", log_path)
    if not in_scope:
        _log(f"[lite_editor] no ===FILE:=== block matched any of {files}. Raw output:\n{raw[:2000]}", log_path)
        return False, f"no parseable block for any of {files}\n\nraw output:\n{raw[:4000]}"

    written = []
    for path, content in in_scope.items():
        p = project_dir / path
        p.parent.mkdir(parents=True, exist_ok=True)
        # Model output is one edit behind a trailing newline convention
        # half the time either way; normalize to exactly one, the same
        # way content_mismatches() normalizes for the byte-exact check.
        p.write_text(content.rstrip("\n") + "\n", encoding="utf-8")
        written.append(path)

    add_result = _git(["add", "-A", "--", *written], project_dir)
    if add_result.returncode != 0:
        return False, f"git add failed: {add_result.stderr}"
    status = _git(["status", "--porcelain", "--", *written], project_dir)
    if not status.stdout.strip():
        _log(f"[lite_editor] wrote {written} but git reports no change - nothing to commit", log_path)
        return False, "model's output was identical to the current content; nothing changed"
    commit_result = _git(["commit", "-q", "-m", f"lite_editor: {item_text.splitlines()[0][:72]}"], project_dir)
    if commit_result.returncode != 0:
        return False, f"git commit failed: {commit_result.stderr}"

    _log(f"[lite_editor] wrote and committed {written}", log_path)
    return True, raw
