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

import datetime
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

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

_FILE_BLOCK_RE = re.compile(
    r"===\s*FILE:\s*(?P<path>[^\n=]+?)\s*===\r?\n(?P<content>.*?)(?:\r?\n)?===\s*END\s*===",
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


def _log(msg: str, log_path: Path | None) -> None:
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_path:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def build_prompt(project_dir: Path, item_text: str, files: list[str]) -> str:
    parts = [SYSTEM_PREAMBLE]
    for f in files:
        p = project_dir / f
        if p.is_file():
            parts.append(f"### Current content of {f}\n\n{p.read_text(encoding='utf-8', errors='replace')}")
        else:
            parts.append(f"### {f} does not exist yet - you are creating it.")
    parts.append(f"### Task\n\n{item_text}")
    # Always non-empty: run_lite_on_item refuses a call with no named
    # files rather than letting the model choose its own scope, which is
    # the entire point of this harness (see the module docstring).
    parts.append("### Files to write, and the only paths allowed in a FILE: marker:\n"
                 + "\n".join(f"- {f}" for f in files))
    return "\n\n".join(parts)


def call_ollama(model: str, prompt: str, url: str, num_ctx: int, num_predict: int,
                timeout: int, log_path: Path | None) -> str:
    """Streamed for the same reason spec_compiler.py streams: a
    non-streaming call to a model that reasons in plain prose produces
    zero visible output until it finishes or the socket times out,
    indistinguishable from a hang."""
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": True,
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
                print(piece, end="", flush=True)
                if obj.get("done"):
                    if obj.get("done_reason") == "length":
                        _log(f"[lite_editor] hit num_predict={num_predict} - output may be truncated", log_path)
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
                     num_predict: int = DEFAULT_NUM_PREDICT, timeout: int = DEFAULT_TIMEOUT
                     ) -> tuple[bool, str]:
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

    prompt = build_prompt(project_dir, item_text, files)
    _log(f"[lite_editor] requesting {files} from {model}", log_path)
    raw = call_ollama(model, prompt, ollama_url, num_ctx, num_predict, timeout, log_path)
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
