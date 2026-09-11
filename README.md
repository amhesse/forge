# aider-loop

Runs [aider](https://aider.chat) unattended against a markdown checklist, with
a safety net between each item and the next.

The loop itself is simple: take the next unchecked item, hand it to aider,
check what came back, mark it and move on. The value is in the checking. A
local model will confidently produce work that is wrong in ways that still
compile, and the checks below exist because each one caught a real failure
that the previous checks missed.

    aider_loop.py --project-dir ~/projects/thing --todo-file TODO.md

## What it checks, in order

1. **Did aider crash?** Retried `--max-retries` times.
2. **Byte-exact content match.** If an item spelled out a file's target
   content (see *Writing items* below), the result is compared to it
   byte-for-byte. This is the only check that is not a heuristic, and it is
   trusted first.
3. **Were the named files touched?** An item naming `` `src/thing.py` ``
   whose commit didn't touch that file is marked needs-review.
4. **Prompt leakage.** Aider sometimes writes its own instructions into the
   file instead of content. Known markers are an instant revert.
5. **Garbage filenames.** A failed SEARCH/REPLACE can become a new file
   named after the model's reasoning text. Also an instant revert.
6. **Syntax**, over the changed files only, plus whatever `[validate]
   commands` the project declares.
7. **Second opinion** from a different local model, if `--review-model` is
   set.

Anything worse than "fine" stops the run rather than letting the next item
build on top of an unreviewed change. Failures are reverted to the
pre-item commit; softer outcomes are left in place and marked.

Item status markers: `[ ]` open, `[x]` done, `[!]` blocked and reverted,
`[?]` needs review.

## Writing items

The safety net is only as good as what it can infer from the item text, and
two conventions turn checks on. **An item that follows neither convention
still runs, but gets almost no verification** — which is worse than none,
because it looks like it worked.

**Name target files in backticks** to enable the touched-files check:

```markdown
- [ ] In `src/report.py`, add a `--json` flag that prints the summary as JSON.
```

Only backtick paths the item should *edit*. A path mentioned as a reference
("match the style in `src/other.py`") will be treated as an edit target and
the item will be marked needs-review for not touching it. Write reference
paths without backticks.

**Say "exactly:" before a fenced block** to enable byte-exact checking:

```markdown
- [ ] Replace the entire contents of `config.toml` with exactly:
```

This is by far the most reliable mode. Local models transcribe well and
decide badly; exact specs play to that.

An item's text runs from its checkbox line to the next checkbox, so fenced
blocks and multi-line prose are included. An HTML comment (`<!-- ... -->`)
also ends an item — useful for notes between items, but don't put one
*inside* one.

## Configuration

Everything project-specific lives in `.aiderloop.toml` in the project. The
file is optional; without it the loop uses inert defaults.

```toml
[model]
author = "qwen25-coder-aider"   # ollama name, no "ollama/" prefix.
                                # Unloaded before a review call so the
                                # reviewer isn't queued behind its VRAM.
review  = "qwen2.5-coder:7b"    # default for --review-model

[validate]
checks   = ["python", "js-html"]           # built-ins; omit to enable both
commands = [".venv/bin/pytest tests -q"]   # this project's own gate

[preflight]
max_file_bytes = 262144
```

`checks` are syntax checkers that run **only over the files an item
changed**. `commands` are project-wide and always run in full — a test
failure in an untouched module is real signal, a syntax error in an
untouched file is not.

Aider's own model settings still come from the project's `.aider.conf.yml`;
this file doesn't replace it.

## .aiderignore is not optional

Aider builds its repo map by walking the working tree, so committed data
files dominate the context. On one real project ~3MB of CSV fixtures and a
6MB generated HTML report produced an estimated context of **1,584,443
tokens** against a 32,768 limit — for a task touching one template — and
killed two runs by exhausting system memory.

The preflight names files over `max_file_bytes` that aren't ignored. Put
test fixtures, generated output, and aider's own history in `.aiderignore`:

```
tests/golden/*.csv
out/
.aider.chat.history.md
.aider.tags.cache.v4/
```

## Requirements

Python 3.11+ (stdlib only), `aider`, `git`, and `ollama` if using local
models. `node` is optional and enables JS/HTML syntax checking.

## What it is not

This does not make a local model trustworthy. In measured use it was
reliable on exact-content items and needed correction on **every**
open-ended one — the failures being confident, plausible, and wrong rather
than obviously broken. The review pass has also approved diffs containing
real defects, especially when the reviewer shares a family with the author.

Read the diffs.
