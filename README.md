# aider-loop

Runs [aider](https://aider.chat) unattended against a markdown checklist, with
a safety net between each item and the next.

The loop itself is simple: take the next unchecked item, hand it to aider,
check what came back, mark it and move on. The value is in the checking. A
local model will confidently produce work that is wrong in ways that still
compile, and the checks below exist because each one caught a real failure
that the previous checks missed.

Every item runs in its own `git worktree`, branched from the project's
current HEAD. Only an item that passes every check is merged back. A
failure is never applied at all, so there is nothing to undo and no reason
to stop — the run keeps going, and each parked item's work is left on its
own branch to read in the morning.

    aider_loop.py --project-dir ~/projects/thing --todo-file TODO.md

## What it checks, in order

1. **Did aider crash?** Retried `--max-retries` times.
   Then, if the item names files, **any other file it changed is put back**
   (and a new one deleted) before anything else runs, so the checks below
   judge the task alone. Turn off with `[scope] restore_unnamed = false`.
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

Anything worse than "fine" is *parked*: the item's branch is left behind,
the project checkout never sees it, and the next item starts from the same
base. Nothing is reverted, because nothing was ever applied.

Item status markers: `[ ]` open, `[x]` done and merged, `[!]` blocked,
`[?]` needs review.

## Worktrees, and what happens to a failed item

The loop used to run items directly in the project checkout. That forced
two behaviours it could never get out of: a failure had to be undone with
`git reset --hard` in the tree you were working in, and the run had to
*stop* at the first unreviewed change so the next item wouldn't build on
top of it. Both are the same problem — there was only one working tree, so
"keep going" and "don't build on this" were in conflict. Reverting was
also the risky half: it wrote to your tree, and it destroyed the evidence.

Now each item gets a throwaway checkout on branch `aider-loop/item-N-<date>`.
Everything happens in there. A passing item fast-forwards the real branch;
a failing one is simply never merged. At the end of a run you get:

```
2 item(s) did not pass and were never applied to the project.
  [?] aider-loop/item-2-20260911-231021 - named file(s) ['never_touched.py'] were never touched
  [!] aider-loop/item-3-20260911-231021 - validation failed
```

Read one with `git log -p aider-loop/item-2-...`, and `git branch -D` it
when you're done. A passing item's branch is deleted automatically
(`--keep-branches` keeps it). Worktrees live under
`~/.cache/aider-loop/<project>/`, outside the repo so aider's repo map and
the preflight scan never walk them; each run also writes one JSON file per
item there recording the branch, the files touched, and why it landed
where it did.

`--stop-on-problem` restores the old halt-at-the-first-failure behaviour.
It's off by default now: a failed item can't reach the project, so there
is nothing to stop for.

### What the loop requires of your checkout

A worktree is populated from a commit, so it contains tracked files and
nothing else. Gitignored config that aider needs — `.aider.conf.yml`,
`.aider.model.settings.yml`, `.aiderignore`, `.aiderloop.toml` — is copied
in, and `.venv` / `node_modules` are symlinked rather than copied (one of
these projects has a 317MB venv; duplicating it per item is not a thing to
do). Without that copy step a project whose aider config is gitignored
would run against whatever `~/.aider.conf.yml` says, silently.

The run also refuses to start if the checkout has uncommitted changes to
tracked files, because `git merge --ff-only` won't run over them — better
to hear that now than after an item has spent half an hour in the model.
Your checklist and `aider_loop.log` are excused: both are the loop's own
bookkeeping, neither can reach a merge, and being told to commit your todo
list before the loop will read it is backwards.

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
                                # Used to check its context size.
edit_format = "diff"            # passed to aider as --edit-format.
                                # The default; overrides .aider.conf.yml.

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

## Edit format: diff by default

The loop passes `--edit-format diff` to aider, overriding whatever the
project's `.aider.conf.yml` says. `whole` was used first. On
qwen2.5-coder:14b it repeatedly left out the filename line before a file
block, so aider applied nothing. It also costs the file twice out of the
context window (once read, once rewritten), capping a 32k context at roughly
800 lines of Python per item. With `diff` the model only emits the changed
hunk, and a failed SEARCH match is caught by the garbage-filename and
touched-files checks above.

Set `edit_format = "whole"` under `[model]` to go back for a model that
handles SEARCH/REPLACE badly. Either way, keep items to one file where you
can.

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

`git` is now a hard requirement rather than a warning — per-item worktrees
are what keep a failed item away from your checkout, and there is no
sensible way to degrade that. The old non-git path ran items directly in
the tree and left broken changes in place for the next item to build on,
which is exactly what worktrees replaced.

## What it is not

This does not make a local model trustworthy. In measured use it was
reliable on exact-content items and needed correction on **every**
open-ended one — the failures being confident, plausible, and wrong rather
than obviously broken. A second-model review pass used to exist and was
removed: it rejected correct, test-verified diffs and approved defective
ones, so it added stops without adding safety. The tests are the check.

Read the diffs.
