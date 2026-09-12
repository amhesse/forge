# aider-loop

Runs [aider](https://aider.chat) unattended against a markdown checklist, with
a safety net between each item and the next.

The loop itself is simple: take the next unchecked item, hand it to aider,
check what came back, mark it and move on. The value is in the checking. A
local model will confidently produce work that is wrong in ways that still
compile, and the checks below exist because each one caught a real failure
that the previous checks missed.

A separate tool, `spec_compiler.py`, drafts the checklist items themselves
from a vague goal - see its own section below. It exists because writing
items in the shape below is most of what makes them reliable, and that
shape is tedious enough by hand that skipping it is the easy mistake.

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

A `TDD:` item (see *Writing items*) runs a different sequence: write the
test, confirm it fails (and that the failure is real, not a crash — a
plain syntax check, before the project's `[validate] commands` run at
all), then implement, then checks 4–6 above against the combined result,
plus one more specific to this mode — the test file must be byte-identical
to what the red phase produced, not just absent from the touched set.

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
where it did. `review_server.py` (below) does the reading and the
`git branch -D` for you, with the diff already open.

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

## Reviewing what got parked: review_server.py

    python review_server.py --project-dir ~/projects/blockroad

Then open http://127.0.0.1:8765/. A page listing every branch
`git branch --list 'aider-loop/*'` currently has: the item's own text,
why it was parked, the diff, and two buttons — **Merge anyway** and
**Discard**. Ctrl+C stops it; nothing runs until a button is clicked, and
every click asks for confirmation first.

The branch is the source of truth, not the JSON run record — the same
principle as the worktree design above. A branch can outlive its record
(an old `~/.cache` cleaned up separately) or the record can outlive the
branch (already handled by hand); either way, what's actually still
sitting there needing a decision is whatever `git branch` currently
lists, so that's what drives the page. A record is only ever used for
context when one still exists for that branch.

Both buttons are real `git` operations on the project's actual
checkout — the same commands the *Worktrees* section above tells you to
run by hand, just with the diff already open and no typing the branch
name three times:

- **Merge anyway** runs `git merge --no-ff <branch>`. A parked item was
  never applied to the checkout in the first place, so this isn't
  undoing a revert — it's a human overriding "needs review" or "blocked"
  after actually reading the diff. A conflict is reported back and the
  merge is aborted automatically, not silently resolved.
- **Discard** runs `git branch -D <branch>`. Nothing else is touched.

Every POST re-checks the branch name against the live branch list before
doing anything — a stale click (the branch was already handled another
way) or a request naming something that was never a pending item (even
`master`) gets a message back, not a git command run against an
unintended ref.

No login, no token: there's nothing reachable here that isn't already a
`git` command sitting in your own shell history, and it binds to
`127.0.0.1` only. Stdlib only, like `aider_loop.py` itself — a browser
tab and a project directory are the only things this needs.

## Parallel workers

    python aider_loop.py --project-dir ~/projects/thing --models qwen25-coder-aider,qwen2.5-coder:7b

One model name changes which model runs (still one item at a time,
otherwise identical to the default). Two or more run that many items
*concurrently*, one worker per model — or set `[worker] models = [...]`
in `.aiderloop.toml` instead of passing it every time.

**Each worker is deliberately a different model, not N copies of the
same one.** This script never raises Ollama's `OLLAMA_NUM_PARALLEL` or
assumes any particular concurrency configuration on the Ollama side —
two workers sharing one model would just take turns on the GPU behind
Ollama's own default of one generation at a time, with none of the
wall-clock benefit and all of the added complexity below. Verified
directly: a 14B and a 7B coder model loaded and generated concurrently on
one 24GB GPU with real headroom left over (~21.8GB used, both at 100%
GPU, both producing real, complete output at once).

What changes under the hood, so two workers can safely share one
checkout:

- **One lock, everywhere it matters.** Every operation that touches the
  project's own shared git state — creating a worktree, merging one in,
  removing it — is serialized behind a single lock, held by whichever
  worker is finishing an item at that moment. Nothing else needs it:
  aider itself and every check run entirely inside that item's own
  worktree, fully isolated from whatever else is in flight.
- **Fast-forward isn't guaranteed anymore, and that's fine.** Two items
  can branch from the same base and only one can still be "the tip" once
  the first lands. `merge_ff()` now falls back to a real merge (`--no-ff`)
  whenever the fast-forward fails, and it still succeeds cleanly whenever
  the two items touched different files — the checklist's own convention
  ("keep items to one file where you can") is exactly what makes this the
  common case, not the exception. A genuine conflict still fails and
  aborts cleanly; nothing here resolves one automatically, and the losing
  item parks with its work intact on its own branch, same as any other
  parked item.
- **Item selection re-reads the checklist fresh each time**, so a worker
  always sees what other workers have already finished, and an in-memory
  set of already-claimed items is what stops two workers claiming the
  same open one in the gap before either has written a status back.

Verified with a controlled stub covering: two workers genuinely
overlapping on independent files (both merge, one via fast-forward, the
next via the `--no-ff` fallback, clean merge graph), and two workers
genuinely conflicting on the same file (the first merges, the second's
merge attempt hits a real conflict, aborts cleanly with zero corruption
to the checkout, and parks — `git status` afterward shows nothing but the
loop's own bookkeeping).

### Which item goes to which model

Nothing content-aware here beyond one signal this project actually has
evidence for. The first model in `--models` is the strong tier and
prefers a **hard** item; every other model is the weak tier and prefers
an **easy** one. Either tier falls back to whatever's open if none of its
preferred difficulty is left, so a worker never idles just because its
preferred kind of work ran out.

"Easy" means exactly one thing: the item has a byte-exact content spec
(see *Writing items* below). That's not a claim the *content* is simple —
it's that getting it wrong is low-stakes, because it's checked
byte-for-byte regardless of which model writes it, so a weaker model's
mistake costs one parked item, never a silent wrong answer. This is the
project's own README claim turned into a routing rule, not a new guess:
"Local models transcribe well and decide badly; exact specs play to
that." Everything else — including every TDD item and every item naming
more than one file — defaults to **hard** and goes to the strong tier,
because this project has only ever measured the 14B model, not the 7B,
make the judgment calls those need (TDD's green phase correctly
overriding an ambiguous placement instruction; a multi-file item's two
halves actually agreeing with each other).

Verified with a stub checklist mixing easy and hard items in list order
that deliberately didn't match tier preference: the strong worker skipped
over an earlier easy item to take a later hard one, the weak worker took
both easy items, and a follow-up test with more hard items than easy
confirmed the weak worker falls back to hard work rather than idling once
its preferred kind ran out — no double-claims, no starvation, all items
processed.

### What to actually expect, measured

Same two trivial, independent tasks, same machine, immediately before and
after: **168s sequential** (one 14B worker, one item after another) vs.
**120s with two workers** (that 14B plus a 7B) — a real 1.4x, not 2x.
Worth setting expectations by, not the raw worker count:

- The tasks were deliberately easy on both models, so this isolates
  scheduling overhead from difficulty effects — a mixed real checklist
  will vary.
- A duplicate-model "third worker" was also measured, for comparison:
  fully serialized, ~zero benefit, because Ollama's own
  `OLLAMA_NUM_PARALLEL` defaults to one generation at a time per model.
  Concurrency here only ever comes from *different* models running at
  once, never from adding more of the same one.
- The smaller model's real-world reliability on non-trivial items is
  still mostly unmeasured. If it parks more often than the 14B does, that
  wall-clock gain can shrink or vanish — a park produces nothing for that
  worker's slot of time, and routing only decides which item a model
  *attempts*, not whether it succeeds.

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

**Start an item with `TDD:`** to run it red-then-green instead of in one
pass:

```markdown
- [ ] TDD: implement `src/thing.py` so `test/test_thing.py` passes: given a
  negative amount, `format_price` raises `ValueError`.
```

Name exactly two files, one of them test-shaped (a `test/`/`tests/` path
component, or `.test.`/`_test.`/`test_` in the filename — covering both
`test/check.test.js` and `test_thing.py`/`thing_test.py` conventions).
Aider is called twice: first asked to write *only* the test — told
explicitly that it's expected to fail and not to touch the implementation
or write a stub to fake a pass — then, once that failure is confirmed to be
a real one (see below) and not a crash, asked to implement the other file
without touching the test again.

This is "the tests are the check" (below) applied to the test itself: a
test a human never looked at is exactly the kind of thing that can pass
by accident — checking the wrong thing, or nothing at all — and the usual
checks here can't tell that from a genuine one. Confirming it fails first
is the same thing a human reviewer does by habit before trusting a new
test to grade anything. Failure modes this catches, not just the happy
path: the test passing immediately (parked, needs-review — it may not
test the described behavior, or already passes), and the implementation
step rewriting the test to make it pass instead of writing real code
(parked, needs-review, byte-compared against the test as the red phase
left it — not just "was it in the touched set," since a same-content
rewrite would still show as touched).

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

## Drafting items: spec_compiler.py

Writing an item in the shape the checks above can actually verify -
one file, a named anchor, an enumerated structure instead of a vague
quantity - is what makes the difference between an item that passes on
the first try and one that burns every retry re-failing the same way.
That shape is real work, and skipping it is the easy mistake to make
when you're the one writing the checklist by hand.

    python spec_compiler.py --project-dir ~/projects/blockroad \
        --goal 'Add a check that a story title is not longer than 40 characters' \
        --dry-run   # drop this once the draft looks right

It calls a local model to draft items, validates their *shape* (a real
backtick path, a real "exactly:" fence, an anchor line that actually
exists in the file right now) against the same parsers aider_loop uses
to read them back, and appends them to the checklist - it never touches
project files and never runs aider itself. A human reads the draft
before the loop ever sees it.

Every rule it enforces is a real failure from a real run on this
project, baked into its system prompt rather than left as advice:

- **Don't split a change across a consistency constraint.** Asked to
  create a story and register it in the stories index as two separate
  items, the model wrote a structurally perfect story and it was
  rejected anyway - solely for not being in the index yet, because the
  project's checker requires each file to agree with the other. Neither
  half is valid alone, so both edits belong in one item.
- **Name the exact anchor for an edit into existing code**, and say
  outright whether it's a pure insertion. "Add a check for X" got pasted
  over an unrelated check and deleted it, twice, before the anchor was
  named - after which the same task merged in one attempt, 18 seconds,
  zero retries.
- **Enumerate structure instead of a vague quantity.** "At least 7
  scenes, 2 endings, 2 locks" got satisfied one constraint at a time
  across three retries while breaking earlier ones, and never converged.
  The same story, with every scene and its required parts spelled out,
  passed on the first try.
- **Only quote a line as "existing" if you were actually shown the file
  it's in.** A first version of this tool let the model draft an item
  that named a plausible-looking anchor line for `src/check.js` without
  the model ever having seen that file's real content - it invented one
  that appears nowhere in the actual source. gather_context() now
  inlines the full content of any file the goal names, and validate_item()
  independently checks a claimed anchor against the file as it exists on
  disk, so a hallucinated one is flagged before it's ever written down,
  not discovered later as a corrupted commit.

### Model choice matters here too

The obvious pick for a *drafting* role is the model meant to reason, not
just write code - this project's `qwen3.8-aider` (27B). It failed the
same way it fails at driving aider directly: given nothing but "draft
checklist items for this goal," it spent its entire output budget (405
lines, hit the num_predict cap) reasoning in plain prose about
implementation edge cases and never emitted a single checklist item.
`qwen25-coder-aider` (14B) - the model with no reasoning framing at all -
just answers, and correctly: a real anchor line, a correct insertion,
merged on the first attempt. `spec_compiler.py` defaults to it for
exactly that reason. Measured behavior over presumed capability.

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
