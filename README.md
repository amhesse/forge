# forge

**Leave a local AI coding model working through a to-do list unattended,
and come back to work that is either verified or set aside for you: never
broken work marked "done".**

AI coding assistants often say a task is finished when it isn't. The code
runs and looks plausible, but it's wrong. If you're watching, you catch
it. If you're not, it ships. forge sits between the model and your code.
It hands the model one checklist item at a time, checks what comes back,
and merges only work that passes. Anything it can't confirm is **parked**
on its own git branch, with a note saying why, for you to review later.

**Who it's for:** developers running local models (through Ollama) who
want to hand off a batch of small, clearly written coding tasks, such as
small features, bug fixes, refactors or test-first items, and walk away.
forge can also drive aider, Claude Code or Gemini as the model doing the
editing.

**The goal:** no *silent wrongs*, meaning no broken change that looks
finished. Getting fewer tasks done is acceptable. Getting a wrong change
merged without anyone noticing is not.

## Results

8 problems from aider's own benchmark
([Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark),
Exercism's Python track), all run with the same local model
(`qwen3.8:27b`, one RTX 3090 Ti) and judged by hidden tests the model never
sees:

| | Solved and merged | **Wrong, but reported done** | Set aside for a human |
|---|---:|---:|---:|
| aider, default settings (120 runs) | 73% | **27%** | 0% |
| forge, default settings (120 runs) | 20% | **0%** | 80% |
| forge, tuned (80 runs) | 86% | **0%** | 14% |

- **aider never once gave up.** Across 750 runs (this benchmark plus 14
  internal tasks on three models), every failure was reported as a
  success. forge's failures are parked instead.
- **"Tuned"** means two flags: `--lite-hard-think-off
  --hard-validation-retries 5`. The tuning was done on forge, while aider
  ran at its defaults; a like-for-like comparison with aider tuned too is
  still to be done.
- Measured with Leash, a separate benchmark harness (write-up to follow).

**What it doesn't do:** forge catches work that fails your tests or breaks
its structural rules (wrong files touched, content that doesn't match an
exact spec, and so on). It does not make a model trustworthy, and it won't
catch a bug your tests don't exercise, including one put there on purpose.
Read the diffs of anything that matters. See *What it is not* at the end.

## Quick start

Needs Python 3.11+, `git`, and [Ollama](https://ollama.com) with a coding
model pulled (aider is optional).

    git clone https://github.com/amhesse/forge && cd forge
    pipx install -e .

In your project (it must be a git repository), write a `TODO.md` with one
item per line. Name the files and say exactly what you want:

    - [ ] In `app/text.py`, add `slugify(text: str) -> str` that lowercases, replaces runs of non-alphanumerics with "-", and strips leading/trailing "-".

Then run it, and later look at anything it parked:

    forge run --project-dir . --todo-file TODO.md --backend lite --models qwen3-coder:30b
    forge review --project-dir .      # parked items, in a browser view

Other commands:

    forge draft --project-dir ~/projects/thing --goal "..."   # turn a vague goal into checklist items
    forge studio --project-dir ~/projects/thing --port 8888   # live telemetry and studio UI
    forge eval                                                 # forge's own regression suite

Stdlib only — no dependency ever needs installing for the package itself,
only for the project you point it at. `pipx` is the right tool on a
system that blocks bare `pip install` (PEP 668, the default on Arch and
other recent distros): it builds an isolated environment for `forge` and
still puts a single `forge` command on your PATH.

## How it works

The loop itself is simple: take the next unchecked item, hand it to aider,
check what came back, mark it and move on. The value is in the checking. A
local model will confidently produce work that is wrong in ways that still
compile, and the checks below exist because each one caught a real failure
that the previous checks missed.

A separate command, `forge draft`, drafts the checklist items themselves
from a vague goal - see its own section below. It exists because writing
items in the shape below is most of what makes them reliable, and that
shape is tedious enough by hand that skipping it is the easy mistake.

Every item runs in its own `git worktree`, branched from the project's
current HEAD. Only an item that passes every check is merged back. A
failure is never applied at all, so there is nothing to undo and no reason
to stop — the run keeps going, and each parked item's work is left on its
own branch to read in the morning.

    forge run --project-dir ~/projects/thing --todo-file TODO.md

### About the name

Was `aider-loop`, when driving aider was the only option; renamed once a
second editing backend made that name inaccurate. The old name still
works as a directory symlink and the old script names still exist as
`forge/aider_loop.py` etc. inside the package, so nothing that referenced
them elsewhere silently broke. Item branches (`aider-loop/item-N-<date>`)
and the cache directory (`~/.cache/aider-loop/<project>/`) still use the
old name too, deliberately - renaming those would orphan every branch
and run record any project made under the old name, for a purely
cosmetic gain.

**Naming collision to know about:** `forge` is also the command name for
Foundry's Ethereum/Solidity toolchain. Nothing on this machine currently
installs that binary; if it ever does, whichever installs second wins the
PATH entry, silently. `command -v forge` before relying on either.

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

## Reviewing what got parked: `forge review`

    forge review --project-dir ~/projects/blockroad

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
`127.0.0.1` only. Stdlib only, like the rest of this project — a browser
tab and a project directory are the only things this needs.

## Forge Studio: Live Telemetry, Spec Architect & Adversarial Critic

    forge studio --project-dir ~/projects/thing --port 8888 --open

Forge Studio combines real-time loop telemetry, interactive execution controls, an in-browser spec architect with deterministic rule linting, and visual branch diff review with an automated adversarial code critic into a single local web environment.

- **100% Local & Stdlib-Only**: Binds to `127.0.0.1:8888`. Zero third-party Python packages, zero node/npm build step, styled with Catppuccin Mocha.
- **Live Telemetry & Pipeline Stepper**:
  - Live progress stepper through `Idle` -> `Worktree` -> `Coding` -> `Validation` -> `Merged`/`Parked`.
  - Inference speed tracking (live tokens/second and peak rate).
  - Hardware gauges for NVIDIA RTX GPUs (VRAM used/total, GPU temp, wattage) and Ollama loaded models and context limits.
  - Real-time terminal streaming via Server-Sent Events (SSE).
- **Interactive Run Controls**:
  - Start and stop the loop directly from the header toolbar.
  - Switch between `Lite` (whole-file) and `Aider` (diff) backends.
  - Select active Ollama models on the fly.
- **Spec Architect & Rule Linter**:
  - Interactive `TODO.md` editor with auto-discovery and instant saving.
  - Real-time rule linter: flags missing backtick file targets (fatal), hallucinated anchor lines (fatal), or oversized items (advisory warnings).
  - AI Goal Decomposer: enter a high-level feature goal and have the local reasoning model draft valid, single-file tasks.
- **Diff Reviewer & The Adversarial Critic**:
  - Syntax-highlighted unified diffs for all parked worktree branches.
  - One-click **Adversarial Critic**: prompts the model to perform a rigorous security and edge-case review of the diff with a verdict (`APPROVE`, `CAUTION`, `REJECT`) before merging.
  - One-click **Merge Anyway** (`git merge --no-ff`) and **Discard** (`git branch -D`).

## Parallel workers

    forge run --project-dir ~/projects/thing --models qwen25-coder-aider,qwen2.5-coder:7b

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

### Measuring it: `forge calibrate`

    forge calibrate --models qwen2.5-coder:14b,qwen2.5-coder:7b --trials 3
    forge calibrate --report                  # re-print from calibration.jsonl
    forge calibrate --self-test               # check the bench, no model

Runs every item in `bench/items.toml` alone, once per model per trial,
against a fresh copy of a small fixture project (`bench/fixture/`). The
items cover exact specs, small and logic-heavy single-file prose edits, a
bug fix from a symptom, multi-file changes and a TDD item. After forge
finishes, a grader the model never sees (`bench/graders/<id>.py`) decides
whether the work is actually correct. Parked branches get graded too, so
each trial ends up as one of four outcomes:

- `pass`: merged and correct.
- `silent_wrong`: merged but wrong. forge's checks missed it, and this is
  the outcome that makes a model unsafe for a category.
- `parked_ok`: parked even though the work was correct (the checks were
  too strict).
- `parked`: parked and wrong (the safety net worked).

The report lists, per model, the categories with at least an 80% pass
rate and zero `silent_wrong`. Those are the ones safe to route to that
tier. Results are appended one trial at a time, so an interrupted run can
be resumed with the same command. `--self-test` confirms each grader
fails on the untouched fixture and passes on `bench/reference/<id>/`.

The bench itself is **not in this repository**: hidden graders published
on the internet stop being hidden from models trained on it. Point
`--bench-dir` at your own (default: `bench/` next to `src/`, which is
gitignored), laid out as:

    items.toml              [[item]] tables: id, category, text
    fixture/                the starting project, with its own tests/
    graders/<id>.py         hidden grader; exit 0 = correct
    reference/<id>/         overlay of a known-correct solution

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

## Editing backend: aider, or lite_editor

    forge run --project-dir ~/projects/thing --backend lite

Or `[model] backend = "lite"` in `.aiderloop.toml`. Defaults to `aider`,
so a project that says nothing keeps exactly the behaviour it had.

`lite_editor.py` does the same job as calling aider — task and files in,
committed edits out — but is built for this loop's one usage pattern
(headless, one message, no chat history, no multi-turn negotiation)
rather than aider's general-purpose machinery. It exists because of three
failures aider produced on this project's own real runs:

1. **SEARCH/REPLACE brittleness.** `diff` format needs the model's block
   to match the file character-for-character. Measured on a real
   `voltagedrop` item: the model emitted spaces, the file used tabs, the
   edit was silently discarded and three reflections were burned finding
   out.
2. **The "which files should I add?" reflection loop.** A model that
   reasons in prose answers that question instead of editing — what
   disqualified qwen3.8:27b twice (12k tokens sent, 1.6k back, zero
   edits).
3. **Invented filenames.** aider infers a target path from surrounding
   text when a block doesn't parse; a model's own reasoning has twice
   become an actual file (`File Listing: stories/index.json`).

All three come from asking the model to describe *where* to write and to
produce a *diff*. So `lite_editor` asks for neither. Every file it will
touch is decided in Python before the model is called (from the same
`expected_files()` parsing that already feeds `--file`), and the model is
asked for each file's **entire new content** — the one thing this project
has repeatedly measured local models being good at. Python does the
diffing, because Python does it correctly.

The output contract is a sentinel block, parsed by fixed regex rather
than inferred from context:

```
===FILE: path/relative/to/project===
<the file's complete new content>
===END===
```

Prose outside the blocks is ignored. **A block naming a file outside the
agreed set is never written** — not sanitized, not repaired, ignored. An
invented path structurally cannot reach disk, which is the same bug class
`restore_unnamed_files()` exists to clean up after; here there is no
"after".

Verified: eight parser cases (multi-file, prose around blocks, markdown
fences *inside* file content, unterminated block, whitespace in markers)
and six end-to-end cases (a block literally named `File Listing:
invented.py` ignored while the real file was still written; new file with
parent directories created; a no-op rewrite rejected without an empty
commit; unparseable output failing without touching disk; commit actually
made on success; refusing to run with no named files or no model). Then
against a real model on the exact tab-indented shape that broke aider's
diff format — merged in 26s with tab indentation preserved byte-for-byte.

The tradeoff is the same one `whole` format has always had: the file
costs context twice, once read and once rewritten. Keep items to one
file, and prefer `lite` on projects whose files are small enough to
rewrite comfortably.

## Escalating to Claude: `--fallback-backend claude` and `--review claude`

The local model is the workhorse (`qwen3-coder:30b` here). Claude is
an escalation path, run through the Claude Code CLI (`claude -p`):

    forge run --project-dir ~/projects/thing --todo-file TODO.md \
        --models qwen3-coder:30b --backend lite \
        --review claude --review-model sonnet \
        --fallback-backend claude --fallback-model opus

- `--review claude` is the last check before a merge. It runs after every
  mechanical check has passed, and Claude reads the diff against the item
  looking for wrong logic. That is the silent-wrong class `forge calibrate`
  counts, and none of the other checks can see it. A rejection goes back
  to the same editor as feedback, for up to `--max-validation-retries`
  rounds. If the review can't run at all, the item is parked as
  needs-review rather than merged unreviewed.
- `--fallback-backend claude` hands a parked item to Claude in a fresh
  worktree, which gets its own branch (`...-retry`). Claude's work goes
  through exactly the same checks, including scope restore, validate
  commands and the review. `--fallback-model` is optional here.

No API bill: `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` are removed
from the child environment, so `claude -p` always uses the logged-in
subscription. It still counts against the plan's usage limits. Claude
gets only Read/Edit/Write/Glob/Grep (Read/Glob/Grep when reviewing) and
no shell, because forge runs the project's validate commands itself.

## Token usage, not cost

Every run ends with a line like:

    Tokens: ~3,600 prompt + ~39 completion across 1 item(s). Local model,
    no bill - this is where compute went, not what it cost.

These are local models with no per-token bill, so "cost" here means
compute, not money - which items were expensive, whether a retry paid
for itself, not a dollar figure to budget against.

The two backends report this differently, and the gap is real, not a
bug: `lite` reads `prompt_eval_count` / `eval_count` straight off
Ollama's own response for every call, plus wall-clock generation time.
`aider` is a subprocess whose API traffic this script never sees, so its
number comes from parsing the `Tokens: X sent, Y received.` line aider
prints to its own stdout - real, but self-reported and without a time
figure. Per-item detail (including which phase of a TDD item spent what)
is in each item's own JSON record under `~/.cache/aider-loop/<project>/runs/`.

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

## Drafting items: `forge draft`

Writing an item in the shape the checks above can actually verify -
one file, a named anchor, an enumerated structure instead of a vague
quantity - is what makes the difference between an item that passes on
the first try and one that burns every retry re-failing the same way.
That shape is real work, and skipping it is the easy mistake to make
when you're the one writing the checklist by hand.

    forge draft --project-dir ~/projects/blockroad \
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
merged on the first attempt. `forge draft` defaults to it for
exactly that reason. Measured behavior over presumed capability.

## Requirements

Python 3.11+ (stdlib only), `aider`, `git`, and `ollama` if using local
models. `node` is optional and enables JS/HTML syntax checking.

`git` is now a hard requirement rather than a warning — per-item worktrees
are what keep a failed item away from your checkout, and there is no
sensible way to degrade that. The old non-git path ran items directly in
the tree and left broken changes in place for the next item to build on,
which is exactly what worktrees replaced.

## Running this project's own tests

    PYTHONPATH=src python3 -m unittest discover -s tests -q

Every case in `tests/` traces to a real bug found by running this tool
against a real checklist, not a hypothetical - the negation-detection
tests exist because 12 of 20 items parked incorrectly on a real
overnight run before `_negates_before()` was added, and the fence-
stripping tests exist because the first real `lite` run wrote a literal
` ```python ` into every file. A tool that enforces TDD on every project
it touches had, until this section, none of its own; these are not
exhaustive, but every one of them is load-bearing.

    forge eval

The other half, one level up: `unittest discover` checks pure functions
in isolation, in milliseconds. `forge eval` runs the real installed
`forge run` command end-to-end against throwaway git repos with a stub
`aider`, and asserts the actual observable outcome - which items merged,
which parked, whether a forbidden file ever reached a real checkout.
Slower (seconds, not milliseconds) and heavier, but it's the level every
real bug found this session actually lived at: a wrong merge decision, a
file that shouldn't exist, a status that doesn't match what happened.
Run it after any change to the loop's own logic, before trusting it
against a real project.

It was built by running this exact command against this exact repo -
`forge` wrote its own `eval.py` and wired its own `eval` subcommand into
`cli.py`, both merged through its own worktree-and-checks pipeline, no
different from any other project this tool has been pointed at. The one
real snag was a self-inflicted item-authoring mistake (a byte-exact spec
written for a single line inside a larger file, which the checker always
compares against the WHOLE file) - caught, fixed, re-run, merged clean.

## What it is not

This does not make a local model trustworthy. In measured use it was
reliable on exact-content items and needed correction on **every**
open-ended one — the failures being confident, plausible, and wrong rather
than obviously broken. A second-model review pass used to exist and was
removed: it rejected correct, test-verified diffs and approved defective
ones, so it added stops without adding safety. The tests are the check.

Read the diffs.
