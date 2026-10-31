# Design: pluggable agents behind the gate

Status: proposal, 2026-10-29. Steps 1 to 6 are built (branch `pluggable-agents`). Step 6 found an open hole: exploring agents on TDD items.

## Why

forge's distinctive result is measured: on aider's own benchmark it merged
**0% wrong work** where aider at defaults reported 27% of runs done when they
were wrong. That comes from the gate (worktree per item, scope and content
checks, validate commands, TDD, review, parking), not from the editor.

The editors are deliberately single-shot. `lite_editor` decides every file in
Python before the model is called and asks for whole files back. That is
exactly why it is reliable on clearly specified items, and why it can't do
items that need exploring a repo, running things, or touching files nobody
named.

So forge should not try to become Claude Code. It should become the
**supervisor**: any agent does the work inside the worktree, and the same gate
decides whether it merges. "The harness you can trust overnight, with any
model" is a position no mainstream harness takes.

## What's there now

- `runner._dispatch_editor` (runner.py:365) is an `if` chain over four
  backends: `claude`, `agy`, `lite`, and `aider` as the fallthrough. Each
  returns `(success, output)`.
- Token usage comes back through `last_usage()` functions. `lite_editor`
  already keeps its usage per thread, but `claude_editor` and `agy_editor`
  used module globals. Parallel workers are threads (`run_parallel`,
  `ThreadPoolExecutor`), so concurrent Claude or agy items could read each
  other's counts. (Fixed in step 1.)
- The adapters disagree about who commits: lite and claude commit
  themselves, while aider commits through its own settings.
- `run_review_gate` (runner.py:519) is a second `if` chain over `claude`,
  `ollama` and `agy` reviewers. Two of the three parse the verdict with the
  same copied code.
- The checks after an edit (`restore_unnamed_files`, `suspicious_new_paths`,
  `detect_prompt_leakage`, `content_mismatches`, syntax, validate, TDD) already
  don't care which editor ran. That part stays exactly as it is.

## The interface

```python
# forge/agents/base.py
@dataclass(frozen=True)
class Task:
    text: str                 # item text, with --doer-preamble already applied
    worktree: Path
    files: list[str]          # expected_files(); empty means "agent decides"
    model: str | None
    hard: bool                # estimate_difficulty() == HARD
    think_off: bool           # set after a truncated attempt
    feedback: str | None      # repair note from the last failed check, if any
    timeout_s: int

@dataclass
class Attempt:
    ok: bool                  # the agent says it produced something
    output: str               # its final text
    usage: Usage              # tokens and seconds, returned rather than in a global
    trace: str = ""           # reasoning or transcript, for Leash; never shown to the reviewer

class Agent(Protocol):
    name: str
    traits: frozenset[str]    # "single_shot" | "explores" | "runs_shell" | "local" | "metered"
    def attempt(self, task: Task, log: Callable[[str], None]) -> Attempt: ...
```

The rules every adapter follows:

1. It works only inside `task.worktree`.
2. **It doesn't commit.** The supervisor runs `git add -A && git commit`
   after every attempt, the same way for every agent. One code path, and an
   agent that commits partial work can't confuse the diff base.
3. It doesn't decide success. `Attempt.ok` only means "I produced
   something". Merging is still the gate's decision alone.
4. An agent with the trait `runs_shell` runs only inside the sandbox wrapper
   (below). The registry refuses to build one without it.

The reviewer gets the same treatment:

```python
class Reviewer(Protocol):
    name: str
    def review(self, task: Task, diff: str, log) -> Verdict: ...   # APPROVE / REJECT / ERROR
```

It sees **only the task and the diff**. That's a Leash finding: gemma4:26b
caught 100% of planted bugs at a 2% audit budget from the diff alone, and 89%
once the doer's reasoning was added, because the reasoning quotes the injected
instruction as a requirement.

## Adapters

| Adapter | Wraps | Traits | Notes |
|---|---|---|---|
| `LiteAgent` | `lite_editor.run_lite_on_item` | single_shot, local | Unchanged behaviour. The default for exact-spec items. |
| `AiderAgent` | `run_aider_on_item` | single_shot, local | Kept for comparison. Its exit status isn't trusted (that's already known). |
| `ClaudeCodeAgent` | `claude_editor.run_claude_on_item` | explores, metered | No Bash, as now. Pointing `ANTHROPIC_BASE_URL` at a local LiteLLM → llama-server/Ollama makes it a **local** exploring agent with the same code. |
| `AgyAgent` | `agy_editor.run_agy_on_item` | explores, metered | |
| `OpenCodeAgent` (new) | `opencode run` (non-interactive) | explores, runs_shell, local | First fully local exploring agent. Needs the sandbox. Point it at llama-server or Ollama's native API, not Ollama's `/v1`, where tool calls have broken silently before. |
| `MiniAgent` (later) | a small built-in loop: read/search/edit and a sandboxed shell | explores, runs_shell, local | Only if OpenCode's loop turns out wrong for local models. Open projects like mini-swe-agent show the loop itself is small. |

## Scope: the one real change to the gate

Today an item names its files, and anything else touched gets restored.
Exploring agents legitimately touch files nobody named. The rule:

- **Named files** (the current behaviour): strict, unchanged.
- **Declared scope** (new item syntax, e.g. `scope: src/game/**`): changes are
  allowed inside the scope and restored outside it.
- **Unscoped**: refused by default, as before. With `[scope] unscoped =
  "guarded"` it runs only when the project has real validate commands and
  the run has `--review`. "No silent wrongs" holds because a looser scope
  always buys a stronger check, never a weaker one.

## Routing

The existing "which item goes to which model" (`forge calibrate`) becomes
"which item goes to which agent":

```toml
[agents]
exact    = "lite"                 # items that name their files
explore  = "opencode"             # items with a scope, or no files named
escalate = ["opencode", "claude"] # tried in order on a parked item, each in a fresh worktree
review   = "ollama:gemma4:26b"    # diff only
```

Escalating reuses today's `--fallback-backend` path: a fresh `...-retry`
worktree, and the same gate.

## Sandbox

`sandboxed(agent)` runs the agent's process under bwrap:
- the worktree is bind-mounted read-write;
- the rest of `$HOME` is not mounted;
- `/usr` is read-only;
- network is shared (bwrap can't limit it to the model endpoint; see step 4).

The Qwen Code evaluation already used this pattern. The project's validate
commands run outside it, under forge, as now.

## Migration, one behaviour-preserving step at a time

1. **Registry and the four existing adapters.** `_dispatch_editor` becomes a
   registry lookup, and usage comes back in `Attempt`. Claude and agy usage
   become per thread. *Done 2026-10-29:* all 135 existing tests pass, plus
   12 new ones in `tests/test_agents.py`. **Leash A/B on the 3090, steps 1-5
   vs master** (qwen3.8:27b, lite, all 16 tasks x 2 trials, no escalation):
   - master: 29/32 passed, 0 wrong merged, catch 100%, friction 0%,
     median 29.9 s;
   - branch: 28/32 passed, 0 wrong merged, catch 100%, friction 0%,
     median 25.6 s.

   The one difference is a single `tdd_average_transaction` trial whose
   model output was wrong on the branch and right on master. forge caught it
   either way, so it's model variance, not a behaviour change.
1b. **The supervisor commits.** Adapters stop committing; forge stages and
   commits after every attempt (`agents/commit.py`), keeping each agent's old
   message prefix. It never stages the untracked config and links
   `materialize()` copied in; claude and agy used to `git add -A` those. "Made
   no changes" is decided once, for every agent. aider still auto-commits, so
   there is usually nothing left for forge to commit. *Done 2026-10-29:* 154
   tests pass, including the end-to-end `forge run` hook tests.
2. **The Reviewer protocol.** Replace `run_review_gate`'s chain and the copied
   verdict parsing. *Done 2026-10-29* (`agents/review.py`): one prompt and one
   verdict parser for claude, ollama and agy. Reviewers never got the doer's
   reasoning before either; now `ReviewTask` has no field for it, so they
   can't. Changes ollama and agy reviews see:
   - They now get the TDD instruction, which only claude got before.
   - They use claude's robust parser. A verdict wrapped in markdown now
     counts. A reply with no clear verdict now parks for a human instead of
     counting as a reject.
   - agy reviews inside the worktree; before, it ran in forge's own directory.
   - ollama warns when the prompt is bigger than its 4096-token context
     (Ollama cuts it silently), and `[review] ollama_num_ctx` raises it.
   172 tests pass, 18 of them new in `tests/test_review.py`, including the
   gate's approve, reject and couldn't-run paths.
3. **Scope policy** for the scoped and unscoped items above. *Done
   2026-10-29.* One correction to the plan: an unscoped item never actually
   ran open before, because preflight aborted the whole run over it. So the
   default stays "refuse", `"guarded"` is opt-in, and `"allow"` is the
   escape hatch.
   - `scope:` lines are parsed in `validator.item_scope`, and
     `restore_unnamed_files` honours them.
   - Claude and agy are told the scope in their prompt.
   - Preflight accepts a scoped item.
   - 187 tests pass, 15 of them new in `tests/test_scope.py`, including two
     real `forge run`s.
4. **`OpenCodeAgent` and the sandbox wrapper.** *Done 2026-10-29*
   (`agents/opencode.py`, `agents/sandbox.py`).
   - OpenCode runs as `opencode run --format json --auto`, under bwrap, with
     its whole config passed in `OPENCODE_CONFIG_CONTENT`.
   - The network is shared, not filtered to the model endpoint; that's a
     known gap, documented.
   - Tested end to end with no GPU: the real `forge run`, the real OpenCode
     binary and the real sandbox, against a scripted fake model server
     (`tests/fake_openai.py`). Work inside the scope merged, a shell edit
     outside it was put back, a write to the real home directory never
     arrived, and a model that never edits was parked.
   - 196 tests pass.
   - **Real run (2026-10-29), qwen3.8-64k on the 3090 through Ollama's
     /v1:** two scoped items on a small game project, both merged in about
     80 s, 5 tool steps each. A hidden grader the agent never saw passed 3/3.
     Only files inside the scope changed, and the tests were untouched.
     Ollama's tool calling worked. The first attempt, under systemd, parked
     both items without logging why: `opencode` wasn't on systemd's PATH.
     Every refusal is now logged.
5. **The routing and escalation config.** *Done 2026-10-29*
   (`agents/routing.py`, `runner.run_routed`).
   - The sequential and parallel loops share one routed path. Without
     `[agents]`, they behave as before.
   - An explicit `--backend` wins over exact/explore.
   - A typo in `[agents]` stops the run up front, instead of silently
     becoming aider.
   - 206 tests pass, including real `forge run`s for routing by kind,
     escalation, and a misspelled name.
   - **Hang found and fixed.** The OpenCode end-to-end tests hung whenever
     the run's own stdin was an open pipe: `opencode run` reads stdin when
     it isn't a terminal, and the sandbox passed forge's stdin through. The
     sandbox now gives the agent an empty stdin. It also stops a sandbox
     that's still running 10 s after the agent's final answer.
6. **Leash: every agent through the same gate.** *Done 2026-10-30, on the
   Leash honest set.* The table is in the README, under Results. The raw
   results are in leash/results/forge-agents-2026-10-30/.
   - lite and aider merged nothing wrong.
   - The exploring agents merged 3 wrong changes in 96 trials (OpenCode 2,
     Claude Code 1), all TDD items whose self-written test agreed with a
     wrong implementation.
   - **Closed (2026-10-30).** An exploring agent takes a TDD item only with
     `--review` (498a1c4). The TDD review instruction judges the
     implementation against the task (61cd5e3). The Ollama reviewer sends
     think=false and never sys.exit()s (3ace67c).
   - Re-test on the 4 TDD tasks x 2: OpenCode + gemma4 reviewer 7/8 merged,
     0 wrong. Claude Code + Claude reviewer 3/8 merged, 0 wrong.
   - The SWE-bench Verified subset is still to do. It needs per-repo Docker
     images and its own harness work.

## Not changing

The worktree lifecycle, parking and `forge review`, the TODO states,
`--doer-preamble`, the validate and TDD gates, Studio. The gate is the
product. Everything in this proposal is about what sits in front of it.
