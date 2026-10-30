# Design: pluggable agents behind the gate

Status: proposal, 2026-10-29. Step 1 is built (branch `pluggable-agents`); the rest is not.

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
- **Unscoped**: allowed only when the project has real validate commands, and
  the review gate becomes mandatory for that item. "No silent wrongs" holds
  because a looser scope always buys a stronger check, never a weaker one.

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
- network goes only to the model endpoint.

The Qwen Code evaluation already used this pattern. The project's validate
commands run outside it, under forge, as now.

## Migration, one behaviour-preserving step at a time

1. **Registry and the four existing adapters.** `_dispatch_editor` becomes a
   registry lookup, and usage comes back in `Attempt`. Claude and agy usage
   become per thread. *Done 2026-10-29:* all 135 existing tests pass, plus
   12 new ones in `tests/test_agents.py`. Still to do: a Leash rerun of a
   small honest subset per backend, which needs the 3090.
1b. **The supervisor commits.** Adapters stop committing. Kept separate from
   step 1 because it changes what lands in git history (commit messages), so
   step 1 could stay a pure refactor.
2. **The Reviewer protocol.** Replace `run_review_gate`'s chain and the copied
   verdict parsing, and default to diff only.
3. **Scope policy** for the scoped and unscoped items above.
4. **`OpenCodeAgent` and the sandbox wrapper.**
5. **The routing and escalation config.**
6. **Leash: every agent through the same gate.** Report solved, wrong merged
   and parked per agent, on the honest set and on a SWE-bench Verified subset.
   That table is the public claim.

## Not changing

The worktree lifecycle, parking and `forge review`, the TODO states,
`--doer-preamble`, the validate and TDD gates, Studio. The gate is the
product. Everything in this proposal is about what sits in front of it.
