"""Which agent gets an item, and who takes over when it's parked (design step 5).

    [agents]
    exact    = "lite"                       # items that name their files and declare no scope
    explore  = "opencode"                   # items with a `scope:` line, or no named file
    escalate = ["opencode", "claude"]       # tried in order on a parked item, each in a fresh worktree
    review   = "ollama:gemma4:26b"          # the default --review, "reviewer" or "reviewer:model"

An entry is "backend" or "backend:model" (the model part may itself contain
colons, as Ollama tags do). Everything is optional, and a project with no
[agents] section runs exactly as before: one backend from --backend /
[model] backend, and at most one fallback from --fallback-backend /
--fallback-model. An explicit --backend wins over exact/explore.
"""
from dataclasses import dataclass

from ..config import cfg
from ..validator import expected_files, item_scope

CLOUD_BACKENDS = ("claude", "agy")


@dataclass(frozen=True)
class Step:
    backend: str | None   # None: whatever --backend / [model] backend says
    model: str | None
    label: str


def parse_entry(entry: str) -> tuple[str, str | None]:
    backend, _, model = str(entry).partition(":")
    return backend.strip(), (model.strip() or None)


def item_kind(item_text: str) -> str:
    """"exact" when the item names files and declares no scope; otherwise
    "explore" (it has a scope, or names nothing)."""
    return "exact" if expected_files(item_text) and not item_scope(item_text) else "explore"


def first_step(item_text: str, model: str | None, explicit_backend: bool) -> Step:
    if not explicit_backend:
        entry = cfg("agents", item_kind(item_text), default=None)
        if entry:
            backend, entry_model = parse_entry(entry)
            return Step(backend, entry_model or model, f"{backend} ({item_kind(item_text)} item)")
    return Step(None, model, "")


def escalation_steps(args, first: Step) -> list[Step]:
    """The agents a parked item goes to next, in order. [agents] escalate if
    set; otherwise the single --fallback-backend / --fallback-model, under
    the same rule as before (a cloud backend always; a local one only with
    a fallback model different from the one that just failed)."""
    chain = cfg("agents", "escalate", default=None)
    if chain:
        if isinstance(chain, str):
            chain = [chain]
        steps = []
        for entry in chain:
            backend, model = parse_entry(entry)
            if (backend, model) == (first.backend, first.model):
                continue  # don't hand it straight back to the agent that just parked it
            steps.append(Step(backend, model, f"{backend} ({model})" if model else backend))
        return steps
    backend = getattr(args, "fallback_backend", None) or "aider"
    model = getattr(args, "fallback_model", None)
    if backend in CLOUD_BACKENDS:
        return [Step(backend, model, f"{backend} ({model or 'default model'})")]
    if model and model != first.model:
        return [Step(backend, model, f"fallback model {model}")]
    return []


def default_review() -> tuple[str | None, str | None]:
    entry = cfg("agents", "review", default=None)
    return parse_entry(entry) if entry else (None, None)
