"""Agents forge can hand an item to. See docs/design-pluggable-agents.md."""
from .adapters import AgyAgent, AiderAgent, ClaudeCodeAgent, LiteAgent
from .base import Agent, Attempt, Task

REGISTRY: dict[str, Agent] = {a.name: a for a in (LiteAgent(), AiderAgent(), ClaudeCodeAgent(), AgyAgent())}
DEFAULT = "aider"


def get_agent(name: str | None) -> Agent:
    """The agent registered under `name`. An unknown or empty name gets aider,
    as the old dispatch's fall-through did, so a project that says nothing
    (or something forge doesn't know) keeps the behaviour it always had."""
    return REGISTRY.get(name or DEFAULT, REGISTRY[DEFAULT])


__all__ = ["Agent", "Attempt", "Task", "REGISTRY", "DEFAULT", "get_agent"]
