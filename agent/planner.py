"""Planner node (Task 1.2)."""

from __future__ import annotations

import json
import re

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _extract_question(state: AnalystState) -> str:
    """Pull the latest human question out of the messages channel."""
    for msg in reversed(state["messages"]):
        # Works for both LangChain message objects and raw dicts (the
        # OpenAI-compatible request the deployed endpoint receives).
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None
            )
            return content
    # Fallback: last message of any kind.
    last = state["messages"][-1]
    return getattr(last, "content", None) or last.get("content", "")


def _parse_plan(raw: str, question: str) -> list[str]:
    """Parse the LLM's JSON array; fall back to a single-step plan on failure."""
    text = raw.strip()
    # Strip markdown code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        steps = json.loads(text)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps) and steps:
            return steps
    except (json.JSONDecodeError, TypeError):
        pass
    # Parse failure or malformed output — fall back to treating the whole
    # question as one step rather than crashing the graph.
    return [question]


def make_planner(llm):
    """Return a planner node bound to the given chat LLM."""

    def planner(state: AnalystState) -> dict:
        question = _extract_question(state)
        response = llm.invoke(
            [
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": question},
            ]
        )
        plan = _parse_plan(response.content, question)
        return {"plan": plan, "current_step_index": 0}

    return planner
