"""Planner node (Task 1.2).

The planner is the graph's entry point after START. It reads the user's question and
decomposes it into an ordered list of 2-5 atomic steps (retrieval steps and computation
steps). Those steps drive the supervisor loop that follows.

Design notes:
  - Output is parsed as a JSON array. LLMs sometimes wrap JSON in prose or ```json
    fences, so parsing is defensive: try strict json, then fall back to extracting the
    first [...] block, then fall back to a single-step plan (the whole question).
  - A single-step fallback means a parse failure degrades to "answer the question in one
    step" rather than crashing the graph.
"""

from __future__ import annotations

import json
import re

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _last_user_text(messages: list) -> str:
    """Return the text of the most recent human/user message."""
    for msg in reversed(messages):
        # Messages may be LangChain objects (HumanMessage) or plain dicts.
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            return getattr(msg, "content", None) or msg["content"]
    # No explicit user message found — fall back to the last message's content.
    last = messages[-1]
    return getattr(last, "content", None) or last["content"]


def _parse_plan(raw: str) -> list[str]:
    """Parse the LLM output into a list of step strings, defensively."""
    text = raw.strip()
    # Strip a ```json ... ``` (or ``` ... ```) fence if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Attempt 1: the whole thing is valid JSON.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Attempt 2: grab the first [...] array anywhere in the text.
        match = re.search(r"\[.*\]", text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else None

    if isinstance(parsed, list):
        steps = [str(s).strip() for s in parsed if str(s).strip()]
        if steps:
            return steps[:5]  # cap at 5 per the spec
    return []  # signal "could not parse" to the caller


def make_planner(llm):
    """Build the planner node bound to a chat `llm`."""

    def planner(state: AnalystState) -> dict:
        question = _last_user_text(state["messages"])
        response = llm.invoke(
            [
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": question},
            ]
        )
        steps = _parse_plan(getattr(response, "content", str(response)))

        # Fallback: if parsing failed, treat the whole question as one step so the
        # graph still runs instead of crashing.
        if not steps:
            steps = [question]

        return {
            "plan": steps,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner
