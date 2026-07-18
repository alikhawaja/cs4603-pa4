"""Planner node (Task 1.2).

Decomposes the user question into 2–5 atomic steps. The LLM must return a JSON
array of strings; anything unparseable falls back to a single step containing
the raw question, so a planner failure degrades to "answer directly" instead
of crashing the graph.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _message_content(message) -> str:
    """Message content, whether it's a LangChain message or a plain dict."""
    content = message.content if hasattr(message, "content") else message["content"]
    return content if isinstance(content, str) else str(content)


def get_user_question(state: AnalystState) -> str:
    """The most recent human/user message in the conversation."""
    for message in reversed(state["messages"]):
        msg_type = getattr(message, "type", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        if msg_type in ("human", "user"):
            return _message_content(message)
    # Degenerate input (no user message at all): plan over whatever is last.
    return _message_content(state["messages"][-1])


def parse_plan(raw: str) -> list[str] | None:
    """Extract a JSON array of non-empty strings from an LLM reply, else None."""
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    steps = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
    return steps or None


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        question = get_user_question(state)
        response = llm.invoke(
            [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=question)]
        )
        plan = parse_plan(_message_content(response)) or [question]
        return {"plan": plan, "current_step_index": 0, "step_results": []}

    return planner
