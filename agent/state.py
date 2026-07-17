"""State schema for the Document Analyst graph (Task 1.1)."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AnalystState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step_index: int
    step_results: Annotated[list[str], operator.add]
    next_agent: str
    final_answer: str