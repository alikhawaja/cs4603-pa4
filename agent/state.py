"""State schema for the Document Analyst graph (Task 1.1).

`messages` is the entry/exit channel of the deployed endpoint (messages in →
messages out), so it uses the `add_messages` reducer. The remaining fields are
internal scratch space with no reducer: a node that updates them returns the
complete new value, which keeps every transition explicit in the trace.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AnalystState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step_index: int
    step_results: list[str]
    next_agent: str
    final_answer: str
