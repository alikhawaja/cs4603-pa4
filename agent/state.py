"""State schema for the Document Analyst graph (Task 1.1).

`AnalystState` is the single shared object every node reads from and writes to as the
graph runs. In LangGraph, each node returns a dict of updates; LangGraph merges those
updates into the state and passes the new state to the next node.

Most fields are *replaced* on update (last write wins). The exception is `messages`,
which uses the `add_messages` reducer so returned messages are **appended** to the
history instead of overwriting it — this is what makes `messages` a growing transcript
and is the channel the deployed endpoint reads its answer from (Task 1.6 / Part 2).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AnalystState(TypedDict):
    # Conversation transcript. `add_messages` = append (not overwrite) on update.
    # Entry channel (user question comes in here) AND exit channel (final answer
    # is appended here as an AIMessage so the served endpoint can return it).
    messages: Annotated[list, add_messages]

    # Ordered list of atomic steps the planner produced from the user question.
    plan: list[str]

    # Which step in `plan` is currently being executed. The RAG/MCP nodes advance
    # this so the supervisor knows when every step is done.
    current_step_index: int

    # Results collected from completed steps (one string per finished step). The
    # synthesizer combines these into the final answer.
    step_results: list[str]

    # The supervisor's routing decision for the current step:
    # "rag_agent" | "mcp_tools" | "synthesizer".
    next_agent: str

    # The synthesized final answer (also mirrored into `messages` as an AIMessage).
    final_answer: str
