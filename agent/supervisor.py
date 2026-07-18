"""Supervisor node + routing edge (Task 1.3).

Routes the current plan step to a specialist. Completion is decided by code
(index vs. plan length), not by the LLM — the model only classifies a single
step, which is a task small models get right far more reliably than managing
loop state. If the LLM reply is unusable, a keyword heuristic breaks the tie
so the graph always makes progress.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"

_CALC_HINTS = re.compile(
    r"\b(calculat|comput|project|growth|increase|decrease|percent|%|multiply|divide|"
    r"add|subtract|compare|convert|ratio|cagr|compound|sum|difference)\w*",
    flags=re.IGNORECASE,
)


def classify_step_heuristic(step: str) -> str:
    """Deterministic fallback when the LLM routing reply can't be parsed."""
    return MCP if _CALC_HINTS.search(step) else RAG


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)
        if index >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[index]
        response = llm.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=step)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        reply = str(content).strip().lower()

        if MCP in reply:
            decision = MCP
        elif RAG in reply:
            decision = RAG
        elif SYNTH in reply:
            # LLM says the step needs no new fact/math (pure presentation):
            # skip it and let the completion check fire on the next visit.
            return {"next_agent": SYNTH}
        else:
            decision = classify_step_heuristic(step)
        return {"next_agent": decision}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]
