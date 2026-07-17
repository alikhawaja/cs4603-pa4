"""Supervisor node + routing edge (Task 1.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

# Node-name constants — shared with graph.py so routing never relies on
# magic strings scattered across files.
RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


class RouteDecision(BaseModel):
    """Structured output schema for the supervisor's routing decision."""

    next_agent: Literal["rag_agent", "mcp_tools"] = Field(
        description="Which specialist should execute this step"
    )


_CALC_KEYWORDS = (
    "calculate", "compute", "growth", "percent", "%", "compare",
    "convert", "cagr", "multiply", "divide", "ratio",
)


def _keyword_route(step: str) -> str:
    """Fallback routing when structured output fails or is unavailable."""
    step_lower = step.lower()
    if any(kw in step_lower for kw in _CALC_KEYWORDS):
        return MCP
    return RAG


def make_supervisor(llm):
    """Return a supervisor node bound to the given chat LLM."""

    def supervisor(state: AnalystState) -> dict:
        plan = state["plan"]
        idx = state["current_step_index"]

        if idx >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[idx]
        try:
            structured_llm = llm.with_structured_output(RouteDecision)
            decision = structured_llm.invoke(
                [
                    {"role": "system", "content": SUPERVISOR_PROMPT},
                    {"role": "user", "content": step},
                ]
            )
            next_agent = decision.next_agent
        except Exception:
            next_agent = _keyword_route(step)

        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    """Conditional edge function: map next_agent directly to the node name."""
    return state["next_agent"]