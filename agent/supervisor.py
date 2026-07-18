"""Supervisor node + routing edge (Task 1.3).

The supervisor is the loop controller. After the planner, control returns here before
every step. It does two things:

  1. If all planned steps are done -> route to the synthesizer.
     Otherwise, classify the current step as needing document retrieval ("rag_agent")
     or calculation ("mcp_tools") and record that in `next_agent`.

  2. `route_from_supervisor` is the conditional-edge function the graph uses to actually
     branch. It simply reads `next_agent` and returns the node name to jump to.

Separation of concerns: the *node* makes the decision (an LLM/keyword classification and
a "are we done?" check); the *edge function* is pure and just maps that decision to a
destination. Keeping the edge function side-effect free is what makes the graph easy to
reason about and test.
"""

from __future__ import annotations

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"

# Keyword hints used as a deterministic fallback if the LLM answer is unclear. Keeping a
# non-LLM backstop means an ambiguous classification still routes somewhere sensible
# instead of crashing.
_MATH_HINTS = (
    "calculate", "compute", "growth", "cagr", "percentage", "percent", "%",
    "increase", "decrease", "compare", "ratio", "convert", "multiply", "sum",
    "difference", "average", "project", "×", "cagr",
)


def _classify_step(llm, step: str) -> str:
    """Return 'rag_agent' or 'mcp_tools' for a single step."""
    response = llm.invoke(
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": step},
        ]
    )
    answer = (getattr(response, "content", str(response)) or "").strip().lower()

    if MCP in answer:
        return MCP
    if RAG in answer:
        return RAG

    # LLM was unclear -> deterministic keyword fallback.
    if any(h in step.lower() for h in _MATH_HINTS):
        return MCP
    return RAG


def make_supervisor(llm):
    """Build the supervisor node bound to a chat `llm`."""

    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)

        # All steps executed -> hand off to synthesis.
        if idx >= len(plan):
            return {"next_agent": SYNTH}

        return {"next_agent": _classify_step(llm, plan[idx])}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    """Conditional-edge function: map the supervisor's decision to a graph node."""
    nxt = state.get("next_agent", "")
    if nxt in (RAG, MCP, SYNTH):
        return nxt
    # Defensive default: if routing is somehow unset, end the loop via synthesis
    # rather than looping forever.
    return SYNTH


def main() -> None:  # pragma: no cover - manual smoke check
    """Quick offline check of the routing edge (no LLM needed)."""
    assert route_from_supervisor({"next_agent": "rag_agent"}) == "rag_agent"
    assert route_from_supervisor({"next_agent": "mcp_tools"}) == "mcp_tools"
    assert route_from_supervisor({"next_agent": "synthesizer"}) == "synthesizer"
    assert route_from_supervisor({}) == "synthesizer"
    print("route_from_supervisor OK")


if __name__ == "__main__":
    main()
