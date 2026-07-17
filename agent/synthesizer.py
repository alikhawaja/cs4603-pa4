"""Synthesizer node (Task 1.6)."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _extract_question(state: AnalystState) -> str:
    """Pull the latest human question out of the messages channel."""
    for msg in reversed(state["messages"]):
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            return getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else ""
            )
    last = state["messages"][-1]
    return getattr(last, "content", None) or last.get("content", "")


def make_synthesizer(llm):
    """Return a synthesizer node bound to the given chat LLM."""

    def synthesizer(state: AnalystState) -> dict:
        question = _extract_question(state)
        step_results = state.get("step_results", [])
        context = "\n".join(step_results) if step_results else "(no step results produced)"

        response = llm.invoke(
            [
                {"role": "system", "content": SYNTHESIZER_PROMPT},
                {
                    "role": "user",
                    "content": f"Original question: {question}\n\nStep results:\n{context}",
                },
            ]
        )
        final_answer = response.content

        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }

    return synthesizer
