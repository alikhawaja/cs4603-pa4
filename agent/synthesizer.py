"""Synthesizer node (Task 1.6).

The final node. It combines every `step_result` into one coherent, cited answer to the
user's original question.

CRITICAL — messages in / messages out. The answer is written to BOTH:
  - `final_answer` (internal convenience field), and
  - the `messages` channel as an `AIMessage` (via the add_messages reducer).

The deployed endpoint (Part 2) reads its response from the LAST message in the returned
state. If we only set `final_answer` and never append an AIMessage, the endpoint returns
an empty answer even though local state looks correct — this is the #1 "empty completion"
deployment bug the spec warns about.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _original_question(messages: list) -> str:
    """Return the first human/user message — the user's original question."""
    for msg in messages:
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            return getattr(msg, "content", None) or msg["content"]
    last = messages[-1]
    return getattr(last, "content", None) or last["content"]


def make_synthesizer(llm):
    """Build the synthesizer node bound to a chat `llm`."""

    def synthesizer(state: AnalystState) -> dict:
        question = _original_question(state["messages"])
        results = state.get("step_results", [])
        results_block = "\n".join(results) if results else "(no step results were produced)"

        response = llm.invoke(
            [
                {"role": "system", "content": SYNTHESIZER_PROMPT},
                {
                    "role": "user",
                    "content": f"QUESTION: {question}\n\nSTEP RESULTS:\n{results_block}",
                },
            ]
        )
        answer = (getattr(response, "content", str(response)) or "").strip()

        # Write to BOTH channels. `messages` is what the served endpoint reads back.
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return synthesizer
