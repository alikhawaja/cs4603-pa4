"""Synthesizer node (Task 1.6).

Combines all step results into one cited answer. The answer is written to
BOTH `final_answer` and the `messages` channel: the deployed endpoint reads
the last message of the returned state, so an answer that only lives in
`final_answer` would surface as an empty completion (spec Task 1.6 / guide §5).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.planner import get_user_question
from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        question = get_user_question(state)
        plan = state.get("plan", [])
        results = state.get("step_results", [])

        plan_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan)) or "(no plan)"
        results_block = "\n".join(f"- {r}" for r in results) or "(no step results)"
        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Plan:\n{plan_block}\n\n"
                        f"Step results:\n{results_block}"
                    )
                ),
            ]
        )
        content = response.content if hasattr(response, "content") else str(response)
        answer = str(content).strip()
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    return synthesizer
