"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from agent.state import AnalystState
from agent.prompts import SYNTHESIZER_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        context = '\n\n'.join([f"Step {i+1}: {result}" for i, result in enumerate(state["step_results"])])

        system_prompt = SYNTHESIZER_PROMPT

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}\n Provide the final answer")
        ])

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)]
        }

    return synthesizer
