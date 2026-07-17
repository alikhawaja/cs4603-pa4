"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""

from __future__ import annotations
from langchain.messages import SystemMessage, HumanMessage
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:

        if state["current_step_index"] >= len(state["plan"]):
            return {"next_agent": "synthesizer"}
        
        system_prompt = f"""
        You are supervisor. Classify the steps as either:
        {RAG} - if it requires looking up facts from documents
        {MCP} - if it requires calculation or numerical analysis

        reply with a single word only {RAG} or {MCP}
        """

        current_step = state["plan"][state["current_step_index"]]

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=current_step)
        ])

        agent = response.content.strip().lower()
        next_agent = "mcp_tools" if "mcp" in agent else "rag_agent"

        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]
