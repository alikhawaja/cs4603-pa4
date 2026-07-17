"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations
from langchain.messages import SystemMessage, HumanMessage
from agent.state import AnalystState

import json

def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        system_prompt = """
        You are planner that receives user query and produces a list of 2-5 atomic steps.
        Only output JSON list of step strings
        """

        user_message = state["messages"][-1].content

        response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
        ])

        try:
            plan = json.loads(response.content)
        except:
            plan = [state['messages'][-1].content]
        
        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": []
        }
    
    return planner
