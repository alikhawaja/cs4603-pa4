# deployment/minimal_test_model.py — updated, combined test
from __future__ import annotations

import mlflow
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from rag.store import get_retriever
from config import get_chat_llm

retriever = get_retriever()
llm = get_chat_llm()


class MinimalState(TypedDict):
    messages: list


def echo_node(state: MinimalState) -> dict:
    return {"messages": state["messages"] + [{"role": "assistant", "content": "test response"}]}


builder = StateGraph(MinimalState)
builder.add_node("echo", echo_node)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()

mlflow.models.set_model(graph)