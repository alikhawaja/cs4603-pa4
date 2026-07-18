"""Bonus B — ChatAgent wrapper (v2 interface) for the Document Analyst.

`agents.deploy()` (and the AI Playground / Review App) require the model's output schema to
be a ChatCompletion/StringResponse, i.e. the MLflow **ChatAgent** interface. Our Part 2
model (`agent_model.py`) logs the bare graph, whose output is raw LangGraph state — not
agent-framework compatible. This file wraps the SAME graph as a `ChatAgent` so the output
becomes a proper `ChatAgentResponse`.

Serialised via models-from-code: ends with `mlflow.models.set_model(agent)`.
"""

from __future__ import annotations

import os
import uuid

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm, get_settings
from rag.store import get_retriever

# Validate configuration at import (same fail-fast as agent_model.py).
get_settings()
_missing = [n for n in ("VECTOR_SEARCH_ENDPOINT", "VECTOR_SEARCH_INDEX") if not os.environ.get(n)]
if _missing:
    raise OSError(f"Missing required environment variables for retrieval: {', '.join(_missing)}")

import tools  # noqa: E402  (shipped via code_paths)

_MCP_SERVER = os.path.join(os.path.dirname(os.path.abspath(tools.__file__)), "mcp_server.py")


class DocumentAnalystChatAgent(ChatAgent):
    """Wraps the LangGraph Document Analyst as an MLflow ChatAgent."""

    def __init__(self) -> None:
        self._graph = build_graph(
            llm=get_chat_llm(),
            retriever=get_retriever(),
            tools=load_mcp_tools(_MCP_SERVER),
        )

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context=None,
        custom_inputs=None,
    ) -> ChatAgentResponse:
        # ChatAgentMessage -> the {role, content} dicts our graph consumes.
        graph_input = {"messages": [{"role": m.role, "content": m.content} for m in messages]}
        result = self._graph.invoke(graph_input)
        answer = result["messages"][-1].content
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=answer, id=str(uuid.uuid4()))]
        )


mlflow.models.set_model(DocumentAnalystChatAgent())
