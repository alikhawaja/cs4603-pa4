"""MLflow models-from-code definition (Task 2.1)."""

from __future__ import annotations

import os

import mlflow

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever

import tools as _tools_pkg


_REQUIRED_ENV_VARS = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL")
_missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
if _missing:
    raise OSError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        "Set these in the serving endpoint's environment_vars (see "
        "deployment/deploy.py) or in your local .env for local testing."
    )


_MCP_SERVER_PATH = os.path.join(os.path.dirname(_tools_pkg.__file__), "mcp_server.py")

graph = build_graph(
    llm=get_chat_llm(),
    retriever=get_retriever(),
    tools=load_mcp_tools(_MCP_SERVER_PATH),
)

mlflow.models.set_model(graph)
