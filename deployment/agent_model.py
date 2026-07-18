"""MLflow models-from-code definition (Task 2.1).

This is the file MLflow serialises and runs inside the serving container. It must be
self-contained in the sense that everything it needs is either imported from packages
shipped via `code_paths` (agent, rag, tools, config) or reachable as a managed Databricks
service (the LLM endpoint and the Vector Search index) — never from local-only state.

At import time it:
  1. Validates the required environment variables and fails with a clear message (so the
     Serving Logs name the missing var instead of a cryptic DEPLOYMENT_FAILED).
  2. Rebuilds the graph with production clients:
       - LLM:       ChatOpenAI pointed at $DATABRICKS_HOST/serving-endpoints (config.py).
       - Retriever: DatabricksVectorSearch over the managed index (rag/store.py).
       - Tools:     the bundled stdio MCP server (tools/mcp_server.py).
  3. Calls `mlflow.models.set_model(graph)` so MLflow knows what to serve.

Must import cleanly (with .env loaded / endpoint env vars set):
    python -c "import deployment.agent_model"
"""

from __future__ import annotations

import os

import mlflow

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm, get_settings
from rag.store import get_retriever

# ── 1. Fail fast on missing configuration ───────────────────────────────────
# get_settings() raises OSError naming any missing DATABRICKS_HOST/TOKEN/MODEL. We also
# require the Vector Search vars, since the retriever cannot reach the index without them.
_settings = get_settings()
_missing = [
    name
    for name in ("VECTOR_SEARCH_ENDPOINT", "VECTOR_SEARCH_INDEX")
    if not os.environ.get(name)
]
if _missing:
    raise OSError(
        "Missing required environment variables for retrieval: "
        + ", ".join(_missing)
        + ". Set them in your .env (local) or the endpoint environment_vars (deployed)."
    )

# ── 2. Resolve the bundled MCP server via the importable `tools` package ─────
# MLflow places code_paths under a `code/` subdirectory in the serving artifact, so a
# path computed relative to THIS file would be wrong there. Instead resolve the server
# from the `tools` package itself, which is on sys.path in both local and container runs.
import tools  # noqa: E402  (shipped via code_paths)

_MCP_SERVER = os.path.join(os.path.dirname(os.path.abspath(tools.__file__)), "mcp_server.py")

# ── 3. Build the graph with production clients and register it with MLflow ───
graph = build_graph(
    llm=get_chat_llm(),
    retriever=get_retriever(),
    tools=load_mcp_tools(_MCP_SERVER),
)

mlflow.models.set_model(graph)
