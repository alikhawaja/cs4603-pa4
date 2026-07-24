"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import os

import nest_asyncio
import mlflow

from agent.graph import build_graph
from config import get_chat_llm, get_settings
from rag.store import get_retriever

nest_asyncio.apply()

required_vars = ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL"]
missing = [name for name in required_vars if not os.environ.get(name)]
if missing:
    raise OSError(
        f"Missing required environment variables for deployment: {', '.join(missing)}"
    )

settings = get_settings()
llm = get_chat_llm()
retriever = get_retriever()

graph = build_graph(llm=llm, retriever=retriever, tools=None)

mlflow.models.set_model(graph)


