"""MLflow models-from-code definition (Task 2.1).

MLflow serialises this FILE, not a pickled object: the serving container
re-executes it at startup, so everything it needs must be importable from the
packaged code (`code_paths`) or installed via `pip_requirements`, and all
configuration must come from environment variables.

Env vars are validated up front so a misconfigured endpoint fails with a log
line naming the missing variable instead of a cryptic DEPLOYMENT_FAILED.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import os

import mlflow
from dotenv import load_dotenv

load_dotenv()  # local runs read .env; the container has real env vars instead

_REQUIRED_ENV = [
    # LLM access (secret-scope references on the endpoint):
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_MODEL",
    # Vector Search retrieval (plaintext env vars on the endpoint):
    "VECTOR_SEARCH_ENDPOINT",
    "VECTOR_SEARCH_INDEX",
]

_missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
if _missing:
    raise OSError(
        f"Missing required environment variables: {', '.join(_missing)}. "
        "Locally: set them in .env. Deployed: pass them via the endpoint's "
        "environment_vars (secrets as {{secrets/<scope>/<key>}} references)."
    )

# Production graph: Databricks LLM + managed Vector Search retriever + the
# bundled stdio MCP server (its path is resolved relative to the packaged
# code inside agent/graph.py, so it works both locally and in the container).
from agent.graph import build_graph  # noqa: E402

graph = build_graph()

mlflow.models.set_model(graph)
