"""Bonus B — deploy via the databricks-agents SDK.

Run:  uv run python deployment/deploy_agents.py

Reuses the ENTIRE manual pipeline from deploy.py (models-from-code definition,
code_paths, pinned pip_requirements, Unity Catalog registration) and swaps only
the final WorkspaceClient endpoint step for one `agents.deploy(...)` call,
which auto-provisions the serving endpoint AND a Review App, and handles
credential injection itself (no secret scope wiring).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def log_and_register_chat_agent() -> tuple[str, str]:
    """Log the ChatAgent wrapper (agents.deploy needs a chat schema, not the
    raw LangGraph state that Part 2's model returns) and register it in UC."""
    from importlib.metadata import version as pkg_version

    import mlflow
    from databricks.sdk import WorkspaceClient

    from deployment.deploy import (
        _PACKAGES,
        ROOT,
        _make_model_code_path_portable,
    )

    _make_model_code_path_portable()
    user = WorkspaceClient().current_user.me().user_name
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Users/{user}/pa4-document-analyst")

    with mlflow.start_run(run_name="document-analyst-chatagent"):
        model_info = mlflow.pyfunc.log_model(
            python_model=os.path.join(ROOT, "deployment", "agent_model_chat.py"),
            name="agent",
            code_paths=[
                os.path.join(ROOT, "agent"),
                os.path.join(ROOT, "rag"),
                os.path.join(ROOT, "tools"),
                os.path.join(ROOT, "config.py"),
            ],
            pip_requirements=[f"{p}=={pkg_version(p)}" for p in _PACKAGES],
            input_example={
                "messages": [{"role": "user", "content": "What was the net income in 2023?"}]
            },
        )

    uc_name = f"{os.environ['UC_CATALOG']}.{os.environ['UC_SCHEMA']}.zakariya_document_analyst_chat"
    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered {uc_name} version {registered.version}")
    return uc_name, registered.version


def main() -> None:
    from databricks import agents

    uc_name, version = log_and_register_chat_agent()

    environment_vars = {
            # agents.deploy injects its own OAuth credentials for Databricks
            # resources, but our model reads DATABRICKS_HOST/TOKEN explicitly
            # (ChatOpenAI + VectorSearchClient), so supply them as secret refs
            # exactly like Part 2. The retriever/MCP settings are plaintext.
            "DATABRICKS_HOST": "{{secrets/cs4603-deploy/DATABRICKS_HOST}}",
            "DATABRICKS_TOKEN": "{{secrets/cs4603-deploy/DATABRICKS_TOKEN}}",
            "DATABRICKS_MODEL": "{{secrets/cs4603-deploy/DATABRICKS_MODEL}}",
            "VECTOR_SEARCH_ENDPOINT": os.environ["VECTOR_SEARCH_ENDPOINT"],
            "VECTOR_SEARCH_INDEX": os.environ["VECTOR_SEARCH_INDEX"],
            "EMBEDDINGS_ENDPOINT": os.environ.get(
                "EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"
            ),
        }
    # Bonus C parity: route the agents-deployed container's tool calls to the
    # remote MCP app as well, when one is configured.
    if os.environ.get("MCP_SERVER_URL"):
        environment_vars["MCP_SERVER_URL"] = os.environ["MCP_SERVER_URL"]
        environment_vars["MCP_CLIENT_ID"] = "{{secrets/cs4603-deploy/MCP_CLIENT_ID}}"
        environment_vars["MCP_CLIENT_SECRET"] = "{{secrets/cs4603-deploy/MCP_CLIENT_SECRET}}"

    deployment = agents.deploy(
        model_name=uc_name,
        model_version=version,
        scale_to_zero=True,
        environment_vars=environment_vars,
    )
    print(f"endpoint_name:  {deployment.endpoint_name}")
    print(f"query_endpoint: {deployment.query_endpoint}")
    print(f"review_app_url: {deployment.review_app_url}")


if __name__ == "__main__":
    main()
