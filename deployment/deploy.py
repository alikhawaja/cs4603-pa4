"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

_ROOT = Path(__file__).resolve().parent.parent
_AGENT_MODEL_PATH = "deployment/agent_model.py"

_PIP_REQUIREMENTS = [
    "mlflow", "langgraph", "langchain-core", "langchain-openai",
    "databricks-langchain", "databricks-vectorsearch", "databricks-sdk",
    "langchain-mcp-adapters", "mcp", "openai", "pydantic", "python-dotenv",
]


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise OSError(f"Missing required environment variable: {name}")
    return val


def log_and_register() -> tuple[str, str]:
    uc_catalog = _require("UC_CATALOG")
    uc_schema = _require("UC_SCHEMA")
    model_name = _require("SERVING_ENDPOINT_NAME").replace("-", "_")
    uc_name = f"{uc_catalog}.{uc_schema}.{model_name}"

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Users/{_current_user()}/pa4-document-analyst")

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model=_AGENT_MODEL_PATH,
            name="agent",
            code_paths=[
                str(_ROOT / "agent"), str(_ROOT / "rag"),
                str(_ROOT / "tools"), str(_ROOT / "config.py"),
            ],
            pip_requirements=_PIP_REQUIREMENTS,
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
        )

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered model: {uc_name}, version {registered.version}")
    return uc_name, registered.version


def _current_user() -> str:
    try:
        w = WorkspaceClient()
        return w.current_user.me().user_name
    except Exception:
        return os.environ.get("DATABRICKS_HOST", "unknown-user")


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    endpoint_name = _require("SERVING_ENDPOINT_NAME")
    secret_scope = _require("SECRET_SCOPE")

    w = WorkspaceClient()

    config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(
                entity_name=uc_name,
                entity_version=version,
                workload_size="Small",
                scale_to_zero_enabled=True,
                environment_vars={
                    "DATABRICKS_HOST": f"{{{{secrets/{secret_scope}/DATABRICKS_HOST}}}}",
                    "DATABRICKS_TOKEN": f"{{{{secrets/{secret_scope}/DATABRICKS_TOKEN}}}}",
                    "DATABRICKS_MODEL": f"{{{{secrets/{secret_scope}/DATABRICKS_MODEL}}}}",
                    "VECTOR_SEARCH_ENDPOINT": _require("VECTOR_SEARCH_ENDPOINT"),
                    "VECTOR_SEARCH_INDEX": _require("VECTOR_SEARCH_INDEX"),
                    "EMBEDDINGS_ENDPOINT": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
                },
            )
        ]
    )

    existing = {e.name for e in w.serving_endpoints.list()}
    if endpoint_name in existing:
        print(f"Endpoint '{endpoint_name}' exists — updating served entities...")
        w.serving_endpoints.update_config_and_wait(name=endpoint_name, served_entities=config.served_entities)
    else:
        print(f"Creating endpoint '{endpoint_name}'...")
        w.serving_endpoints.create_and_wait(name=endpoint_name, config=config)

    for _ in range(60):
        status = w.serving_endpoints.get(endpoint_name)
        state = status.state.ready if status.state else None
        if str(state) == "EndpointStateReady.READY":
            break
        time.sleep(10)

    url = f"{_require('DATABRICKS_HOST').rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint READY: {url}")
    return url


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)


