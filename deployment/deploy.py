"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""


from __future__ import annotations

import os
import time
from datetime import timedelta

os.environ["MLFLOW_UV_AUTO_DETECT"] = "false"
import mlflow
from config import get_settings
from databricks.sdk.errors.platform import ResourceConflict
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex


def log_and_register():
    s = get_settings()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(root)

    experiment_name = "/Users/alifayyaz0613@gmail.com/document-analyst"
    experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model_info = mlflow.langchain.log_model(
            lc_model=os.path.join(root, "agent_model.py"),
            name="agent", 
            code_paths=[   
                os.path.join(project_root, "agent"),
                os.path.join(project_root, "rag"),
                os.path.join(project_root, "tools"),
                os.path.join(project_root, "config.py"),
            ],
            pip_requirements=[
                "mlflow==3.14.0",
                "langgraph==1.2.9",
                "langchain-core==1.4.9",
                "langchain-openai==1.3.5",
                "langchain-mcp-adapters==0.3.0",
                "databricks-langchain==0.20.0",
                "databricks-vectorsearch==0.75",
                "mcp==1.28.1",
                "python-dotenv==1.2.2",
                "nest-asyncio==1.6.0",
                "langchain-mcp-adapters==0.3.0",
            ],
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
            resources=[
            DatabricksServingEndpoint(endpoint_name=os.environ["DATABRICKS_MODEL"]),
            DatabricksVectorSearchIndex(index_name=os.environ["VECTOR_SEARCH_INDEX"]),
        ],
        )
    
    catalog = os.environ["UC_CATALOG"]
    schema = os.environ["UC_SCHEMA"]
    model_name = f"{catalog}.{schema}.document_analyst"
    registered = mlflow.register_model(model_info.model_uri, model_name)

    print(f"Registered model: {model_name}, version {registered.version}")

    return model_name, registered.version

def _wait_for_endpoint_ready(workspace_client, endpoint_name: str, timeout_seconds: int = 7200, poll_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = workspace_client.serving_endpoints.get(endpoint_name)
        ready = str(status.state.ready)
        config_update = str(status.state.config_update)
        pending = status.pending_config
        served_entities = getattr(pending, "served_entities", None) or []
        print(f"ready={ready}, config_update={config_update}")
        if served_entities:
            for entity in served_entities:
                state = getattr(getattr(entity, "state", None), "deployment", None)
                message = getattr(getattr(entity, "state", None), "deployment_state_message", None)
                print(f"  served_entity={entity.name}: deployment={state}, message={message}")

        if ready == "EndpointStateReady.READY":
            return
        if "FAILED" in ready or "FAILED" in config_update:
            raise RuntimeError("Deployment FAILED — check the Logs tab in the Databricks UI.")

        time.sleep(poll_seconds)

    raise TimeoutError(f"Endpoint {endpoint_name} did not become READY within {timeout_seconds} seconds.")


def _update_endpoint_config(workspace_client, endpoint_name: str, served_entities: list, max_retries: int = 3) -> None:
    for attempt in range(max_retries):
        try:
            workspace_client.serving_endpoints.update_config_and_wait(
                name=endpoint_name,
                served_entities=served_entities,
                timeout=timedelta(minutes=20),
            )
            return
        except ResourceConflict:
            if attempt == max_retries - 1:
                raise
            print("Endpoint update already in progress; waiting for the current update to finish...")
            workspace_client.serving_endpoints.wait_get_serving_endpoint_not_updating(
                name=endpoint_name,
                timeout=timedelta(minutes=20),
            )
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            print(f"Endpoint update failed ({exc}); retrying...")
            time.sleep(30)


def _create_endpoint(workspace_client, endpoint_name: str, served_entities: list) -> None:
    from databricks.sdk.service.serving import EndpointCoreConfigInput

    workspace_client.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=served_entities,
        ),
        timeout=timedelta(minutes=20),
    )


def _delete_endpoint(workspace_client, endpoint_name: str) -> None:
    try:
        workspace_client.serving_endpoints.delete(endpoint_name)
    except Exception as exc:
        print(f"Delete skipped for {endpoint_name}: {exc}")


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    print("1. Creating WorkspaceClient...")
    w = WorkspaceClient()
    endpoint_name = os.environ["SERVING_ENDPOINT_NAME"]
    print("2. WorkspaceClient created")

    print("3. Endpoint name:", endpoint_name)
    served_entities = [
        ServedEntityInput(
            entity_name=uc_name,
            entity_version=version,
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_HOST": "{{secrets/cs4603-deploy/DATABRICKS_HOST}}",
                "DATABRICKS_TOKEN": "{{secrets/cs4603-deploy/DATABRICKS_TOKEN}}",
                "DATABRICKS_MODEL": "{{secrets/cs4603-deploy/DATABRICKS_MODEL}}",
                "VECTOR_SEARCH_ENDPOINT": os.environ["VECTOR_SEARCH_ENDPOINT"],
                "VECTOR_SEARCH_INDEX": os.environ["VECTOR_SEARCH_INDEX"],
                "EMBEDDINGS_ENDPOINT": os.environ["EMBEDDINGS_ENDPOINT"],
            },
        )
    ]

    print("4. Listing endpoints...")
    endpoints = list(w.serving_endpoints.list())
    print(f"5. Found {len(endpoints)} endpoints")

    existing = [e for e in endpoints if e.name == endpoint_name]
    print("6. Existing endpoint:", bool(existing))

    if existing:
        try:
            _update_endpoint_config(w, endpoint_name, served_entities)
        except Exception as exc:
            print(f"Update path failed, recreating endpoint: {exc}")
            _delete_endpoint(w, endpoint_name)
            time.sleep(20)
            _create_endpoint(w, endpoint_name, served_entities)
    else:
        _create_endpoint(w, endpoint_name, served_entities)

    _wait_for_endpoint_ready(w, endpoint_name)

    url = f"{os.environ['DATABRICKS_HOST']}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint URL: {url}")
    return url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
    print(f"::notice::Deployed model {name} version {ver}")
    print(f"::notice::Endpoint status: READY at {name}")
