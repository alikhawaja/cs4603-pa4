"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

Pipeline (same as databricks_deployment_v1, plus the PA4 add-ons):
    mlflow.langchain.log_model (models-from-code, code_paths, pinned pips)
      -> mlflow.register_model into Unity Catalog
      -> WorkspaceClient serving endpoint (Small, scale-to-zero, secret refs)
      -> poll until READY, print the endpoint URL
"""

from __future__ import annotations

import os
import time
from importlib.metadata import version as pkg_version

import mlflow
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_model_code_path_portable() -> None:
    """Record `model_code_path` as a bare filename in the MLmodel metadata.

    MLflow resolves the models-from-code path with `Path(...).resolve()`, so
    logging from Windows bakes a backslashed absolute path into the MLmodel.
    The Linux serving container then computes `posixpath.basename(...)` on it,
    which cannot split backslashes, and model loading dies with an
    MlflowException before any user code runs. The loader only ever joins the
    basename onto the model directory, so storing just the filename is both
    sufficient and portable across operating systems.
    """
    from mlflow.models import Model

    original_save = Model.save

    def save_portable(self, path):
        for flavor in self.flavors.values():
            if isinstance(flavor, dict) and flavor.get("model_code_path"):
                flavor["model_code_path"] = os.path.basename(
                    flavor["model_code_path"].replace("\\", "/")
                )
        return original_save(self, path)

    Model.save = save_portable

# Everything the serving container must pip-install. Pinned to the locally
# verified versions so requirement inference can't drift (troubleshooting
# table: missing databricks-vectorsearch / langchain-mcp-adapters).
_PACKAGES = [
    "mlflow",
    "langgraph",
    "langchain",
    "langchain-core",
    "langchain-openai",
    "databricks-langchain",
    "databricks-vectorsearch",
    "databricks-sdk",
    "mcp",
    "langchain-mcp-adapters",
    "openai",
    "python-dotenv",
]

INPUT_EXAMPLE = {
    "messages": [{"role": "user", "content": "What was the net income in 2023?"}]
}


def _uc_model_name() -> str:
    catalog = os.environ["UC_CATALOG"]
    schema = os.environ["UC_SCHEMA"]
    model = os.environ.get("UC_MODEL_NAME", "zakariya_document_analyst")
    return f"{catalog}.{schema}.{model}"


def log_and_register() -> tuple[str, str]:
    """Log the models-from-code model and register it in Unity Catalog."""
    from databricks.sdk import WorkspaceClient

    _make_model_code_path_portable()
    user = WorkspaceClient().current_user.me().user_name
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Users/{user}/pa4-document-analyst")

    with mlflow.start_run(run_name="document-analyst"):
        model_info = mlflow.langchain.log_model(
            # Forward slashes are vital when logging from Windows: the Linux
            # serving container resolves this with posixpath.basename, and a
            # backslashed path makes model loading fail (MlflowException).
            lc_model=os.path.join(ROOT, "deployment", "agent_model.py").replace(os.sep, "/"),
            name="agent",
            code_paths=[  # ship the local packages the container must import
                os.path.join(ROOT, "agent"),
                os.path.join(ROOT, "rag"),
                os.path.join(ROOT, "tools"),
                os.path.join(ROOT, "config.py"),
            ],
            pip_requirements=[f"{p}=={pkg_version(p)}" for p in _PACKAGES],
            input_example=INPUT_EXAMPLE,
        )

    uc_name = _uc_model_name()
    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered {uc_name} version {registered.version}")
    return uc_name, registered.version


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    """Create/update the Model Serving endpoint and wait for READY."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    endpoint_name = os.environ.get("SERVING_ENDPOINT_NAME", "zakariya-document-analyst")
    scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")
    settings_env = {
        # Secrets — only ever referenced, never inlined:
        "DATABRICKS_HOST": f"{{{{secrets/{scope}/DATABRICKS_HOST}}}}",
        "DATABRICKS_TOKEN": f"{{{{secrets/{scope}/DATABRICKS_TOKEN}}}}",
        "DATABRICKS_MODEL": f"{{{{secrets/{scope}/DATABRICKS_MODEL}}}}",
        # Not secrets — the retriever needs these to reach the index:
        "VECTOR_SEARCH_ENDPOINT": os.environ["VECTOR_SEARCH_ENDPOINT"],
        "VECTOR_SEARCH_INDEX": os.environ["VECTOR_SEARCH_INDEX"],
        "EMBEDDINGS_ENDPOINT": os.environ.get(
            "EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"
        ),
    }
    # Bonus C — when a remote MCP server is configured, the container calls it
    # over HTTPS instead of spawning the bundled stdio subprocess. The service
    # principal credentials (used to mint the app OAuth token) are secrets.
    mcp_url = os.environ.get("MCP_SERVER_URL")
    if mcp_url:
        settings_env["MCP_SERVER_URL"] = mcp_url
        settings_env["MCP_CLIENT_ID"] = f"{{{{secrets/{scope}/MCP_CLIENT_ID}}}}"
        settings_env["MCP_CLIENT_SECRET"] = f"{{{{secrets/{scope}/MCP_CLIENT_SECRET}}}}"
    served_entity = ServedEntityInput(
        entity_name=uc_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars=settings_env,
    )

    w = WorkspaceClient()
    existing = {e.name for e in w.serving_endpoints.list()}
    if endpoint_name in existing:
        print(f"Updating endpoint {endpoint_name} -> version {version} ...")
        w.serving_endpoints.update_config(
            name=endpoint_name, served_entities=[served_entity]
        )
    else:
        print(f"Creating endpoint {endpoint_name} ...")
        w.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name, served_entities=[served_entity]
            ),
        )

    deadline = time.time() + 40 * 60
    while True:
        info = w.serving_endpoints.get(endpoint_name)
        ready = getattr(info.state.ready, "value", str(info.state.ready))
        update = getattr(info.state.config_update, "value", str(info.state.config_update))
        print(f"  state: ready={ready} config_update={update}")
        if update == "UPDATE_FAILED":
            raise RuntimeError(
                f"Deployment failed — check Serving > {endpoint_name} > Logs"
            )
        if ready == "READY" and update == "NOT_UPDATING":
            break
        if time.time() > deadline:
            raise TimeoutError("Endpoint not READY after 40 minutes")
        time.sleep(30)

    url = f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint READY: {url}")
    return url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
