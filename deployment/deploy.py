"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

Task 2.2 (this file, `log_and_register`):
  - point MLflow at the Databricks workspace + the Unity Catalog registry,
  - log `deployment/agent_model.py` with models-from-code, shipping the local packages
    via `code_paths` (without this the endpoint fails at startup with
    `ModuleNotFoundError: No module named 'agent'` — the #1 PA4 deployment error),
  - register the logged model as a new version in Unity Catalog.

Task 2.3 (`create_or_update_endpoint`) then serves that version.
"""

from __future__ import annotations

import os
from datetime import timedelta

import mlflow
from dotenv import load_dotenv

# Load .env so MLflow / the Databricks SDK see DATABRICKS_HOST/TOKEN when run as a script.
load_dotenv()

# Repo root = parent of this deployment/ directory. Used to build absolute paths so the
# script works no matter what directory it is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Local packages the serving container must import. `config.py` is a top-level module.
CODE_PATHS = [
    os.path.join(ROOT, "agent"),
    os.path.join(ROOT, "rag"),
    os.path.join(ROOT, "tools"),
    os.path.join(ROOT, "config.py"),
]

# Pin EXACT versions matching the working local env. Unpinned requirements send the
# serving image build into pip resolution-backtracking (it source-builds hundreds of old
# `regex` releases and hits the build timeout). `regex`/`tiktoken` are transitive but must
# be pinned too, since they are the packages pip backtracks on.
PIP_REQUIREMENTS = [
    "mlflow==3.14.0",
    "langgraph==1.2.9",
    "langchain==1.3.14",
    "langchain-core==1.4.9",
    "langchain-openai==1.3.5",
    "databricks-langchain==0.20.0",
    "databricks-vectorsearch==0.75",
    "langchain-mcp-adapters==0.3.0",
    "mcp==1.28.1",
    "openai==2.46.0",
    # Transitive pins to stop the resolver backtracking on `regex`:
    "regex==2026.7.10",
    "tiktoken==0.13.0",
]

INPUT_EXAMPLE = {"messages": [{"role": "user", "content": "What was the revenue?"}]}


def _force_forward_slash_model_path() -> None:
    """Make MLflow record the models-from-code path with forward slashes.

    When logging on Windows, MLflow records `model_code_path` with backslashes. The Linux
    serving container loads it via `os.path.join(model_dir, os.path.basename(path))`, but
    Linux `os.path.basename` does NOT split on '\\', so the whole Windows path is kept and
    the load fails with FileNotFoundError on '/model/D:\\...\\agent_model.py'. Forward
    slashes are valid on Windows too, so this keeps local logging working while letting the
    container correctly extract 'agent_model.py'. No-op on non-Windows.
    """
    import mlflow.langchain.model as _lc
    import mlflow.pyfunc as _pf

    # langchain flavor (Part 2): wrap the langchain-specific resolver.
    orig_lc = _lc._validate_and_prepare_lc_model_or_path
    if not getattr(orig_lc, "_pa4_patched", False):
        def patched_lc(lc_model, loader_fn, temp_dir=None):
            result = orig_lc(lc_model, loader_fn, temp_dir)
            return result.replace("\\", "/") if isinstance(result, str) else result

        patched_lc._pa4_patched = True
        _lc._validate_and_prepare_lc_model_or_path = patched_lc

    # pyfunc flavor (Bonus B ChatAgent): wrap the shared code-path resolver pyfunc uses.
    orig_pf = getattr(_pf, "_validate_and_get_model_code_path", None)
    if orig_pf is not None and not getattr(orig_pf, "_pa4_patched", False):
        def patched_pf(model_code_path, temp_dir):
            result = orig_pf(model_code_path, temp_dir)
            return result.replace("\\", "/") if isinstance(result, str) else result

        patched_pf._pa4_patched = True
        _pf._validate_and_get_model_code_path = patched_pf


def _uc_model_name() -> str:
    """Fully-qualified Unity Catalog model name: <catalog>.<schema>.<model>."""
    catalog = os.environ.get("UC_CATALOG", "27100082_pa4")
    schema = os.environ.get("UC_SCHEMA", "default")
    model = os.environ.get("UC_MODEL", "document_analyst")
    return f"{catalog}.{schema}.{model}"


def log_and_register() -> tuple[str, str]:
    """Log the model to MLflow and register a new Unity Catalog version.

    Returns (uc_model_name, version).
    """
    # Track to the Databricks workspace and register into Unity Catalog.
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    # Experiments live under the current user's workspace folder.
    from databricks.sdk import WorkspaceClient

    user = WorkspaceClient().current_user.me().user_name
    experiment = os.environ.get("MLFLOW_EXPERIMENT", f"/Users/{user}/pa4-document-analyst")
    mlflow.set_experiment(experiment)

    uc_name = _uc_model_name()
    # Run from the repo root and pass a RELATIVE lc_model path. An absolute path gets
    # baked into the model metadata and becomes an invalid '/model/D:\...' path inside the
    # Linux serving container (FileNotFoundError at load). Relative keeps it portable.
    os.chdir(ROOT)
    _force_forward_slash_model_path()
    print(f"[deploy] logging model -> experiment {experiment}")
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model=os.path.join("deployment", "agent_model.py"),
            name="agent",
            code_paths=CODE_PATHS,
            pip_requirements=PIP_REQUIREMENTS,
            input_example=INPUT_EXAMPLE,
        )

    print(f"[deploy] registering -> {uc_name}")
    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"[deploy] registered {uc_name} version {registered.version}")
    return uc_name, registered.version


def ensure_secrets() -> str:
    """Create the secret scope (idempotent) and store the three credential secrets.

    The serving container has no .env, so credentials are injected as secret references.
    We store DATABRICKS_HOST/TOKEN/MODEL once here from the local environment.
    Returns the scope name.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import DatabricksError

    scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")
    w = WorkspaceClient()

    existing = {s.name for s in w.secrets.list_scopes()}
    if scope not in existing:
        w.secrets.create_scope(scope)
        print(f"[deploy] created secret scope '{scope}'")
    else:
        print(f"[deploy] secret scope '{scope}' already exists")

    for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL"):
        value = os.environ.get(key)
        if not value:
            raise OSError(f"Cannot store secret: {key} is not set in the environment")
        try:
            w.secrets.put_secret(scope=scope, key=key, string_value=value)
        except DatabricksError as exc:
            raise RuntimeError(f"Failed to store secret {key} in scope {scope}: {exc}") from exc
    print(f"[deploy] stored DATABRICKS_HOST/TOKEN/MODEL in scope '{scope}'")
    return scope


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    """Create or update the Model Serving endpoint for the given UC model version.

    Credentials come from the secret scope; the Vector Search vars are plaintext (not
    secrets) so the container's retriever can reach the index. Waits for READY.
    Returns the /invocations URL.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    w = WorkspaceClient()
    endpoint = os.environ.get("SERVING_ENDPOINT_NAME", "27100082-document-analyst")
    scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")
    host = os.environ["DATABRICKS_HOST"].rstrip("/")

    environment_vars = {
        # Secrets — injected as {{secrets/scope/key}} references, never plaintext.
        "DATABRICKS_HOST": f"{{{{secrets/{scope}/DATABRICKS_HOST}}}}",
        "DATABRICKS_TOKEN": f"{{{{secrets/{scope}/DATABRICKS_TOKEN}}}}",
        "DATABRICKS_MODEL": f"{{{{secrets/{scope}/DATABRICKS_MODEL}}}}",
        # Not secrets — the retriever needs these to reach the Vector Search index.
        "VECTOR_SEARCH_ENDPOINT": os.environ["VECTOR_SEARCH_ENDPOINT"],
        "VECTOR_SEARCH_INDEX": os.environ["VECTOR_SEARCH_INDEX"],
        "EMBEDDINGS_ENDPOINT": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
    }

    served = ServedEntityInput(
        entity_name=uc_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars=environment_vars,
    )

    # First-time container builds can exceed the SDK's 20-minute default wait.
    wait = timedelta(minutes=45)

    existing = {e.name for e in w.serving_endpoints.list()}
    if endpoint in existing:
        print(f"[deploy] updating endpoint '{endpoint}' -> version {version} (waiting for READY)…")
        w.serving_endpoints.update_config_and_wait(
            name=endpoint, served_entities=[served], timeout=wait
        )
    else:
        print(f"[deploy] creating endpoint '{endpoint}' -> version {version} (waiting for READY)…")
        w.serving_endpoints.create_and_wait(
            name=endpoint,
            config=EndpointCoreConfigInput(name=endpoint, served_entities=[served]),
            timeout=wait,
        )

    url = f"{host}/serving-endpoints/{endpoint}/invocations"
    print(f"[deploy] endpoint READY: {url}")
    return url


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy the Document Analyst.")
    parser.add_argument(
        "--version",
        help="Serve this existing UC model version instead of logging a new one.",
    )
    parser.add_argument(
        "--skip-secrets",
        action="store_true",
        help="Skip creating/updating the secret scope (assume it already exists).",
    )
    args = parser.parse_args()

    if args.version:
        model_name, model_version = _uc_model_name(), args.version
        print(f"[deploy] serving existing version {model_version} (skipping log/register)")
    else:
        model_name, model_version = log_and_register()

    if not args.skip_secrets:
        ensure_secrets()

    endpoint_url = create_or_update_endpoint(model_name, model_version)
    print(f"[deploy] done. Invocations URL: {endpoint_url}")
