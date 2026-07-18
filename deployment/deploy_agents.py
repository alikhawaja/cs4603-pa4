"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

`agents.deploy()` requires a ChatAgent-compatible model (output = ChatCompletionResponse),
so this path logs the **ChatAgent wrapper** (`deployment/agent_chat.py`) rather than the
bare-graph model from Part 2. Everything else — `code_paths`, pinned `pip_requirements`,
the forward-slash model-path fix — is shared with `deploy.py`. The single `agents.deploy()`
call then provisions a serving endpoint AND a Review App with auth handled automatically.

Run from the Python 3.12 deploy env:

    uv sync --extra agents
    python deployment/deploy_agents.py                 # log the ChatAgent, then deploy
    python deployment/deploy_agents.py --version 1      # deploy an existing chat version
"""

from __future__ import annotations

import argparse
import os

import mlflow
from dotenv import load_dotenv

from deployment.deploy import (
    CODE_PATHS,
    PIP_REQUIREMENTS,
    ROOT,
    _force_forward_slash_model_path,
)

load_dotenv()


def _uc_chat_model_name() -> str:
    catalog = os.environ.get("UC_CATALOG", "27100082_pa4")
    schema = os.environ.get("UC_SCHEMA", "default")
    model = os.environ.get("UC_MODEL_CHAT", "document_analyst_chat")
    return f"{catalog}.{schema}.{model}"


def log_and_register_chat() -> tuple[str, str]:
    """Log the ChatAgent (pyfunc/models-from-code) and register a UC version."""
    from databricks.sdk import WorkspaceClient

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    user = WorkspaceClient().current_user.me().user_name
    mlflow.set_experiment(
        os.environ.get("MLFLOW_EXPERIMENT", f"/Users/{user}/pa4-document-analyst")
    )

    os.chdir(ROOT)
    _force_forward_slash_model_path()
    uc_name = _uc_chat_model_name()
    print(f"[agents] logging ChatAgent -> {uc_name}")
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=os.path.join("deployment", "agent_chat.py"),
            code_paths=CODE_PATHS,
            pip_requirements=PIP_REQUIREMENTS,
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
        )
    registered = mlflow.register_model(info.model_uri, uc_name)
    print(f"[agents] registered {uc_name} version {registered.version}")
    return uc_name, registered.version


def deploy_with_agents(model_name: str, version: str):
    """Deploy a registered UC model version with the databricks-agents SDK."""
    from databricks import agents

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    print(f"[agents] deploying {model_name} v{version} via agents.deploy() …")
    deployment = agents.deploy(
        model_name=model_name,
        model_version=version,
        scale_to_zero=True,
    )
    print("[agents] endpoint:  ", getattr(deployment, "endpoint_name", "<unknown>"))
    print("[agents] review app:", getattr(deployment, "review_app_url", "<unknown>"))
    return deployment


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the Document Analyst via databricks-agents.")
    parser.add_argument(
        "--version",
        help="Deploy this existing chat-model UC version instead of logging a new one.",
    )
    args = parser.parse_args()

    if args.version:
        model_name, model_version = _uc_chat_model_name(), args.version
        print(f"[agents] using existing chat version {model_version} (skipping log/register)")
    else:
        model_name, model_version = log_and_register_chat()

    deploy_with_agents(model_name, model_version)


if __name__ == "__main__":
    main()
