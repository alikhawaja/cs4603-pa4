"""Bonus B — ChatAgent wrapper for the Document Analyst (v2 style).

`agents.deploy()` refuses the raw-LangGraph-state schema of Part 2's model
("output schema must be ChatCompletionResponse or StringResponse"), so this
models-from-code definition wraps the SAME compiled graph in an
`mlflow.pyfunc.ChatAgent`: messages in, a single assistant message out. This
is the interface `databricks_deployment_v2/agent_chat.py` uses, and it plugs
into the AI Playground / Review App / evaluation out of the box.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import mlflow
from dotenv import load_dotenv

load_dotenv()  # local runs read .env; the container has real env vars instead

_REQUIRED_ENV = [
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_MODEL",
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

from mlflow.pyfunc import ChatAgent  # noqa: E402
from mlflow.types.agent import (  # noqa: E402
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from agent.graph import build_graph  # noqa: E402


class DocumentAnalystChatAgent(ChatAgent):
    """Chat-native facade over the Part 1 multi-agent graph."""

    def __init__(self):
        self.graph = build_graph()

    def _invoke(self, messages: list[ChatAgentMessage]) -> str:
        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in messages]
        }
        result = self.graph.invoke(payload)
        return result["messages"][-1].content

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: ChatContext | None = None,
        custom_inputs: dict[str, Any] | None = None,
    ) -> ChatAgentResponse:
        answer = self._invoke(messages)
        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant", content=answer, id=str(uuid.uuid4())
                )
            ]
        )

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: ChatContext | None = None,
        custom_inputs: dict[str, Any] | None = None,
    ):
        # The graph produces one final answer; emit it as a single chunk.
        answer = self._invoke(messages)
        yield ChatAgentChunk(
            delta=ChatAgentMessage(
                role="assistant", content=answer, id=str(uuid.uuid4())
            )
        )


AGENT = DocumentAnalystChatAgent()
mlflow.models.set_model(AGENT)
