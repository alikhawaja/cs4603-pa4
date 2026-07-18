"""Python client SDK for the deployed Document Analyst (Part 3).

`DocumentAnalystClient` is a small, dependency-light wrapper any downstream app can use to
call the deployed Model Serving endpoint. It handles authentication, retries with
exponential backoff, timeouts, streaming (with graceful fallback), a health check, and
typed errors.

Response shape: the endpoint was logged with `mlflow.langchain.log_model`, so it returns
raw LangGraph state as a one-element batch **list** (Path A) — the answer is at
`data[0]["messages"][-1]["content"]`. The parser below also accepts an OpenAI
`ChatCompletion` (Path B) and a `{"predictions": ...}` wrapper, so the same client works if
the model is later re-logged as a ChatModel/ChatAgent.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx


class AnalystClientError(Exception):
    """Raised for non-timeout HTTP/protocol failures from the endpoint."""

    def __init__(self, message: str, status_code: int | None = None, request_id: str | None = None):
        detail = message
        if status_code is not None:
            detail = f"[HTTP {status_code}] {message}"
        if request_id:
            detail = f"{detail} (request_id={request_id})"
        super().__init__(detail)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    """Client for the deployed Document Analyst serving endpoint."""

    # HTTP statuses worth retrying: 429 rate-limited, 503 endpoint scaling/starting.
    _RETRY_STATUSES = frozenset({429, 503})

    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        if not self.host or not self.token:
            raise AnalystClientError(
                "Missing credentials: pass host/token or set DATABRICKS_HOST / DATABRICKS_TOKEN."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self._invocations_url = f"{self.host}/serving-endpoints/{endpoint_name}/invocations"
        self._endpoint_url = f"{self.host}/api/2.0/serving-endpoints/{endpoint_name}"
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ── Public API ──────────────────────────────────────────────────────────
    def ask(self, question: str) -> str:
        """Send a question and return the analyst's answer string."""
        resp = self._post_with_retry({"messages": [{"role": "user", "content": question}]})
        return self._parse_answer(resp.json())

    def ask_streaming(self, question: str) -> Iterator[str]:
        """Yield answer text as it arrives.

        Attempts a streaming (SSE) request. A models-from-code LangChain endpoint usually
        does NOT support streaming — it either rejects `stream: true` (HTTP 400) or returns
        a single completion — so this method degrades gracefully: on anything other than a
        real SSE stream, it falls back to a normal `ask()` and yields the full answer once.
        That fallback (rather than raising) is the intended behaviour for a single-chunk
        endpoint; genuine errors still surface because `ask()` re-issues the request.
        """
        payload = {"messages": [{"role": "user", "content": question}], "stream": True}
        try:
            with httpx.stream(
                "POST", self._invocations_url, headers=self._headers, json=payload,
                timeout=self.timeout,
            ) as resp:
                if resp.status_code == 200 and "text/event-stream" in resp.headers.get(
                    "content-type", ""
                ):
                    emitted = False
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data in ("", "[DONE]"):
                            continue
                        chunk = self._extract_delta(data)
                        if chunk:
                            emitted = True
                            yield chunk
                    if emitted:
                        return
                # Not an SSE stream (non-200 like "streaming not supported", or a single
                # JSON completion) -> fall through to the non-streaming path below.
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Streaming request to '{self.endpoint_name}' timed out (timeout={self.timeout}s)."
            ) from exc

        # Fallback: yield the full answer once (also surfaces any genuine error via ask()).
        yield self.ask(question)

    def health_check(self) -> bool:
        """Return True only if the endpoint exists and is in the READY state."""
        try:
            resp = httpx.get(self._endpoint_url, headers=self._headers, timeout=self.timeout)
        except httpx.TimeoutException:
            return False
        if resp.status_code != 200:
            return False
        state = resp.json().get("state", {}) or {}
        return state.get("ready") == "READY"

    # ── Internals ───────────────────────────────────────────────────────────
    def _post_with_retry(self, payload: dict) -> httpx.Response:
        """POST to /invocations with exponential backoff on 429/503; raise on failure."""
        start = time.monotonic()
        last_error: AnalystClientError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = httpx.post(
                    self._invocations_url, headers=self._headers, json=payload,
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to '{self.endpoint_name}' timed out after {elapsed:.3f}s "
                    f"(timeout={self.timeout}s)."
                ) from exc
            except httpx.HTTPError as exc:
                # Network/transport error — treat as retryable.
                last_error = AnalystClientError(f"Connection error: {exc}")
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code not in self._RETRY_STATUSES:
                    raise self._error_from_response(resp)
                last_error = self._error_from_response(resp)

            if attempt < self.max_retries:
                # Exponential backoff: 1s, 2s, 4s, ...
                time.sleep(2 ** attempt)

        assert last_error is not None
        raise last_error

    def _parse_answer(self, data) -> str:
        """Extract the answer string from any of the accepted response shapes."""
        # Path A — raw LangGraph state as a one-element batch list.
        if isinstance(data, list) and data:
            return self._last_message_content(data[0])
        if isinstance(data, dict):
            # {"predictions": [...]} wrapper around the state.
            if "predictions" in data:
                preds = data["predictions"]
                if isinstance(preds, list) and preds:
                    return self._last_message_content(preds[0])
                return self._last_message_content(preds)
            # Path B — OpenAI ChatCompletion.
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            # A bare state dict.
            if "messages" in data:
                return self._last_message_content(data)
        raise AnalystClientError(f"Unexpected response shape: {type(data).__name__}")

    @staticmethod
    def _last_message_content(state) -> str:
        msgs = state.get("messages") if isinstance(state, dict) else None
        if not msgs:
            raise AnalystClientError("Response contained no messages.")
        last = msgs[-1]
        return last["content"] if isinstance(last, dict) else getattr(last, "content", str(last))

    @staticmethod
    def _extract_delta(data: str) -> str:
        """Pull incremental text out of one SSE `data:` payload, if present."""
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return data  # already plain text
        if isinstance(obj, dict) and "choices" in obj:
            delta = obj["choices"][0].get("delta") or obj["choices"][0].get("message") or {}
            return delta.get("content", "") or ""
        return ""

    def _error_from_response(self, resp: httpx.Response) -> AnalystClientError:
        request_id = (
            resp.headers.get("x-request-id")
            or resp.headers.get("x-databricks-org-id")
            or resp.headers.get("x-databricks-request-id")
        )
        try:
            message = resp.json().get("message") or resp.text[:300]
        except (json.JSONDecodeError, ValueError):
            message = resp.text[:300] or resp.reason_phrase
        return AnalystClientError(message, status_code=resp.status_code, request_id=request_id)
