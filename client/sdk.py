"""Python client SDK for the deployed Document Analyst (Part 3, Task 3.1).

Thin wrapper around the OpenAI-compatible `/serving-endpoints/<name>/invocations`
route that:
  - reads DATABRICKS_HOST / DATABRICKS_TOKEN from the environment when not
    passed explicitly,
  - retries 429 (rate limited) / 503 (endpoint scaling up) responses with
    exponential backoff,
  - raises a plain TimeoutError (with elapsed time) when the overall budget
    is exceeded,
  - wraps any other HTTP error in AnalystClientError(status_code, message,
    request_id),
  - supports both a single-shot ask() and a streaming ask_streaming() that
    falls back to yielding one full chunk if the endpoint doesn't emit
    incremental deltas.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import httpx


class AnalystClientError(Exception):
    """Raised for any non-retryable (or retries-exhausted) HTTP error."""

    def __init__(self, message: str, status_code: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:
        base = super().__str__()
        extras = []
        if self.status_code is not None:
            extras.append(f"status_code={self.status_code}")
        if self.request_id is not None:
            extras.append(f"request_id={self.request_id}")
        return f"{base} ({', '.join(extras)})" if extras else base


class DocumentAnalystClient:
    """Client for a deployed Document Analyst Model Serving endpoint."""

    _RETRYABLE_STATUS = (429, 503)

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
        self.token = token or os.environ.get("DATABRICKS_TOKEN")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.host:
            raise OSError("DATABRICKS_HOST not provided and not set in the environment")
        if not self.token:
            raise OSError("DATABRICKS_TOKEN not provided and not set in the environment")

        self._invocations_url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        self._auth_headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _backoff_seconds(self, attempt: int) -> float:
        return min(2**attempt, 30)

    def _raise_for_status(self, resp: httpx.Response) -> None:
        request_id = resp.headers.get("x-request-id")
        try:
            if hasattr(resp, "read"):
                resp.read()
            message = resp.json().get("message", resp.text)
        except Exception:
            try:
                message = resp.text
            except Exception:
                message = "Unexpected streaming error"
        raise AnalystClientError(message, status_code=resp.status_code, request_id=request_id)

    @staticmethod
    def _extract_answer(data: object) -> str:
        if isinstance(data, list):
            if not data:
                return ""
            data = data[0]

        if not isinstance(data, dict):
            return ""

        choices = data.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content")
            if content:
                return content

        # Fallback: some models-from-code endpoints echo the raw messages-out
        # contract instead of the OpenAI `choices[]` shape.
        messages = data.get("messages")
        if messages:
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    content = last.get("content")
                    if content:
                        return content
        final_answer = data.get("final_answer")
        if isinstance(final_answer, str) and final_answer:
            return final_answer
        return ""

    def _post(self, payload: dict, *, stream: bool, client: httpx.Client) -> httpx.Response:
        """POST with exponential backoff on 429/503. Raises TimeoutError / AnalystClientError."""
        start = time.monotonic()
        attempt = 0
        while True:
            elapsed = time.monotonic() - start
            remaining = self.timeout - elapsed
            if remaining <= 0:
                raise TimeoutError(
                    f"Request to '{self.endpoint_name}' timed out after {elapsed:.3f}s"
                )
            try:
                if stream:
                    request = client.build_request(
                        "POST", self._invocations_url, json=payload, timeout=remaining
                    )
                    resp = client.send(request, stream=True)
                else:
                    resp = client.post(self._invocations_url, json=payload, timeout=remaining)
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to '{self.endpoint_name}' timed out after {elapsed:.3f}s"
                ) from exc

            if resp.status_code in self._RETRYABLE_STATUS and attempt < self.max_retries:
                if stream:
                    resp.close()
                time.sleep(self._backoff_seconds(attempt))
                attempt += 1
                continue

            if resp.status_code >= 400:
                self._raise_for_status(resp)

            return resp

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> str:
        payload = {"messages": [{"role": "user", "content": question}]}
        with httpx.Client(headers=self._auth_headers) as client:
            resp = self._post(payload, stream=False, client=client)
            return self._extract_answer(resp.json())

    def ask_streaming(self, question: str) -> Iterator[str]:
        payload = {
            "messages": [{"role": "user", "content": question}],
            "stream": True,
        }
        try:
            with httpx.Client(headers=self._auth_headers) as client:
                resp = self._post(payload, stream=True, client=client)
                yielded_any = False
                try:
                    lines = resp.iter_lines()
                    for line in lines:
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or choices[0].get("message") or {}
                        content = delta.get("content")
                        if content:
                            yielded_any = True
                            yield content
                finally:
                    resp.close()

            if not yielded_any:
                # The endpoint may not emit OpenAI-style incremental deltas at all —
                # fall back to a single, non-incremental full answer.
                answer = self.ask(question)
                if answer:
                    yield answer
        except AnalystClientError as exc:
            if "does not support streaming" in str(exc).lower():
                answer = self.ask(question)
                if answer:
                    yield answer
            else:
                raise

    def health_check(self) -> bool:
        """True only when the serving endpoint reports state READY."""
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient(host=self.host, token=self.token)
            status = w.serving_endpoints.get(self.endpoint_name)
            state = status.state.ready if status.state else None
            return str(state) == "EndpointStateReady.READY"
        except Exception:
            return False
