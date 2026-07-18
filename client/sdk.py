"""Python client SDK for the deployed Document Analyst (Part 3).

Any downstream application can do:

    from client.sdk import DocumentAnalystClient
    client = DocumentAnalystClient("zakariya-document-analyst")
    if client.health_check():
        print(client.ask("What was the net income in 2023?"))

Design notes
------------
- Auth falls back to DATABRICKS_HOST / DATABRICKS_TOKEN from the environment
  (a .env is honoured for parity with the rest of the project).
- 429 (rate limit) and 503 (endpoint scaling / temporarily unavailable) are
  retried with exponential backoff, honouring a Retry-After header if present.
- Timeouts raise the builtin TimeoutError with the elapsed time; every other
  HTTP failure is wrapped in AnalystClientError (status code + request id).
- The endpoint was logged with mlflow.langchain.log_model (Path A), so it
  returns RAW LangGraph state as a one-element batch list; answers are read
  from data[0]["messages"][-1]["content"].
- ask_streaming() parses SSE `data: ...` lines when the server streams, but a
  models-from-code endpoint may legitimately answer with one JSON body — in
  that case the full answer is yielded as a single chunk (see Task 3.1 caveat).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 503}


class AnalystClientError(Exception):
    """An HTTP failure from the serving endpoint, with debugging context."""

    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:  # keep the context visible in notebooks/logs
        base = super().__str__()
        return f"[status={self.status_code} request_id={self.request_id}] {base}"


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if host is None or token is None:
            try:  # optional: mirror the project's .env-based workflow
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
        self.endpoint_name = endpoint_name
        self.host = (host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        if not self.host or not self.token:
            raise AnalystClientError(
                "No credentials: pass host/token or set DATABRICKS_HOST and "
                "DATABRICKS_TOKEN in the environment."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.token}"},
        )

    # ── internals ───────────────────────────────────────────────────────────

    @property
    def invocations_url(self) -> str:
        return f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"

    def _backoff_delay(self, attempt: int, response: httpx.Response) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return float(retry_after)
        return float(2**attempt)  # 1s, 2s, 4s, ...

    def _request(self, payload: dict, *, stream: bool = False) -> httpx.Response:
        """POST with retry on 429/503, TimeoutError on timeout, wrapped errors."""
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                request = self._client.build_request(
                    "POST", self.invocations_url, json=payload
                )
                response = self._client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - start
                raise TimeoutError(
                    f"Request to {self.endpoint_name} timed out after "
                    f"{elapsed:.2f}s (timeout={self.timeout}s)"
                ) from exc

            if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                delay = self._backoff_delay(attempt, response)
                logger.warning(
                    "Got %s from %s — retry %d/%d in %.1fs",
                    response.status_code,
                    self.endpoint_name,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                response.close()
                time.sleep(delay)
                attempt += 1
                continue

            if response.status_code >= 400:
                request_id = response.headers.get("x-request-id")
                if stream:
                    response.read()
                try:
                    detail = response.json().get("message", response.text)
                except Exception:
                    detail = response.text
                response.close()
                raise AnalystClientError(
                    f"Endpoint '{self.endpoint_name}' returned HTTP "
                    f"{response.status_code}: {detail[:500]}",
                    status_code=response.status_code,
                    request_id=request_id,
                )
            return response

    @staticmethod
    def _extract_answer(data) -> str:
        """Parse the raw-LangGraph-state batch shape (Path A)."""
        try:
            batch = data.get("predictions", data) if isinstance(data, dict) else data
            return batch[0]["messages"][-1]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AnalystClientError(
                f"Unexpected response shape from endpoint: {str(data)[:300]}"
            ) from exc

    # ── public API ──────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Send one question and return the analyst's final answer."""
        response = self._request(
            {"messages": [{"role": "user", "content": question}]}
        )
        return self._extract_answer(response.json())

    def ask_streaming(self, question: str) -> Iterator[str]:
        """Yield answer chunks as they arrive.

        Parses SSE (`data: ...` lines) when the endpoint streams. A
        models-from-code endpoint may instead return one JSON completion — or,
        like this one, reject `stream: true` outright with HTTP 400 ("This
        endpoint does not support streaming"). Both are valid outcomes per the
        Task 3.1 caveat: fall back to a normal request and yield the full
        answer as a single chunk.
        """
        try:
            response = self._request(
                {"messages": [{"role": "user", "content": question}], "stream": True},
                stream=True,
            )
        except AnalystClientError as exc:
            if exc.status_code == 400:  # endpoint doesn't implement predict_stream
                logger.info("Endpoint rejects streaming; yielding full answer once")
                yield self.ask(question)
                return
            raise
        content_type = response.headers.get("content-type", "")
        try:
            if "text/event-stream" not in content_type:
                # Non-incremental response: yield the whole answer once.
                response.read()
                yield self._extract_answer(response.json())
                return
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if isinstance(event, dict):
                    choices = event.get("choices") or []
                    if choices:  # OpenAI-style delta chunk
                        chunk = (choices[0].get("delta") or {}).get("content", "")
                    else:  # raw-state chunk
                        messages = event.get("messages") or []
                        chunk = messages[-1].get("content", "") if messages else ""
                    if chunk:
                        yield chunk
        finally:
            response.close()

    def health_check(self) -> bool:
        """True only when the serving endpoint reports READY."""
        try:
            response = self._client.get(
                f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"
            )
        except httpx.TimeoutException:
            return False
        if response.status_code != 200:
            return False
        state = response.json().get("state", {})
        return state.get("ready") == "READY"
