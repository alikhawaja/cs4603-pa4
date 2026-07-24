"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""

from __future__ import annotations

from collections.abc import Iterator
import os


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.host = host if host else os.environ["DATABRICKS_HOST"]
        self.token = token if token else os.environ["DATABRICKS_TOKEN"] 
        self.timeout = timeout
        self.max_retries = max_retries

    def ask(self, question: str) -> str:
        import time
        import requests

        url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"messages": [{
            "role": "user",
            "content": question
        }]}

        last_exc = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.exceptions.Timeout:
                elapsed = time.monotonic() - start
                raise TimeoutError(f"Request timed out after {elapsed:.1f}s (limit: {self.timeout}s)")

            if response.status_code == 200:
                data = response.json()
                return data[0]["messages"][-1]["content"]

            if response.status_code in (429, 503) and attempt < self.max_retries:
                wait = 2 ** attempt 
                time.sleep(wait)
                last_exc = AnalystClientError(
                    f"Retryable error {response.status_code}, retrying after {wait}s",
                    status_code=response.status_code,
                    request_id=response.headers.get("x-request-id"),
                )
                continue

            raise AnalystClientError(
                f"Request failed: {response.status_code} {response.text}",
                status_code=response.status_code,
                request_id=response.headers.get("x-request-id"),
            )

        raise last_exc

    def ask_streaming(self, question: str) -> Iterator[str]:
        import json
        import requests

        url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"messages": [{"role": "user", "content": question}]}

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout, stream=True
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.timeout}s")

        if response.status_code != 200:
            raise AnalystClientError(
                f"Request failed: {response.status_code} {response.text}",
                status_code=response.status_code,
                request_id=response.headers.get("x-request-id"),
            )

        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            got_any_chunk = False
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        got_any_chunk = True
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            if not got_any_chunk:
                yield self.ask(question)
        else:
            data = response.json()
            yield data[0]["messages"][-1]["content"]


    def health_check(self) -> bool:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import EndpointStateReady

        w = WorkspaceClient()

        return (
        w.serving_endpoints.get(self.endpoint_name).state.ready
        == EndpointStateReady.READY
    )
