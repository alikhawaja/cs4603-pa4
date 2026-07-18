"""Vector Search retriever factory (Task 1.4 support).

Returns a LangChain retriever over the managed Databricks Vector Search index
built by `rag/ingest.py`. Because the index is a managed service reachable
with DATABRICKS_HOST/DATABRICKS_TOKEN, this exact code path runs both locally
and inside the deployed serving container — no separate embedding path.
"""

from __future__ import annotations

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


def get_vector_store():
    from databricks_langchain import DatabricksVectorSearch

    settings = get_settings()
    if not settings["vs_endpoint"] or not settings["vs_index"]:
        raise OSError(
            "Missing required environment variables: VECTOR_SEARCH_ENDPOINT and/or "
            "VECTOR_SEARCH_INDEX. The retriever cannot reach the Vector Search index "
            "without them — set both in .env (local) or endpoint environment_vars (deployed)."
        )
    # Delta Sync index with managed embeddings: the service embeds the query
    # server-side, so no embedding model is configured here.
    return DatabricksVectorSearch(
        endpoint=settings["vs_endpoint"],
        index_name=settings["vs_index"],
        columns=CITATION_COLUMNS,
    )


def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})
