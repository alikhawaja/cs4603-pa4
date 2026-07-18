"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

`get_retriever()` returns a LangChain retriever over the Databricks Vector Search
index built by `ingest.py`, using `DatabricksVectorSearch` from `databricks_langchain`.
Endpoint/index names come from `config.get_settings()`.

This exact retriever is reused verbatim by the deployed model (Part 2): because the
index is a managed Databricks service reachable with DATABRICKS_HOST/DATABRICKS_TOKEN,
the same code path serves both local testing and the serving container — no separate
embedding path for deployment.
"""

from __future__ import annotations

from functools import lru_cache

from config import get_settings

# Column produced by ingest.py that holds the human-readable chunk text.
TEXT_COLUMN = "chunk_to_retrieve"
# Extra columns returned alongside each hit so the RAG agent can cite sources.
CITATION_COLUMNS = ["chunk_id", "source", "page"]


@lru_cache(maxsize=1)
def get_vector_store():
    """Return a DatabricksVectorSearch handle over the Task 0.3 index.

    The index is a Delta Sync index with *managed* embeddings, so we do NOT pass an
    embedding function — Databricks embeds the query for us using the same
    `embedding_model_endpoint_name` the index was created with.
    """
    from databricks_langchain import DatabricksVectorSearch

    s = get_settings()
    if not s["vs_endpoint"] or not s["vs_index"]:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT / VECTOR_SEARCH_INDEX are not set. "
            "Run Task 0.3 ingestion and set them in your .env (or endpoint env vars)."
        )
    # Managed-embeddings Delta Sync index: Databricks already knows the source
    # text column, so we must NOT pass `text_column` (it raises if we do).
    return DatabricksVectorSearch(
        endpoint=s["vs_endpoint"],
        index_name=s["vs_index"],
        columns=CITATION_COLUMNS + [TEXT_COLUMN],
    )


def get_retriever(k: int = 4):
    """Return a top-k retriever over the Vector Search index."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})
