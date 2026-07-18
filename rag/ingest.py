"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Pipeline (mirrors PA2 Part 1):

    annual_report.pdf  (UC Volume)
        --ai_parse_document-->  parsed text + page metadata
        --ai_prep_search----->  Delta table of chunks (CDF enabled)
        --Delta Sync index--->  Vector Search index (managed embeddings)

`build_chunks_table()` MUST run inside a Databricks notebook — it needs Spark and the
`ai_parse_document` / `ai_prep_search` SQL functions, which do not exist on your laptop.
`create_index()` / `wait_until_ready()` / `similarity_test()` use the Vector Search REST
client and can run either in the notebook or locally (they only need
DATABRICKS_HOST/DATABRICKS_TOKEN).

Chunk table schema (required by the spec):
    chunk_id STRING           -- primary key for the Delta Sync index
    chunk_to_retrieve STRING  -- text returned to the RAG agent (and embedded)
    chunk_to_embed STRING     -- text sent to the embedding model
    source STRING             -- e.g. 'annual_report.pdf'  (citation)
    page INT                  -- page number                (citation)

>>> IMPORTANT — reconcile the two ai_* SQL calls with your PA2 Part 1 notebook. <<<
The exact argument names for `ai_parse_document` / `ai_prep_search` are workspace/
course specific. The SQL below follows the PA4 spec; if your PA2 notebook used slightly
different arguments (e.g. chunk size / column names), match those — you already ran that
pipeline successfully, so it is the authoritative example.
"""

from __future__ import annotations

import time

from config import get_settings

# ── Tuning knobs (adjust to match PA2 if needed) ────────────────────────────
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 150     # overlap between adjacent chunks


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 + 2 — parse the PDF and chunk it into a Delta table  (Databricks only)
# ════════════════════════════════════════════════════════════════════════════
def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse `volume_path` (a PDF in a UC Volume) and write chunks to `chunks_table`.

    Args:
        spark:        the active SparkSession (provided in a Databricks notebook).
        volume_path:  e.g. '/Volumes/27100082_pa4/default/pa4/annual_report.pdf'.
        chunks_table: fully-qualified Delta table, e.g.
                      '27100082_pa4.default.27100082_analyst_chunks'.
    """
    source_name = volume_path.rsplit("/", 1)[-1]  # 'annual_report.pdf'

    # --- Step 1: parse the PDF into page-level text with ai_parse_document -----
    # Read the file as binary, then parse. ai_parse_document returns structured
    # output whose `document.pages` is an array of {page number, text content}.
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW _pa4_parsed AS
        SELECT
            '{source_name}' AS source,
            ai_parse_document(content) AS parsed
        FROM READ_FILES('{volume_path}', format => 'binaryFile')
    """)

    # Explode the parsed pages so each row is one page of text.
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW _pa4_pages AS
        SELECT
            source,
            page.pageNumber              AS page,
            page.content                 AS page_text
        FROM _pa4_parsed
        LATERAL VIEW explode(parsed:document.pages) t AS page
    """)

    # --- Step 2: chunk each page with ai_prep_search --------------------------
    # ai_prep_search splits `page_text` into search-ready chunks. Each chunk row
    # carries its parent page/source so we can cite it later.
    spark.sql(f"""
        CREATE OR REPLACE TABLE {chunks_table}
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        AS
        SELECT
            uuid()                       AS chunk_id,
            chunk.chunk_text             AS chunk_to_retrieve,
            chunk.chunk_text             AS chunk_to_embed,
            source,
            page
        FROM _pa4_pages
        LATERAL VIEW explode(
            ai_prep_search(page_text, chunkSize => {CHUNK_SIZE}, chunkOverlap => {CHUNK_OVERLAP})
        ) c AS chunk
        WHERE chunk.chunk_text IS NOT NULL AND length(trim(chunk.chunk_text)) > 0
    """)

    n = spark.table(chunks_table).count()
    print(f"[ingest] wrote {n} chunks to {chunks_table} (CDF enabled)")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — create the Vector Search endpoint + Delta Sync index
# ════════════════════════════════════════════════════════════════════════════
def create_index() -> None:
    """Create a STANDARD Vector Search endpoint and a TRIGGERED Delta Sync index
    over the chunks table, using managed embeddings.

    Reads endpoint/index/source-table/embeddings names from config + env.
    Safe to run locally (only needs DATABRICKS_HOST/DATABRICKS_TOKEN).
    """
    import os

    from databricks.vector_search.client import VectorSearchClient

    s = get_settings()
    endpoint = s["vs_endpoint"]
    index = s["vs_index"]
    embeddings = s["embeddings"]
    source_table = os.environ.get("SOURCE_TABLE")
    if not (endpoint and index and source_table):
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT / VECTOR_SEARCH_INDEX / SOURCE_TABLE must be set in .env"
        )

    vsc = VectorSearchClient(disable_notice=True)

    # 1) Endpoint (idempotent) --------------------------------------------------
    existing = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
    if endpoint not in existing:
        print(f"[ingest] creating STANDARD endpoint '{endpoint}' …")
        vsc.create_endpoint(name=endpoint, endpoint_type="STANDARD")
        _wait_endpoint_online(vsc, endpoint)
    else:
        print(f"[ingest] endpoint '{endpoint}' already exists")

    # 2) Delta Sync index with managed embeddings (idempotent) -----------------
    try:
        vsc.get_index(endpoint_name=endpoint, index_name=index).describe()
        print(f"[ingest] index '{index}' already exists — trigger a sync to refresh")
        vsc.get_index(endpoint_name=endpoint, index_name=index).sync()
        return
    except Exception:
        pass  # index does not exist yet — create it

    print(f"[ingest] creating Delta Sync index '{index}' …")
    vsc.create_delta_sync_index(
        endpoint_name=endpoint,
        index_name=index,
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_to_retrieve",
        embedding_model_endpoint_name=embeddings,
    )
    print("[ingest] index creation submitted")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — wait for READY and run a smoke similarity query
# ════════════════════════════════════════════════════════════════════════════
def wait_until_ready(timeout_s: int = 1800, poll_s: int = 30) -> None:
    """Block until the index reports ONLINE / READY (first sync can take minutes)."""
    from databricks.vector_search.client import VectorSearchClient

    s = get_settings()
    vsc = VectorSearchClient(disable_notice=True)
    idx = vsc.get_index(endpoint_name=s["vs_endpoint"], index_name=s["vs_index"])
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = idx.describe().get("status", {})
        state = status.get("detailed_state", "")
        ready = status.get("ready", False)
        print(f"[ingest] index state={state} ready={ready}")
        if ready and "ONLINE" in state:
            print("[ingest] index is READY")
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Index not READY after {timeout_s}s")


def similarity_test(query: str = "What was Meridian's net revenue in FY2023?", k: int = 4):
    """Run one similarity query and print the hits — proves the index answers."""
    from rag.store import get_retriever

    docs = get_retriever(k=k).invoke(query)
    print(f"[ingest] query: {query!r} -> {len(docs)} hits")
    for d in docs:
        meta = d.metadata
        print(f"  - p{meta.get('page')} [{meta.get('source')}]: {d.page_content[:120]}...")
    return docs


def _wait_endpoint_online(vsc, endpoint: str, timeout_s: int = 900, poll_s: int = 20) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = vsc.get_endpoint(endpoint).get("endpoint_status", {}).get("state", "")
        print(f"[ingest] endpoint state={state}")
        if state == "ONLINE":
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Endpoint '{endpoint}' not ONLINE after {timeout_s}s")


# ════════════════════════════════════════════════════════════════════════════
# Orchestration — call this from your Databricks notebook
# ════════════════════════════════════════════════════════════════════════════
def ingest(spark, volume_path: str) -> None:
    """End-to-end Task 0.3: parse → chunk → index → wait → smoke test.

    Example (in a Databricks notebook, `spark` is pre-defined):
        from rag.ingest import ingest
        ingest(spark, "/Volumes/27100082_pa4/default/pa4/annual_report.pdf")
    """
    import os

    source_table = os.environ["SOURCE_TABLE"]
    build_chunks_table(spark, volume_path, source_table)
    create_index()
    wait_until_ready()
    similarity_test()
