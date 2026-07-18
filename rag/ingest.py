"""Corpus ingestion into Databricks Vector Search (Task 0.3).

Mirrors the PA2 Part 1 pipeline:
    volume PDF -> ai_parse_document -> parse table (VARIANT)
               -> ai_prep_search    -> chunks table (Delta, Change Data Feed on)
               -> Delta Sync Vector Search index (TRIGGERED, managed embeddings)

The SQL runs on a SQL warehouse via the Statement Execution API, so this
script works from a laptop with no local Spark:

    uv run python rag/ingest.py

Inside a Databricks notebook you may pass a `spark` session to
`build_chunks_table` and the same SQL runs through it instead.
"""

from __future__ import annotations

import os
import time

from config import get_settings

PAGE_EXPR = "try_cast(chunk.value:pages[0].page_id AS INT) + 1"


def _default_names() -> dict[str, str]:
    catalog = os.environ.get("UC_CATALOG", "cs4603")
    schema = os.environ.get("UC_SCHEMA", "default")
    chunks_table = os.environ.get("SOURCE_TABLE") or f"{catalog}.{schema}.analyst_chunks"
    return {
        "volume_dir": f"/Volumes/{catalog}/{schema}/pa4/",
        "parse_table": chunks_table.replace("_chunks", "") + "_parsed",
        "chunks_table": chunks_table,
    }


def _make_sql_runner(spark=None):
    """Return run_sql(statement): Spark session if given, else SQL warehouse."""
    if spark is not None:
        return lambda statement: spark.sql(statement)

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState

    w = WorkspaceClient()
    warehouse_id = os.environ.get("SQL_WAREHOUSE_ID")
    if not warehouse_id:
        warehouses = list(w.warehouses.list())
        if not warehouses:
            raise RuntimeError("No SQL warehouse available; set SQL_WAREHOUSE_ID")
        warehouse_id = warehouses[0].id

    def run_sql(statement: str, timeout_s: int = 600):
        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id, statement=statement, wait_timeout="50s"
        )
        start = time.time()
        while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
            if time.time() - start > timeout_s:
                raise TimeoutError(f"SQL still running after {timeout_s}s")
            time.sleep(5)
            resp = w.statement_execution.get_statement(resp.statement_id)
        if resp.status.state != StatementState.SUCCEEDED:
            raise RuntimeError(f"SQL failed: {resp.status.error}")
        return resp

    return run_sql


def build_chunks_table(
    spark=None,
    volume_path: str | None = None,
    chunks_table: str | None = None,
) -> None:
    """Parse the corpus PDF and chunk it into a Delta table with CDF enabled."""
    names = _default_names()
    volume_path = volume_path or names["volume_dir"]
    chunks_table = chunks_table or names["chunks_table"]
    parse_table = names["parse_table"]
    run_sql = _make_sql_runner(spark)

    run_sql(
        f"""CREATE TABLE IF NOT EXISTS {parse_table} (
            path STRING,
            parsed VARIANT
        ) TBLPROPERTIES (delta.enableChangeDataFeed = true)"""
    )
    run_sql(
        f"""INSERT OVERWRITE {parse_table}
        SELECT path, ai_parse_document(content) AS parsed
        FROM READ_FILES('{volume_path}', format => 'binaryFile')"""
    )

    run_sql(
        f"""CREATE TABLE IF NOT EXISTS {chunks_table} (
            chunk_id STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed STRING,
            source STRING,
            page INT
        ) TBLPROPERTIES (delta.enableChangeDataFeed = true)"""
    )
    run_sql(
        f"""INSERT OVERWRITE {chunks_table}
        SELECT
            chunk.value:chunk_id::STRING AS chunk_id,
            chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
            chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
            regexp_extract(path, '[^/]+$', 0) AS source,
            {PAGE_EXPR} AS page
        FROM (
            SELECT path, ai_prep_search(parsed) AS result FROM {parse_table}
        ) prepped,
            LATERAL variant_explode(result:document.contents) AS chunk"""
    )

    count = _make_sql_runner(spark)(f"SELECT count(*) FROM {chunks_table}")
    if spark is None:
        n = count.result.data_array[0][0]
    else:
        n = count.collect()[0][0]
    print(f"Chunks table {chunks_table}: {n} chunks")


def _index_ready(index) -> bool:
    status = index.describe().get("status", {})
    # `ready` flips true before the index is queryable; require ONLINE too.
    return bool(status.get("ready", False)) and str(
        status.get("detailed_state", "")
    ).startswith("ONLINE")


def create_index(chunks_table: str | None = None, wait_timeout_s: int = 1800) -> None:
    """Create the STANDARD VS endpoint (if missing) and the TRIGGERED Delta
    Sync index with managed embeddings, then wait until it is READY."""
    from databricks.vector_search.client import VectorSearchClient

    settings = get_settings()
    endpoint = settings["vs_endpoint"]
    index_name = settings["vs_index"]
    chunks_table = chunks_table or _default_names()["chunks_table"]

    client = VectorSearchClient(
        workspace_url=settings["host"],
        personal_access_token=settings["token"],
        disable_notice=True,
    )

    endpoints = [e["name"] for e in client.list_endpoints().get("endpoints", [])]
    if endpoint not in endpoints:
        print(f"Creating STANDARD Vector Search endpoint {endpoint} ...")
        client.create_endpoint(name=endpoint, endpoint_type="STANDARD")
    else:
        print(f"Vector Search endpoint {endpoint} already exists")

    existing = [
        i["name"] for i in client.list_indexes(name=endpoint).get("vector_indexes", [])
    ]
    if index_name not in existing:
        print(f"Creating Delta Sync index {index_name} ...")
        client.create_delta_sync_index(
            endpoint_name=endpoint,
            index_name=index_name,
            source_table_name=chunks_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=settings["embeddings"],
        )
    else:
        print(f"Index {index_name} already exists — triggering a sync")

    index = client.get_index(endpoint_name=endpoint, index_name=index_name)
    if index_name in existing:
        # Re-ingestion into an existing TRIGGERED index: pick up the new rows.
        index.sync()
    start = time.time()
    while not _index_ready(index):
        if time.time() - start > wait_timeout_s:
            raise TimeoutError(f"Index not READY after {wait_timeout_s}s")
        print("  waiting for index to be READY ...")
        time.sleep(30)
    print(f"Index {index_name} is READY")


def verify(query: str = "What was the net revenue in fiscal year 2023?") -> None:
    """Similarity-search smoke test straight against the index."""
    from databricks.vector_search.client import VectorSearchClient

    settings = get_settings()
    client = VectorSearchClient(
        workspace_url=settings["host"],
        personal_access_token=settings["token"],
        disable_notice=True,
    )
    index = client.get_index(
        endpoint_name=settings["vs_endpoint"], index_name=settings["vs_index"]
    )
    result = index.similarity_search(
        query_text=query, columns=["chunk_id", "source", "page"], num_results=3
    )
    rows = result.get("result", {}).get("data_array", [])
    if not rows:
        raise RuntimeError("Similarity search returned no rows — index not ready?")
    print(f"Similarity search for {query!r} returned {len(rows)} rows:")
    for row in rows:
        print(f"  chunk={row[0]} source={row[1]} page={row[2]} score={row[-1]}")


if __name__ == "__main__":
    build_chunks_table()
    create_index()
    verify()
