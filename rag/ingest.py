"""Corpus ingestion into Databricks Vector Search (Task 0.3)."""

from __future__ import annotations

import os

from config import get_settings

def build_chunks_table(spark, volume_path, chunks_table):
    parse_table = f"{chunks_table}_parsed"

    spark.sql(f"""
        CREATE OR REPLACE TABLE {parse_table} (
            path STRING,
            parsed VARIANT
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    spark.sql(f"""
        INSERT INTO {parse_table}
        SELECT
            path,
            ai_parse_document(content) AS parsed
        FROM READ_FILES('{volume_path}/', format => 'binaryFile')
    """)

    n_parsed = spark.sql(f"SELECT count(*) AS n FROM {parse_table}").collect()[0]["n"]
    print(f"Parsed {n_parsed} document(s) into {parse_table}")

    spark.sql(f"""
        CREATE OR REPLACE TABLE {chunks_table} (
            chunk_id STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed STRING,
            source STRING,
            page INT
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    spark.sql(f"""
        WITH prepped AS (
            SELECT
                path,
                ai_prep_search(parsed) AS result
            FROM {parse_table}
        )
        INSERT INTO {chunks_table}
        SELECT
            chunk.value:chunk_id::STRING AS chunk_id,
            chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
            chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
            prepped.path AS source,
            CAST(chunk.value:metadata.page_number AS INT) AS page
        FROM prepped,
             LATERAL variant_explode(prepped.result:document.contents) AS chunk
    """)

    n_chunks = spark.sql(f"SELECT count(*) AS n FROM {chunks_table}").collect()[0]["n"]
    print(f"Chunked into {n_chunks} rows in {chunks_table}")

    spark.sql(f"""
        SELECT source, count(*) AS n_chunks
        FROM {chunks_table}
        GROUP BY source
    """).show(truncate=False)


def create_index() -> None:
    """Create a STANDARD Vector Search endpoint and a TRIGGERED Delta Sync index."""
    from databricks.vector_search.client import VectorSearchClient

    s = get_settings()
    source_table = os.environ["SOURCE_TABLE"]
    vs_endpoint = s["vs_endpoint"]
    vs_index = s["vs_index"]

    client = VectorSearchClient()

    existing_endpoints = {e["name"] for e in client.list_endpoints().get("endpoints", [])}
    if vs_endpoint not in existing_endpoints:
        print(f"Creating STANDARD Vector Search endpoint '{vs_endpoint}'...")
        client.create_endpoint(name=vs_endpoint, endpoint_type="STANDARD")
    else:
        print(f"Endpoint '{vs_endpoint}' already exists, reusing it.")

    existing_indexes = {
        idx["name"] for idx in client.list_indexes(vs_endpoint).get("vector_indexes", [])
    }
    index_existed = vs_index in existing_indexes
    
    if not index_existed:
        print(f"Creating Delta Sync index '{vs_index}' over '{source_table}'...")
        client.create_delta_sync_index(
            endpoint_name=vs_endpoint,
            index_name=vs_index,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=s["embeddings"],
        )
    else:
        print(f"Index '{vs_index}' already exists — will sync after ready.")

    index = client.get_index(vs_endpoint, vs_index)
    import datetime
    index.wait_until_ready(timeout=datetime.timedelta(seconds=1800))
    
    # Only sync if index already existed (not newly created)
    if index_existed:
        index.sync()
    status = index.describe()
    print(f"Index status: {status.get('status', {}).get('detailed_state', 'UNKNOWN')}")

    results = index.similarity_search(
        query_text="What was Meridian's net revenue in fiscal year 2023?",
        columns=["chunk_id", "chunk_to_retrieve", "source", "page"],
        num_results=3,
    )
    print("Smoke-test similarity search results:")
    for row in results.get("result", {}).get("data_array", []):
        print(row)
