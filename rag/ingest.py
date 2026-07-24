"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    spark.sql(f"DROP TABLE IF EXISTS {chunks_table}")

    spark.sql(f"""
    CREATE TABLE {chunks_table} AS
    SELECT
        chunk.value:chunk_id::STRING          AS chunk_id,
        chunk.value:chunk_position::INT        AS chunk_position,
        chunk.value:chunk_to_retrieve::STRING  AS chunk_to_retrieve,
        chunk.value:chunk_to_embed::STRING     AS chunk_to_embed,
        chunk.value:pages[0]:page_id::INT      AS page,
        '{volume_path.split('/')[-1]}'         AS source
    FROM (
        SELECT ai_prep_search(ai_parse_document(content)) AS result
        FROM READ_FILES('{volume_path}', format => 'binaryFile')
    ) AS prepped,
    LATERAL variant_explode(prepped.result:document.contents) AS chunk
    """)

    spark.sql(f"""
    ALTER TABLE {chunks_table}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    print(f"Created {chunks_table}")


def create_index() -> None:
    from databricks.vector_search.client import VectorSearchClient
    from config import get_settings

    s = get_settings()
    client = VectorSearchClient(workspace_url=s["host"], personal_access_token=s["token"])

    client.create_endpoint(name=s["vs_endpoint"], endpoint_type="STANDARD")

    client.create_delta_sync_index(
        endpoint_name=s["vs_endpoint"],
        index_name=s["vs_index"],
        source_table_name="main.default.analyst_chunks",
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_to_retrieve",
        embedding_model_endpoint_name=s["embeddings"],
    )
    print(f"Index {s['vs_index']} creation triggered.")
