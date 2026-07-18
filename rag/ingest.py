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
    SELECT *
    FROM ai_prep_search(
        (SELECT ai_parse_document(content) FROM READ_FILES('{volume_path}', format => 'binaryFile'))
    )
    """)

    spark.sql(f"""
    ALTER TABLE {chunks_table}
    SET TBLPROPERTIES (
        delta.enableChangeDataFeed = true
    )
    """)

    print(f"Created {chunks_table}")


def create_index() -> None:
    raise NotImplementedError("Task 0.3: create the Vector Search Delta Sync index")
