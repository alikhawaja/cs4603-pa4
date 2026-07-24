"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

TODO: Implement `get_retriever(k=4)` that returns a LangChain retriever over the
Databricks Vector Search index built by `ingest.py`, using
`DatabricksVectorSearch` from `databricks_langchain`. Read endpoint/index names
from config.get_settings(). This exact retriever is reused by the deployed model.
"""

from __future__ import annotations
from databricks.sdk import WorkspaceClient
from config import get_settings
import os 
os.environ["DATABRICKS_CONFIG_PROFILE"] = ""

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]



def get_vector_store():
    from databricks_langchain import DatabricksVectorSearch

    s = get_settings()

    # return DatabricksVectorSearch(
    #     index_name=s["vs_index"],
    #     endpoint=s["vs_endpoint"],
    #     columns=CITATION_COLUMNS,
    #     token=os.environ["DATABRICKS_TOKEN"],
    #     workspace_url=os.environ["DATABRICKS_HOST"],
    # )

    return DatabricksVectorSearch(
        index_name=s["vs_index"],
        columns=CITATION_COLUMNS,
        workspace_client=WorkspaceClient(),
    )

def get_retriever(k: int = 4):
    store = get_vector_store()

    return store.as_retriever(search_kwargs={"k":k})