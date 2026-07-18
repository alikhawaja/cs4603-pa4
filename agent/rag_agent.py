"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

Retrieves top-k chunks for the *current step's* query (not the full user
question), has the LLM extract one cited fact from them, and appends it to
`step_results`. The same retriever object (from `rag/store.py`) is used
locally and inside the deployed serving container.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState

NOT_FOUND = "not found in documents"


def format_docs(docs) -> str:
    """Number the retrieved chunks and tag each with its source citation."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source", "annual_report.pdf")
        page = meta.get("page", "?")
        if isinstance(page, float) and page.is_integer():
            page = int(page)  # Vector Search returns INT columns as floats
        blocks.append(f"[{i}] (source: {source}, p.{page})\n{doc.page_content}")
    return "\n\n".join(blocks)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        index = state["current_step_index"]
        step = state["plan"][index]

        docs = retriever.invoke(step)
        if not docs:
            result = f"{step} -> {NOT_FOUND}"
        else:
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(
                        content=f"Step: {step}\n\nExcerpts:\n{format_docs(docs)}"
                    ),
                ]
            )
            content = response.content if hasattr(response, "content") else str(response)
            result = f"{step} -> {str(content).strip()}"

        return {
            "step_results": state["step_results"] + [result],
            "current_step_index": index + 1,
        }

    return rag_agent
