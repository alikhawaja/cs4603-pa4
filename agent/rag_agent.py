"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search."""

from __future__ import annotations

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState


def _format_docs(docs) -> str:
    """Format retrieved chunks with numbered [source: file, p.N] citations."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        lines.append(f"[{i}] (source: {source}, p.{page})\n{doc.page_content}")
    return "\n\n".join(lines)


def make_rag_agent(retriever, llm):
    """Return a RAG agent node bound to the given retriever + chat LLM."""

    def rag_agent(state: AnalystState) -> dict:
        idx = state["current_step_index"]
        step = state["plan"][idx]

        docs = retriever.invoke(step)

        if not docs:
            result = f"Step {idx + 1} ('{step}'): Not found in documents."
            return {"step_results": [result], "current_step_index": idx + 1}

        context = _format_docs(docs)
        response = llm.invoke(
            [
                {"role": "system", "content": RAG_EXTRACT_PROMPT},
                {"role": "user", "content": f"Step: {step}\n\nExcerpts:\n{context}"},
            ]
        )
        result = f"Step {idx + 1} ('{step}'): {response.content}"
        return {"step_results": [result], "current_step_index": idx + 1}

    return rag_agent