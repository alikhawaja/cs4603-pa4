"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

This node executes a single *retrieval* step of the plan:
  1. Use the current step text as the query and retrieve top-k chunks from the Vector
     Search index (via the `retriever` built by `rag/store.py::get_retriever()`).
  2. Format the chunks with `[source: file, p.N]` citations.
  3. Ask the LLM to extract one cited fact from those chunks (or "not found in
     documents" if the answer isn't present).
  4. Append that fact to `step_results` and advance `current_step_index` so the
     supervisor moves on to the next step.

The identical `retriever` object is used locally and inside the deployed serving
container, so this code path is deployment-agnostic (that is the point of routing all
retrieval through `rag/store.py`).
"""

from __future__ import annotations

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState

NOT_FOUND = "not found in documents"


def format_docs(docs) -> str:
    """Render retrieved documents as numbered, citation-tagged excerpts."""
    if not docs:
        return ""
    lines = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source", "annual_report.pdf")
        page = meta.get("page")
        # Vector Search may return page as a float (e.g. 4.0); show it as an int.
        try:
            page = int(float(page))
        except (TypeError, ValueError):
            page = "?"
        text = (getattr(d, "page_content", "") or "").strip()
        lines.append(f"[{i}] (source: {source}, p.{page})\n{text}")
    return "\n\n".join(lines)


def make_rag_agent(retriever, llm):
    """Build the RAG node bound to a `retriever` and a chat `llm`."""

    def rag_agent(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)
        step = plan[idx] if idx < len(plan) else ""

        docs = retriever.invoke(step)
        context = format_docs(docs)

        if not context:
            # Empty retrieval -> record a clear "not found" rather than hallucinating.
            fact = f"Step '{step}': {NOT_FOUND}"
        else:
            response = llm.invoke(
                [
                    {"role": "system", "content": RAG_EXTRACT_PROMPT},
                    {"role": "user", "content": f"STEP: {step}\n\nCONTEXT:\n{context}"},
                ]
            )
            answer = (getattr(response, "content", str(response)) or "").strip()
            fact = f"Step '{step}': {answer}"

        return {
            "step_results": state.get("step_results", []) + [fact],
            "current_step_index": idx + 1,
        }

    return rag_agent
