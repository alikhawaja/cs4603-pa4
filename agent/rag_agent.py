"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations

from agent.state import AnalystState
from agent.prompts import RAG_EXTRACT_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

def format_docs(docs) -> str:
    return "\n\n".join([
        f"[Source: {doc.metadata.get('source')}, Page {doc.metadata.get('page')}]\n{doc.page_content}"
        for doc in docs
    ])

def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        # raise NotImplementedError("Task 1.4: implement the RAG node")

        current_step = state["plan"][state["current_step_index"]]

        docs = retriever.invoke(current_step)

        if not docs:
            result = f"Step: {current_step}\n Result: Not found in documents"
        else:
            formatted_doc = format_docs(docs)

            system_prompt = RAG_EXTRACT_PROMPT

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Context {docs} \n\n Question: {current_step}")
            ])

            result = response.content
        
        return {
            "step_results": state["step_results"] + [result],
            "current_step_index": state["current_step_index"] + 1,
        }

    return rag_agent
