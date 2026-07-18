"""Offline smoke test for the Document Analyst graph (Task 1.7 / Bonus A).

Builds the full graph with fake LLM / retriever / tool objects — no Databricks,
no network, no MCP subprocess — and runs a combined retrieval+calculation query
end-to-end. This is the test the Bonus A CI pipeline runs before any deploy.

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import prompts  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from agent.planner import parse_plan  # noqa: E402

FAKE_PLAN = [
    "Find Meridian's net revenue for fiscal year 2023",
    "Calculate a 10% increase on the revenue found in step 1",
]
FAKE_FACT = "Net revenue in FY2023 was ¥16.91 trillion [source: annual_report.pdf, p.4]"
FAKE_ANSWER = (
    "Net revenue in FY2023 was ¥16.91 trillion [source: annual_report.pdf, p.4]; "
    "a 10% increase gives ¥18.60 trillion (16.91 × 1.10)."
)


class FakeLLM:
    """Scripted LLM: dispatches on which system prompt the node sent."""

    def invoke(self, messages):
        system = messages[0].content
        human = messages[-1].content
        if system == prompts.PLANNER_PROMPT:
            return AIMessage(content='["' + '", "'.join(FAKE_PLAN) + '"]')
        if system == prompts.SUPERVISOR_PROMPT:
            return AIMessage(content="mcp_tools" if "Calculate" in human else "rag_agent")
        if system == prompts.RAG_EXTRACT_PROMPT:
            return AIMessage(content=FAKE_FACT)
        if system == prompts.MCP_STEP_PROMPT:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "16.91 * 1.10"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        if system == prompts.SYNTHESIZER_PROMPT:
            return AIMessage(content=FAKE_ANSWER)
        raise AssertionError(f"FakeLLM got an unexpected system prompt: {system[:60]!r}")

    def bind_tools(self, tools):
        return self


class FakeRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def invoke(self, query):
        self.queries.append(query)
        return [
            Document(
                page_content="Net revenue for FY2023 was ¥16.91 trillion.",
                metadata={"source": "annual_report.pdf", "page": 4},
            )
        ]


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return f"{expression} = 18.601"


def _build_fake_graph():
    return build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[calculate])


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


def test_graph_compiles_offline():
    graph = _build_fake_graph()
    assert {"planner", "supervisor", "rag_agent", "mcp_tools", "synthesizer"} <= set(
        graph.get_graph().nodes
    )


def test_combined_query_end_to_end():
    retriever = FakeRetriever()
    graph = build_graph(llm=FakeLLM(), retriever=retriever, tools=[calculate])
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What was the revenue in 2023, and what would a 10% increase look like?",
                }
            ]
        }
    )

    # Planner produced the decomposed plan.
    assert result["plan"] == FAKE_PLAN
    # Both specialists ran: RAG produced a cited fact, MCP executed the tool.
    assert len(result["step_results"]) == 2
    assert "[source: annual_report.pdf, p.4]" in result["step_results"][0]
    assert "18.601" in result["step_results"][1]
    # The RAG agent retrieved for the decomposed step, not the raw question.
    assert retriever.queries == [FAKE_PLAN[0]]
    # Final answer surfaced on BOTH channels — messages[-1] is what serving reads.
    assert result["final_answer"] == FAKE_ANSWER
    assert result["messages"][-1].content == FAKE_ANSWER


def test_parse_plan_falls_back_on_garbage():
    assert parse_plan("I cannot help with that.") is None
    assert parse_plan('```json\n["step one", "step two"]\n```') == ["step one", "step two"]


class EmptyRetriever:
    """Simulates a query with no relevant chunks in the index."""

    def invoke(self, query):
        return []


def test_empty_retrieval_degrades_gracefully():
    """Task 1.4/1.6: empty retrieval yields 'not found in documents' and the
    synthesizer still produces a final answer instead of crashing."""
    graph = build_graph(llm=FakeLLM(), retriever=EmptyRetriever(), tools=[calculate])
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What was the revenue in 2023, plus 10%?"}]}
    )
    assert "not found in documents" in result["step_results"][0]
    assert result["messages"][-1].content  # non-empty answer despite the miss
