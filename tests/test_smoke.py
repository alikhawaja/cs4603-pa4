"""Offline smoke test for the Document Analyst graph (Task 1.7 / Bonus A target).

Builds the full graph with FAKE llm / retriever / tools — no Databricks, no network —
and runs one combined retrieval+calculation query. Asserts the graph compiles, a plan is
produced, both specialists run, and the final answer surfaces on messages[-1].

This is the fast local feedback loop and the exact test Bonus A automates in CI.

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage  # noqa: E402


# ── Fakes ───────────────────────────────────────────────────────────────────
class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeToolLLM:
    """What `llm.bind_tools(tools)` returns: always requests the growth_rate tool."""

    def invoke(self, messages):
        return FakeMessage(
            tool_calls=[
                {
                    "name": "growth_rate",
                    "args": {"start_value": 16.91, "rate": 0.08, "years": 3},
                    "id": "call_1",
                }
            ]
        )


class FakeLLM:
    """Routes by inspecting the system prompt, so one fake serves every node."""

    def invoke(self, messages):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "planner" in system:
            return FakeMessage(
                '["Find Meridian net revenue for FY2023", '
                '"Calculate that value after 3 years of 8% growth"]'
            )
        if "supervisor" in system:
            if any(k in user for k in ("calculate", "growth", "compound")):
                return FakeMessage("mcp_tools")
            return FakeMessage("rag_agent")
        if "extract" in system:
            return FakeMessage("net revenue was ¥16.91 trillion [source: annual_report.pdf, p.4]")
        if "synthesizer" in system:
            return FakeMessage(
                "Meridian's FY2023 net revenue was ¥16.91 trillion "
                "[source: annual_report.pdf, p.4]; after 3 years of 8% growth ≈ ¥21.30 trillion."
            )
        return FakeMessage("ok")

    def bind_tools(self, tools):
        return FakeToolLLM()


class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class FakeRetriever:
    def invoke(self, query):
        return [FakeDoc("Net revenue rose to ¥16.91 trillion", {"source": "annual_report.pdf", "page": 4})]


class FakeTool:
    name = "growth_rate"

    async def ainvoke(self, args):
        return "16.91 at 8% CAGR for 3 years = 21.3017"


# ── Tests ───────────────────────────────────────────────────────────────────
def test_graph_module_imports():
    """The graph module must import cleanly (collection guard)."""
    from agent.graph import build_graph  # noqa: F401


def test_combined_query_end_to_end():
    """Full graph runs offline and returns a non-empty answer on messages[-1]."""
    from agent.graph import build_graph

    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What was Meridian's revenue in FY2023, "
                    "and what would it be after 3 years of 8% growth?",
                }
            ]
        }
    )

    # A plan was produced (2 steps: retrieval + computation).
    assert len(result["plan"]) == 2

    # Both specialists ran -> two collected step results.
    assert len(result["step_results"]) == 2
    joined = " ".join(result["step_results"])
    assert "16.91" in joined          # RAG fact
    assert "21.30" in joined or "21.3017" in joined  # MCP calculation

    # Final answer surfaced on the messages channel (deployment contract).
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content.strip()
    assert result["final_answer"].strip()
