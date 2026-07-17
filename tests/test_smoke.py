"""Offline smoke test for the Document Analyst graph (Bonus A test target)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeResponse:
    def __init__(self, content: str = "", tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FakeRetriever:
    def invoke(self, query: str):
        return [
            FakeDocument(
                "Meridian reported net revenue of \u00a516.91 trillion in fiscal year 2023.",
                {"source": "annual_report.pdf", "page": 4},
            )
        ]


class FakeTool:
    name = "growth_rate"

    def invoke(self, args: dict):
        return "growth_rate(16.91, rate=0.10, years=1) = 18.601"


class FakeStructuredLLM:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        step_text = messages[-1]["content"].lower()
        calc_keywords = ("calculate", "growth", "%", "compare", "convert")
        next_agent = "mcp_tools" if any(kw in step_text for kw in calc_keywords) else "rag_agent"
        return self.schema(next_agent=next_agent)


class FakeToolCallingLLM:
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, messages):
        tool = self.tools[0]
        return FakeResponse(
            content="",
            tool_calls=[{"name": tool.name, "args": {"start_value": 16.91, "rate": 0.10, "years": 1}, "id": "call_1"}],
        )


class FakeLLM:
    def invoke(self, messages):
        system = messages[0]["content"] if messages and "content" in messages[0] else ""
        if "planning module" in system:
            content = json.dumps(
                [
                    "Find Meridian's net revenue for fiscal year 2023",
                    "Calculate 10% growth on that revenue",
                ]
            )
        elif "retrieval-extraction module" in system:
            content = (
                "Meridian's net revenue in FY2023 was \u00a516.91 trillion "
                "[source: annual_report.pdf, p.4]."
            )
        elif "synthesis module" in system:
            content = (
                "Net revenue in FY2023 was \u00a516.91 trillion "
                "[source: annual_report.pdf, p.4]; a 10% increase projects to \u00a518.6 trillion."
            )
        else:
            content = "ok"
        return FakeResponse(content=content)

    def with_structured_output(self, schema):
        return FakeStructuredLLM(schema)

    def bind_tools(self, tools):
        return FakeToolCallingLLM(tools)


def test_graph_module_imports():
    from agent.graph import build_graph  # noqa: F401


def test_prepare_subprocess_streams_uses_real_stderr(monkeypatch):
    import agent.graph as graph_module

    class NoFilenoStream:
        def write(self, text: str):
            return None

        def flush(self):
            return None

    fake_stderr = NoFilenoStream()
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    with graph_module._use_real_os_streams():
        assert sys.stderr is not fake_stderr
        assert hasattr(sys.stderr, "fileno")
        assert sys.stderr.fileno() is not None

    assert sys.stderr is fake_stderr


def test_combined_query_end_to_end():
    from agent.graph import build_graph

    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])

    result = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "What was the FY2023 net revenue, and what would a 10% increase look like?"}
            ],
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "next_agent": "",
            "final_answer": "",
        }
    )

    assert len(result["plan"]) == 2
    assert len(result["step_results"]) == 2
    assert result["current_step_index"] == len(result["plan"])
    assert result["final_answer"]
    assert result["messages"][-1].content == result["final_answer"]
