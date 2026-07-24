"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
import asyncio
from agent.planner import make_planner
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage
from agent.prompts import MCP_STEP_PROMPT


async def load_mcp_tools(server_path: str | None = None):
    # raise NotImplementedError("Task 1.5: connect the MCP server and return its tools")
    import os
    import sys

    if server_path is None:
        server_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tools", "mcp_server.py"
        )

    client = MultiServerMCPClient({
        "analyst": {
            "command": sys.executable,
            "args": [server_path],
            "transport": "stdio"
        }
    })

    tools = await asyncio.wait_for(client.get_tools(), timeout=30)
    return tools

def make_mcp_node(tools, llm):
    _cache = {"tools": tools, "llm_with_tools": None}
   
    def _ensure_tools():
        if _cache["tools"] is None:
            _cache["tools"] = asyncio.run(load_mcp_tools())
        if _cache["llm_with_tools"] is None:
            _cache["llm_with_tools"] = llm.bind_tools(_cache["tools"])
        return _cache["tools"], _cache["llm_with_tools"]
    
    def mcp_tools(state: AnalystState) -> dict:
        # raise NotImplementedError("Task 1.5: implement the MCP tool node")

        tools, llm_with_tools = _ensure_tools()
        tools_name = {tool.name: tool for tool in tools}
        system_prompt = MCP_STEP_PROMPT
        current_step = state["plan"][state["current_step_index"]]

        response = llm_with_tools.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=current_step),
        ])

        if not response.tool_calls:
            result = f"Step: {current_step}\nResult: model did not call a tool"
        else:
            call = response.tool_calls[0]
            tool = tools_name[call["name"]]
            tool_result = asyncio.run(tool.ainvoke(call["args"]))
            result = f"Step: {current_step}\nResult: {tool_result}"
        
        return {
            "step_results": state["step_results"] + [result],
            "current_step_index": state["current_step_index"] + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    # raise NotImplementedError("Task 1.7: wire and compile the full graph")
    from config import get_chat_llm
    from rag.store import get_retriever

    if llm is None:
        llm = get_chat_llm()
    
    if retriever is None:
        retriever = get_retriever()

    planner = make_planner(llm)
    supervisor = make_supervisor(llm)
    rag_agent = make_rag_agent(retriever=retriever, llm=llm)
    mcp_tools = make_mcp_node(tools=tools, llm=llm)
    synthesizer = make_synthesizer(llm)
    
    builder = StateGraph(AnalystState)
    builder.add_node("planner", planner)
    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("mcp_tools", mcp_tools)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "rag_agent": "rag_agent",
        "mcp_tools": "mcp_tools",
        "synthesizer": "synthesizer",
    },
)
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    graph = builder.compile()

    return graph
