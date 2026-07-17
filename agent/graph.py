import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer

_DEFAULT_SERVER_PATH = str(Path(__file__).resolve().parent.parent / "tools" / "mcp_server.py")


@contextmanager
def _use_real_os_streams():
    """Temporarily swap notebook streams for real OS file handles.

    The MCP stdio client uses subprocess.Popen(..., stderr=errlog) and expects
    errlog to be a file-like object that exposes fileno(). Jupyter/VS Code
    notebook streams (IPython's OutStream) do not support fileno() on Windows,
    which causes the subprocess launch to fail before the MCP server starts.
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.__stdout__
    temp_stderr = open(os.devnull, "w", encoding="utf-8")
    sys.stderr = temp_stderr

    original_create_process = None
    try:
        import mcp.client.stdio as mcp_stdio
    except Exception:
        mcp_stdio = None
    else:
        original_create_process = mcp_stdio._create_platform_compatible_process

        async def patched_create_process(command, args, env=None, errlog=None, cwd=None):
            return await original_create_process(
                command,
                args,
                env=env,
                errlog=temp_stderr,
                cwd=cwd,
            )

        mcp_stdio._create_platform_compatible_process = patched_create_process

    try:
        yield
    finally:
        if mcp_stdio is not None and original_create_process is not None:
            mcp_stdio._create_platform_compatible_process = original_create_process
        sys.stdout, sys.stderr = old_stdout, old_stderr
        temp_stderr.flush()
        temp_stderr.close()


def _run_async(coro):
    """Run an async coroutine safely in a notebook kernel."""
    import nest_asyncio

    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


def load_mcp_tools(server_path: str | None = None):
    """Connect to the GIVEN MCP server over stdio and return its LangChain tools."""
    server_path = server_path or _DEFAULT_SERVER_PATH

    with _use_real_os_streams():
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "analyst": {
                    "command": sys.executable,
                    "args": [server_path],
                    "transport": "stdio",
                }
            }
        )
        return _run_async(client.get_tools())

def make_mcp_node(tools, llm):
    """Return a node that executes one calculation step via exactly one MCP tool."""
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def mcp_tools(state: AnalystState) -> dict:
        idx = state["current_step_index"]
        step = state["plan"][idx]
        prior_context = "\n".join(state.get("step_results", [])) or "(no prior results)"

        response = llm_with_tools.invoke(
            [
                {"role": "system", "content": MCP_STEP_PROMPT},
                {"role": "user", "content": f"Step: {step}\n\nEarlier step results:\n{prior_context}"},
            ]
        )

        if not response.tool_calls:
            result = f"Step {idx + 1} ('{step}'): No tool call made — {response.content or 'no result returned'}"
            return {"step_results": [result], "current_step_index": idx + 1}

        call = response.tool_calls[0]  # exactly one tool call, per MCP_STEP_PROMPT
        tool = tools_by_name.get(call["name"])
        if tool is None:
            tool_output = f"Unknown tool '{call['name']}' requested."
        else:
            try:
                tool_output = tool.invoke(call["args"])
            except Exception as exc:  # noqa: BLE001
                tool_output = f"Error calling tool '{call['name']}': {exc}"

        result = f"Step {idx + 1} ('{step}'): {tool_output}"
        return {"step_results": [result], "current_step_index": idx + 1}

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble and compile the full Document Analyst graph."""
    if llm is None:
        from config import get_chat_llm
        llm = get_chat_llm()
    if retriever is None:
        from rag.store import get_retriever
        retriever = get_retriever()
    if tools is None:
        tools = load_mcp_tools()

    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node(RAG, make_rag_agent(retriever, llm))
    builder.add_node(MCP, make_mcp_node(tools, llm))
    builder.add_node(SYNTH, make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: RAG, MCP: MCP, SYNTH: SYNTH},
    )
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)

    return builder.compile()