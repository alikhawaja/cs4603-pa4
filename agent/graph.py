"""Full Document Analyst graph (Tasks 1.5 + 1.7).

MCP integration notes (see spec Task 1.5 caveat):
- Tools are loaded ONCE at graph-build time, never per request.
- Tool invocation is kept synchronous. MCP adapter tools are async-only, so a
  small bridge runs their coroutines with `asyncio.run`, or on a worker thread
  when an event loop is already running (Jupyter, some serving stacks).
- When MCP_SERVER_URL is set the client connects to a remote streamable-HTTP
  server (Bonus C); otherwise it spawns the GIVEN stdio server bundled with
  the model artifact.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import io
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def _run_coroutine(coro):
    """Run a coroutine to completion from synchronous code, loop or no loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # An event loop is already running (e.g. Jupyter): asyncio.run() would
    # raise, so execute on a throwaway thread with its own loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _has_fileno(stream) -> bool:
    try:
        stream.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation):
        return False
    return True


@functools.cache
def _ensure_stdio_errlog_usable() -> None:
    """Make stdio MCP subprocess spawning safe under Jupyter/nbconvert.

    `mcp.client.stdio.stdio_client` defaults its `errlog` to whatever
    `sys.stderr` was at import time, and subprocess creation calls
    `errlog.fileno()`. Notebook kernels replace stderr with a captured stream
    that has no file descriptor, which crashes every server spawn. Wrap
    `stdio_client` so a descriptor-less errlog is replaced with devnull; a
    real stderr (terminal, serving container) is passed through untouched.
    """
    import langchain_mcp_adapters.sessions as sessions_mod
    import mcp.client.stdio as stdio_mod

    original = stdio_mod.stdio_client
    devnull = open(os.devnull, "w")  # noqa: SIM115 — kept open for process lifetime

    def stdio_client_safe(server, errlog=None):
        if errlog is None:
            errlog = sys.stderr
        if not _has_fileno(errlog):
            errlog = devnull
        return original(server, errlog=errlog)

    stdio_mod.stdio_client = stdio_client_safe
    sessions_mod.stdio_client = stdio_client_safe  # imported by name there


def _default_server_path() -> str:
    """Path to the GIVEN stdio server, resolved relative to the packaged code."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "tools", "mcp_server.py")


def _mint_app_oauth_token() -> str | None:
    """OAuth access token for Databricks Apps ingress (Bonus C).

    App URLs reject PATs — they require an OAuth bearer token. When
    MCP_CLIENT_ID / MCP_CLIENT_SECRET (a service principal with CAN_USE on the
    app) are configured, mint a client-credentials token; otherwise fall back
    to MCP_SERVER_TOKEN if the caller supplied one directly.
    """
    client_id = os.environ.get("MCP_CLIENT_ID")
    client_secret = os.environ.get("MCP_CLIENT_SECRET")
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if client_id and client_secret and host:
        import httpx

        resp = httpx.post(
            f"{host}/oidc/v1/token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials", "scope": "all-apis"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    return os.environ.get("MCP_SERVER_TOKEN")


def load_mcp_tools(server_path: str | None = None):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    mcp_url = os.environ.get("MCP_SERVER_URL")
    if mcp_url:  # Bonus C — remote streamable-HTTP server
        try:
            connection = {
                "url": f"{mcp_url.rstrip('/')}/mcp",
                "transport": "streamable_http",
            }
            token = _mint_app_oauth_token()
            if token:
                connection["headers"] = {"Authorization": f"Bearer {token}"}
            client = MultiServerMCPClient({"analyst": connection})
            return _run_coroutine(client.get_tools())
        except Exception as exc:
            # Never let a transient network failure at graph-build time kill a
            # container startup (DEPLOYMENT_FAILED). The bundled stdio server
            # defines identical tools, so fall back to it with a warning.
            print(
                f"WARNING: remote MCP server {mcp_url} unreachable at load "
                f"({type(exc).__name__}: {exc}); falling back to bundled stdio server",
                file=sys.__stderr__ or sys.stderr,
            )

    # Part 1 — stdio subprocess bundled with the model
    _ensure_stdio_errlog_usable()
    connection = {
        "command": sys.executable,
        "args": [server_path or _default_server_path()],
        "transport": "stdio",
    }
    client = MultiServerMCPClient({"analyst": connection})
    return _run_coroutine(client.get_tools())


def _call_tool(tool, args: dict) -> str:
    try:
        result = tool.invoke(args)
    except NotImplementedError:
        # MCP adapter tools are async-only; bridge to their coroutine.
        result = _run_coroutine(tool.ainvoke(args))
    # MCP adapter tools return a list of content blocks; unwrap to plain text.
    if isinstance(result, list):
        texts = [b.get("text", "") for b in result if isinstance(b, dict)]
        if texts:
            return "\n".join(t for t in texts if t)
    return str(result)


def make_mcp_node(tools, llm):
    tool_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        index = state["current_step_index"]
        step = state["plan"][index]
        facts = "\n".join(f"- {r}" for r in state["step_results"]) or "(none yet)"

        response = llm_with_tools.invoke(
            [
                SystemMessage(content=MCP_STEP_PROMPT),
                HumanMessage(content=f"Facts gathered so far:\n{facts}\n\nStep: {step}"),
            ]
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]  # exactly one tool call per step, by design
            tool = tool_by_name.get(call["name"])
            if tool is None:
                outcome = f"error: model requested unknown tool '{call['name']}'"
            else:
                try:
                    outcome = _call_tool(tool, call["args"])
                except Exception as exc:
                    # e.g. the remote MCP app is stopped: record a visible
                    # step failure instead of crashing the whole request.
                    outcome = (
                        f"error: tool call '{call['name']}' failed "
                        f"({type(exc).__name__}: {str(exc)[:200]})"
                    )
            result = f"{step} -> {outcome}"
        else:
            # Model answered in prose despite instructions — keep whatever it
            # said so the synthesizer can still work with it.
            content = response.content if hasattr(response, "content") else str(response)
            result = f"{step} -> {str(content).strip()}"

        return {
            "step_results": state["step_results"] + [result],
            "current_step_index": index + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble and compile the Document Analyst graph.

    Dependencies are injectable so tests can pass fakes; production callers
    omit them and get the configured Databricks clients.
    """
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
    builder.add_conditional_edges("supervisor", route_from_supervisor, [RAG, MCP, SYNTH])
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)

    return builder.compile()
