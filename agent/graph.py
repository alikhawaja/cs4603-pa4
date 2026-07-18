"""Full Document Analyst graph (Tasks 1.5 + 1.7).

Task 1.5 (this file):
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio and
    return its tools as LangChain tools (loaded ONCE at build time).
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM pick
    and call exactly one MCP tool, then append the result and increment the index.

Async caveat (see DEPLOYMENT_GUIDE.md §4): MCP tools are async and stdio calls may
relaunch the subprocess, so we load tools once up front and invoke them synchronously
via `_run_async`, which tolerates being called with or without a running event loop.
"""

from __future__ import annotations

import asyncio
import os

from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer

# Default path to the GIVEN stdio MCP server (tools/mcp_server.py at the repo root).
_DEFAULT_SERVER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "mcp_server.py"
)


def _fresh_loop():
    """Create an event loop that can spawn subprocesses on this platform.

    On Windows, asyncio subprocesses require a ProactorEventLoop. Some hosts (MLflow model
    loading, Jupyter, anything importing pyzmq) install the *SelectorEventLoop* policy,
    which cannot create subprocesses — so `asyncio.run` there fails the stdio MCP handshake
    with "Connection closed". We therefore build a Proactor loop explicitly rather than
    trusting the ambient policy.
    """
    import sys

    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    return asyncio.new_event_loop()


def _run_coro_in_fresh_loop(coro):
    loop = _fresh_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _run_async(coro):
    """Run an async coroutine from sync code, whether or not a loop is already running.

    We never rely on `asyncio.run` (whose loop type depends on the ambient policy). When no
    loop is active we run on a fresh subprocess-capable loop in this thread; when a loop is
    already running (Jupyter, some serving runtimes) we do the same on a dedicated worker
    thread so we don't nest loops. Either way MCP tool calls stay synchronous to the graph.
    """
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if not running:
        return _run_coro_in_fresh_loop(coro)

    import threading

    box: dict = {}

    def _runner():
        try:
            box["result"] = _run_coro_in_fresh_loop(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            box["error"] = exc

    t = threading.Thread(target=_runner)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _ensure_subprocess_stderr() -> None:
    """Ensure the stdio MCP subprocess gets a real stderr (with a file descriptor).

    `mcp.client.stdio.stdio_client` defaults its `errlog` to `sys.stderr` (bound at import
    time). Inside Jupyter — and some serving runtimes — `sys.stderr` is replaced by a
    stream that has no `fileno()`, which makes the Windows subprocess launch fail with
    `UnsupportedOperation: fileno`. If the current default has no usable fileno, point it
    at os.devnull instead (we don't rely on the tool server's stderr).
    """
    import mcp.client.stdio as _stdio

    # stdio_client is wrapped by @asynccontextmanager, so its real `errlog` default lives
    # on the underlying __wrapped__ function, not the wrapper.
    target = getattr(_stdio.stdio_client, "__wrapped__", _stdio.stdio_client)
    defaults = target.__defaults__ or ()
    if not defaults:
        return
    errlog = defaults[0]
    try:
        errlog.fileno()
        return  # already usable
    except Exception:
        target.__defaults__ = (open(os.devnull, "w"),) + defaults[1:]


def load_mcp_tools(server_path: str | None = None):
    """Connect to the MCP server and return its tools as LangChain tools.

    Two transports, selected by the `MCP_SERVER_URL` env var:
      - **Remote HTTP (Bonus C):** if `MCP_SERVER_URL` is set, connect over streamable-http
        with a bearer token to a separately-deployed MCP server (a Databricks App). The
        tool server is then decoupled from the model — scaled/redeployed independently.
      - **Bundled stdio (Part 1/2, default):** otherwise spawn the given `mcp_server.py` as
        a stdio subprocess shipped inside the model artifact.
    Loaded once at graph-build time.
    """
    import sys

    from langchain_mcp_adapters.client import MultiServerMCPClient

    mcp_url = os.environ.get("MCP_SERVER_URL")
    if mcp_url:
        token = os.environ.get("DATABRICKS_TOKEN", "")
        client = MultiServerMCPClient(
            {
                "analyst": {
                    "url": f"{mcp_url.rstrip('/')}/mcp",
                    "transport": "streamable_http",
                    "headers": {"Authorization": f"Bearer {token}"},
                }
            }
        )
        return _run_async(client.get_tools())

    _ensure_subprocess_stderr()
    server_path = server_path or _DEFAULT_SERVER
    client = MultiServerMCPClient(
        {
            "analyst": {
                # Use THIS interpreter (sys.executable), not a bare "python" from PATH —
                # PATH may resolve to a different interpreter that lacks `mcp`/our deps,
                # which makes the stdio subprocess exit with "Connection closed".
                "command": sys.executable,
                "args": [server_path],
                "transport": "stdio",
            }
        }
    )
    return _run_async(client.get_tools())


def _tool_output_to_text(output) -> str:
    """Flatten an MCP tool result into a plain string.

    langchain-mcp-adapters may return a raw string or a list of content blocks like
    [{"type": "text", "text": "..."}]; normalise both to clean text.
    """
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        parts = [
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in output
        ]
        return " ".join(p for p in parts if p).strip()
    return str(output).strip()


def make_mcp_node(tools, llm):
    """Build the MCP calculation node: LLM chooses one tool, we execute it."""
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)
        step = plan[idx] if idx < len(plan) else ""
        results_so_far = "\n".join(state.get("step_results", [])) or "(none yet)"

        response = llm_with_tools.invoke(
            [
                {"role": "system", "content": MCP_STEP_PROMPT},
                {"role": "user", "content": f"STEP: {step}\n\nRESULTS SO FAR:\n{results_so_far}"},
            ]
        )

        calls = getattr(response, "tool_calls", None) or []
        if calls:
            call = calls[0]  # exactly one tool per step
            tool = tools_by_name.get(call["name"])
            if tool is not None:
                output = _run_async(tool.ainvoke(call["args"]))
                fact = f"Step '{step}': {_tool_output_to_text(output)}"
            else:
                fact = f"Step '{step}': error - unknown tool '{call['name']}'"
        else:
            # No tool call — fall back to the model's text (or flag it).
            content = (getattr(response, "content", "") or "").strip()
            fact = f"Step '{step}': {content or 'no calculation performed'}"

        return {
            "step_results": state.get("step_results", []) + [fact],
            "current_step_index": idx + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble and compile the full Document Analyst graph.

    Dependencies are injected so the graph can be built offline with fakes (the smoke
    test) or with real Databricks-backed objects (local runs / deployment):
      - llm:       chat model; defaults to config.get_chat_llm().
      - retriever: Vector Search retriever; defaults to rag.store.get_retriever().
      - tools:     MCP tools list; defaults to load_mcp_tools().

    Graph shape (per spec):
        START -> planner -> supervisor -> {rag_agent | mcp_tools | synthesizer}
        rag_agent -> supervisor,  mcp_tools -> supervisor   (loop until steps done)
        synthesizer -> END
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
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: RAG, MCP: MCP, SYNTH: SYNTH},
    )
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)

    return builder.compile()
