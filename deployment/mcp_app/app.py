"""Bonus C — standalone MCP server as a Databricks App (HTTP transport).

The given `tools/mcp_server.py` uses the stdio transport (bundled inside the model). Here
we reuse the *same tool definitions* but serve them over **streamable-http** so the tool
server runs as its own long-lived Databricks App — decoupled from the model, which then
connects to it remotely (see `MCP_SERVER_URL` in `agent/graph.py::load_mcp_tools`).

Databricks Apps provide the port to bind on via `$DATABRICKS_APP_PORT` (default 8000); the
app must listen on 0.0.0.0.
"""

from __future__ import annotations

import os

from tools.mcp_server import mcp  # reuse the GIVEN tool definitions

if __name__ == "__main__":
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http")
