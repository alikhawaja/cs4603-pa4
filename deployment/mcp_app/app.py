"""Bonus C — standalone MCP tool server as a Databricks App.

Reuses the GIVEN tool definitions from tools/mcp_server.py unchanged, but
serves them over the streamable-http transport instead of stdio, so any agent
(local or deployed) can call the tools remotely at  https://<app-url>/mcp .

Run locally:   uv run python deployment/mcp_app/app.py
On Databricks: deployed via `databricks apps deploy` with app.yaml.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Repo layout: tools/ lives two levels up. Staged app layout: tools/ sits
# right next to app.py. Support both so the same file runs everywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from tools.mcp_server import mcp  # noqa: E402 — needs the sys.path bootstrap above

if __name__ == "__main__":
    # Databricks Apps inject the port to bind via $DATABRICKS_APP_PORT.
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http")
