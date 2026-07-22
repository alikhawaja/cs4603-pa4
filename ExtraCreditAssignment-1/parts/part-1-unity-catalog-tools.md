# Extra-Credit 1 · PART 1

## Unity Catalog Functions as Governed Tools

> Part of **CS 4603 Extra-Credit Assignment 1**. Start here after reading [`../tutorials/uc-function-tools.md`](../tutorials/uc-function-tools.md).
> ← Back to the [overview](../README.md) · Next: [Part 2 — Genie](part-2-genie-structured-data.md)

---

> **Read this first.** Before starting Part 1, read the tutorial [`../tutorials/uc-function-tools.md`](../tutorials/uc-function-tools.md). It walks through the **architecture and design** of working with Unity Catalog Functions as agent tools — what a UC Function is, how it differs from your `@tool` and MCP tools, how `UCFunctionToolkit` turns it into a tool, where execution happens, and how governance and automatic authorization fit together. It is a good, self-contained tutorial and the tasks below assume you have gone through it.

Your PA4 agent calls math/finance tools through an MCP server (`tools/mcp_server.py`). Those tools are invisible to the platform. In this part you make them **first-class governed assets** in Unity Catalog.

### Task 1.1: Register Python functions in Unity Catalog

Port at least **three** of your PA4 tools (e.g. `growth_rate`, `percentage_change`, `compare_values`) into Unity Catalog as Python functions.

```python
# Expected pattern (Python client):
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

def compound_growth(principal: float, rate: float, periods: int) -> float:
    """Return principal * (1 + rate) ** periods.
    Args:
        principal: starting value.
        rate: growth rate per period as a decimal (0.08 = 8%).
        periods: number of periods.
    """
    return principal * (1 + rate) ** periods

client.create_python_function(
    func=compound_growth, catalog="main", schema="default", replace=True
)
```

Requirements:
- Functions live under your UC schema (e.g. `main.default.<name>`).
- **Type hints and a complete docstring are mandatory** — the toolkit turns them into the tool schema the LLM sees. Sloppy docstrings = bad tool calls.
- Verify each function executes with a direct `client.execute_function(...)` (or a SQL `SELECT main.default.compound_growth(100, 0.08, 3)`).

### Task 1.2: Register a SQL function

Author **one** function in SQL (`CREATE FUNCTION`) rather than Python — e.g. a governed lookup or a unit conversion.

```sql
-- Expected pattern:
CREATE OR REPLACE FUNCTION main.default.to_billions(amount_yen DOUBLE)
RETURNS DOUBLE
COMMENT 'Convert a raw JPY amount to billions of yen.'
RETURN amount_yen / 1e9;
```

In your write-up, note **when a SQL function is preferable to Python** (set-based logic, pushdown, no Python runtime).

### Task 1.3: Wire UC Functions into the agent

Replace the MCP tool node in your PA4 graph with a tool node backed by `UCFunctionToolkit`.

```python
# Expected pattern:
from databricks_langchain import UCFunctionToolkit

toolkit = UCFunctionToolkit(
    function_names=[
        "main.default.compound_growth",
        "main.default.percent_change",
        "main.default.to_billions",
    ]
)
tools = toolkit.tools   # LangChain tools -> bind to your LLM / ToolNode
```

Requirements:
- The **planner → supervisor → (rag | tools) → synthesizer** flow is unchanged; only the *tools* node's backing changes.
- Re-run the canonical PA4 query (*"What was Meridian's FY2023 net revenue, and what would it be after 3 years of 8% compound growth?"*) and show it produces the same numeric answer through UC Functions.
- Keep both graphs runnable (`graph.py` = MCP, `graph_uc.py` = UC) so you can compare.

### Task 1.4: Governance — permissions & lineage

- `GRANT EXECUTE` on your functions to an appropriate principal (yourself/a group); show the grant.
- Locate the functions in **Catalog Explorer** and capture their **lineage / metadata** (screenshot in the notebook).
- **Analysis (write-up):** Compare **MCP server tools (PA4)** vs **UC Function tools (this assignment)** across at least: *discovery, permissions/governance, reuse across agents, versioning, transport, and auth.* When would you still choose an MCP server?

### Task 1.5: Redeploy with automatic authorization

Log and redeploy the UC-tool agent so the serving endpoint calls the functions with **automatic short-lived credentials** — no tokens in code — by declaring them as resources (the same mechanism the `databricks_deployment_v2/` example uses for serving endpoints).

```python
# Expected pattern:
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint

resources = [
    DatabricksServingEndpoint(endpoint_name="<your-llm-endpoint>"),
    DatabricksFunction(function_name="main.default.compound_growth"),
    DatabricksFunction(function_name="main.default.percent_change"),
    DatabricksFunction(function_name="main.default.to_billions"),
]
# ...log_model(..., resources=resources) -> register in UC -> agents.deploy(...)
```

Confirm the deployed endpoint answers a tool-using query end-to-end.

---

← Back to the [overview](../README.md) · Next: [Part 2 — Genie](part-2-genie-structured-data.md)
