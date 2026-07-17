# test_mcp.py — quick smoke test for tools/mcp_server.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="uv", args=["run", "python", "tools/mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            result = await session.call_tool("growth_rate", {"start_value": 16.91, "rate": 0.08, "years": 3})
            print("growth_rate result:", result.content[0].text)

asyncio.run(main())