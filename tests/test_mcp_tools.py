from __future__ import annotations

import respx
from httpx import Response

from tg_radar.agent_core.models import ToolDangerLevel
from tg_radar.agent_tools import HttpMcpToolClient, McpToolRegistry


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> list[dict]:
        return [
            {
                "name": "docs_search",
                "description": "Search docs",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "danger_level": "network",
                "timeout_sec": 7,
            }
        ]

    async def call_tool(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        return {"ok": True, "summary": "docs found", "data": {"items": [{"title": args["query"]}]}}


async def test_mcp_tool_registry_maps_specs_and_runs_tool():
    client = FakeMcpClient()
    registry = McpToolRegistry(client)

    specs = await registry.refresh()
    result = await registry.run("docs_search", {"query": "agent runtime"})

    assert specs[0].name == "docs_search"
    assert specs[0].input_schema["required"] == ["query"]
    assert specs[0].danger_level == ToolDangerLevel.network
    assert specs[0].timeout_sec == 7
    assert result.ok is True
    assert result.summary == "docs found"
    assert result.data["items"][0]["title"] == "agent runtime"
    assert client.calls == [("docs_search", {"query": "agent runtime"})]


async def test_mcp_tool_registry_filters_enabled_tools():
    registry = McpToolRegistry(FakeMcpClient(), enabled_tools={"other"})

    specs = await registry.refresh()
    result = await registry.run("docs_search", {"query": "x"})

    assert specs == []
    assert result.ok is False
    assert result.summary == "tool disabled"


@respx.mock
async def test_http_mcp_tool_client_uses_bridge_contract():
    respx.get("http://mcp.local/tools").mock(
        return_value=Response(200, json={"tools": [{"name": "docs_search"}]})
    )
    respx.post("http://mcp.local/tools/call").mock(
        return_value=Response(200, json={"ok": True, "summary": "called", "data": {"count": 1}})
    )
    client = HttpMcpToolClient("http://mcp.local/")

    tools = await client.list_tools()
    result = await client.call_tool("docs_search", {"query": "x"})

    assert tools == [{"name": "docs_search"}]
    assert result["summary"] == "called"
