from __future__ import annotations

from typing import Any, Protocol

import httpx

from tg_radar.agent_core.models import AgentObservation, AgentPhase, ToolDangerLevel, ToolSpec


class McpToolClient(Protocol):
    async def list_tools(self) -> list[dict[str, Any]]:
        ...

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        ...


class McpToolRegistry:
    def __init__(
        self,
        client: McpToolClient,
        enabled_tools: set[str] | None = None,
        default_timeout_sec: int = 30,
        initial_tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.enabled_tools = enabled_tools
        self.default_timeout_sec = default_timeout_sec
        self._specs: list[ToolSpec] | None = self._filter_specs(
            [self._to_spec(item) for item in (initial_tools or [])]
        )

    def specs(self) -> list[ToolSpec]:
        return list(self._specs or [])

    async def refresh(self) -> list[ToolSpec]:
        self._specs = self._filter_specs([self._to_spec(item) for item in await self.client.list_tools()])
        return list(self._specs)

    async def run(self, name: str, args: dict[str, Any]) -> AgentObservation:
        if self.enabled_tools is not None and name not in self.enabled_tools:
            return AgentObservation(tool_name=name, ok=False, summary="tool disabled")
        if name not in {tool.name for tool in self.specs()}:
            await self.refresh()
        if name not in {tool.name for tool in self.specs()}:
            return AgentObservation(tool_name=name, ok=False, summary="unknown tool")
        try:
            result = await self.client.call_tool(name, args)
            return AgentObservation(
                tool_name=name,
                ok=bool(result.get("ok", True)),
                summary=str(result.get("summary") or "mcp tool completed"),
                data=dict(result.get("data") or result),
            )
        except Exception as exc:
            return AgentObservation(tool_name=name, ok=False, summary=f"{type(exc).__name__}: {exc}")

    def _to_spec(self, item: dict[str, Any]) -> ToolSpec:
        name = str(item.get("name") or "")
        return ToolSpec(
            name=name,
            description=str(item.get("description") or ""),
            risk_level=str(item.get("risk_level") or "network"),
            args_schema=dict(item.get("input_schema") or item.get("args_schema") or {}),
            input_schema=dict(item.get("input_schema") or item.get("args_schema") or {}),
            output_schema=dict(item["output_schema"]) if isinstance(item.get("output_schema"), dict) else None,
            side_effects=bool(item.get("side_effects") or False),
            danger_level=ToolDangerLevel(str(item.get("danger_level") or "network")),
            timeout_sec=int(item.get("timeout_sec") or self.default_timeout_sec),
            requires_approval=bool(item.get("requires_approval") or False),
            allowed_phases=[AgentPhase(str(phase)) for phase in item.get("allowed_phases") or []],
        )

    def _filter_specs(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        if self.enabled_tools is None:
            return specs
        return [tool for tool in specs if tool.name in self.enabled_tools]


class HttpMcpToolClient:
    def __init__(self, endpoint: str, timeout_seconds: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def list_tools(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.endpoint}/tools")
            response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return [dict(item) for item in data]
        return [dict(item) for item in data.get("tools", [])]

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        payload = {"name": name, "arguments": args}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.endpoint}/tools/call", json=payload)
            response.raise_for_status()
        data = response.json()
        return dict(data)
