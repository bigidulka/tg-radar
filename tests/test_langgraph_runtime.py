from __future__ import annotations

from tg_radar.agent_core.models import AgentAction, AgentActionType, AgentContextPack, AgentRequest
from tg_radar.agent_runtime import LangGraphRuntime


class AsyncGraph:
    def __init__(self) -> None:
        self.state = {}

    async def ainvoke(self, state):
        self.state = state
        return {
            "action": {
                "type": "tool_call",
                "tool_name": "read_file",
                "args": {"path": "pyproject.toml"},
                "reason": "graph selected tool",
            }
        }


class SyncGraph:
    def invoke(self, state):
        return {"action": AgentAction(type=AgentActionType.final, reason="sync graph done")}


async def test_langgraph_runtime_uses_async_graph_action():
    graph = AsyncGraph()
    runtime = LangGraphRuntime(graph=graph)

    action = await runtime.next_action(
        AgentRequest(goal="inspect"),
        AgentContextPack(user_goal="inspect"),
        [],
    )

    assert action.type == AgentActionType.tool_call
    assert action.tool_name == "read_file"
    assert graph.state["request"]["goal"] == "inspect"


async def test_langgraph_runtime_uses_sync_graph_action():
    runtime = LangGraphRuntime(graph=SyncGraph())

    action = await runtime.next_action(
        AgentRequest(goal="inspect"),
        AgentContextPack(user_goal="inspect"),
        [],
    )

    assert action.type == AgentActionType.final
    assert action.reason == "sync graph done"


async def test_langgraph_runtime_falls_back_without_graph():
    runtime = LangGraphRuntime()

    action = await runtime.next_action(
        AgentRequest(goal="read", inputs={"tool_name": "read_file", "tool_args": {"path": "pyproject.toml"}}),
        AgentContextPack(user_goal="read"),
        [],
    )

    assert action.type == AgentActionType.tool_call
    assert action.tool_name == "read_file"
