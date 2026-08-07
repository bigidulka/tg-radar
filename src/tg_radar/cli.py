import argparse
import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass, field

from tg_radar.app_state import build_state
from tg_radar.config import get_settings
from tg_radar.crawler import TelegramPublicCrawler
from tg_radar.db import init_db
from tg_radar.agent_eval import AgentEvalRunner
from tg_radar.rule_loader import load_rules
from tg_radar.schemas import CollectionAgentRequest, CollectionAgentMode, CollectionAgentResponse, StoryMatchReport
from tg_radar.story_catalog import load_story_rules, regenerate_catalog
from tg_radar.story_match import match_topic_stories

CLI_RULES = load_rules("cli_rules.json")


async def crawl(args: argparse.Namespace) -> None:
    crawler = TelegramPublicCrawler(get_settings())
    pages = await crawler.crawl_channel(args.channel, pages=args.pages)
    total = sum(len(page.messages) for page in pages)
    print({"channel": args.channel, "pages": len(pages), "messages": total})


def crawl_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("channel")
    parser.add_argument("--pages", type=int, default=1)
    args = parser.parse_args()
    with suppress(KeyboardInterrupt):
        asyncio.run(crawl(args))


async def agent(args: argparse.Namespace) -> None:
    state = build_state()
    await init_db(state.engine)
    try:
        payload = CollectionAgentRequest(
            goal=args.goal,
            mode=CollectionAgentMode(args.mode),
            phase=args.phase,
            keywords=args.keyword,
            seed_channels=args.seed_channel,
            task_name=args.task_name,
            depth=args.depth,
            limit=args.limit,
            pages_per_channel=args.pages,
            crawl=not args.no_crawl,
            interval_seconds=args.interval_seconds,
            enabled=not args.disabled,
            tool_name=args.tool_name,
            tool_args=json.loads(args.tool_args),
            approved_tools=args.approve,
            max_steps=args.max_steps,
        )
        result = await state.agent.run(payload)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        await _close_state(state)


def agent_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("goal")
    parser.add_argument("--mode", choices=[mode.value for mode in CollectionAgentMode], default=CollectionAgentMode.auto.value)
    parser.add_argument("--phase", choices=["understand", "plan", "edit", "verify", "fix", "finalize"])
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--seed-channel", action="append", default=[])
    parser.add_argument("--task-name")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--no-crawl", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--tool-name")
    parser.add_argument("--tool-args", default="{}")
    parser.add_argument("--approve", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()
    with suppress(KeyboardInterrupt):
        asyncio.run(agent(args))


async def agent_eval(args: argparse.Namespace) -> None:
    state = build_state()
    await init_db(state.engine)
    try:
        results = await AgentEvalRunner(state.agent.orchestrator).run_file(args.file)
        print(json.dumps([item.model_dump(mode="json") for item in results], ensure_ascii=False, indent=2))
        if any(not item.ok for item in results):
            raise SystemExit(1)
    finally:
        await _close_state(state)


def agent_eval_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    with suppress(KeyboardInterrupt):
        asyncio.run(agent_eval(args))


def story_catalog_main() -> None:
    parser = argparse.ArgumentParser(description="rebuild the story catalog from the publish-engine markdown")
    parser.add_argument("--source", default=get_settings().story_catalog_source)
    args = parser.parse_args()
    cases = regenerate_catalog(args.source)
    missing = [case.number for case in cases if not case.takeaway]
    print(f"cases={len(cases)} source={args.source}")
    print(f"without takeaway: {missing or '-'}")


def _print_story_report(report: StoryMatchReport) -> None:
    labels = load_story_rules()["cli"]
    print(labels["header"].format(topic=report.topic, window=report.window_days, cases=report.cases_total))
    print(
        labels["counters"].format(
            considered=report.signals_considered,
            matched=report.signals_matched,
            pairs=len(report.pairs),
        )
    )
    for note in report.notes:
        print(labels["note"].format(text=note))
    for pair in report.pairs:
        print()
        print(labels["case"].format(number=pair.case.number, title=pair.case.title))
        print(labels["meta"].format(format=pair.case.format, confidence=pair.confidence))
        if pair.multi_channel:
            print(
                labels["wave"].format(
                    signals=len(pair.signals),
                    channels=len(pair.channels),
                    names=", ".join(pair.channels),
                )
            )
        for signal in pair.signals:
            posted = signal.posted_at.strftime("%Y-%m-%d") if signal.posted_at else "-"
            print(labels["signal"].format(channel=signal.channel, posted=posted, url=signal.url))
            print(labels["gist"].format(text=signal.gist))
        print(labels["you_add"].format(text=pair.you_add))


async def stories(args: argparse.Namespace) -> None:
    state = build_state()
    try:
        async with state.sessionmaker() as session:
            report = await match_topic_stories(
                session,
                state.settings,
                args.topic,
                window_days=args.window_days,
                limit=args.limit,
                min_confidence=args.min_confidence,
            )
        if args.json:
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            _print_story_report(report)
    finally:
        await _close_state(state)


def stories_main() -> None:
    parser = argparse.ArgumentParser(description="match fresh topic signals against the owner's story catalog")
    parser.add_argument("topic")
    parser.add_argument("--window-days", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-confidence", choices=["high", "medium", "low"], default="medium")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with suppress(KeyboardInterrupt):
        asyncio.run(stories(args))


@dataclass
class AgentShellConfig:
    mode: CollectionAgentMode = CollectionAgentMode.auto
    phase: str | None = None
    keywords: list[str] = field(default_factory=list)
    seed_channels: list[str] = field(default_factory=list)
    task_name: str | None = None
    depth: int = 1
    limit: int = 20
    pages_per_channel: int = 1
    crawl: bool = True
    interval_seconds: int = 300
    enabled: bool = True
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    approved_tools: list[str] = field(default_factory=list)
    max_steps: int = 6


def _print_shell_help() -> None:
    print(
        "\n".join(
            [
                "commands:",
                "  /run <goal>                 run agent",
                "  /status                     run status",
                "  /mode <auto|status|discover|ingest|create_task|run_task>",
                "  /phase <understand|plan|edit|verify|fix|finalize|->",
                "  /keyword <text>             add keyword",
                "  /keywords                   show keywords",
                "  /clear-keywords             clear keywords",
                "  /seed <channel>             add seed channel",
                "  /seeds                      show seeds",
                "  /clear-seeds                clear seeds",
                "  /task <name|->              set/clear task name",
                "  /limit <n>",
                "  /depth <n>",
                "  /pages <n>",
                "  /crawl <on|off>",
                "  /steps <n>",
                "  /tool <name> [json-args]      run explicit tool",
                "  /approve <tool>             allow approval-gated tool",
                "  /approvals                  show approved tools",
                "  /clear-approvals            clear approved tools",
                "  /config                     show config",
                "  /help",
                "  /quit",
                "",
                "plain text also runs as goal.",
            ]
        )
    )


def _print_agent_result(result: CollectionAgentResponse) -> None:
    print(f"final: {result.final}")
    for step in result.steps:
        action = step.action
        print(f"[{step.step}] action={action.type} tool={action.tool_name or '-'}")
        print(f"    reason: {action.reason}")
        if action.args:
            print(f"    args: {json.dumps(action.args, ensure_ascii=False)}")
        if step.observation:
            obs = step.observation
            print(f"    observation: ok={obs.ok} {obs.summary}")
            data = obs.data
            if "candidates_found" in data:
                print(
                    "    data: "
                    f"candidates={data.get('candidates_found')} "
                    f"crawled={data.get('channels_crawled')} "
                    f"messages={data.get('messages_saved')}"
                )
            elif "engine_running" in data:
                print(
                    "    data: "
                    f"engine_running={data.get('engine_running')} "
                    f"active_tasks={data.get('active_tasks')} "
                    f"tasks={len(data.get('tasks') or [])}"
                )


def _shell_request(goal: str, config: AgentShellConfig) -> CollectionAgentRequest:
    return CollectionAgentRequest(
        goal=goal,
        mode=config.mode,
        phase=config.phase,
        keywords=config.keywords,
        seed_channels=config.seed_channels,
        task_name=config.task_name,
        depth=config.depth,
        limit=config.limit,
        pages_per_channel=config.pages_per_channel,
        crawl=config.crawl,
        interval_seconds=config.interval_seconds,
        enabled=config.enabled,
        tool_name=config.tool_name,
        tool_args=config.tool_args,
        approved_tools=config.approved_tools,
        max_steps=config.max_steps,
    )


def _print_config(config: AgentShellConfig) -> None:
    print(json.dumps(_shell_request("preview", config).model_dump(mode="json"), ensure_ascii=False, indent=2))


async def agent_shell(args: argparse.Namespace) -> None:
    state = build_state()
    await init_db(state.engine)
    config = AgentShellConfig()
    print("tg-radar agent shell. /help for commands.")
    try:
        while True:
            try:
                line = input("agent> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                break
            if not line:
                continue
            if line in {"/quit", "/exit", "quit", "exit"}:
                break
            if line == "/help":
                _print_shell_help()
                continue
            if line == "/config":
                _print_config(config)
                continue
            if line == "/keywords":
                print(json.dumps(config.keywords, ensure_ascii=False))
                continue
            if line == "/seeds":
                print(json.dumps(config.seed_channels, ensure_ascii=False))
                continue
            if line == "/approvals":
                print(json.dumps(config.approved_tools, ensure_ascii=False))
                continue
            if line == "/clear-keywords":
                config.keywords.clear()
                print("ok")
                continue
            if line == "/clear-seeds":
                config.seed_channels.clear()
                print("ok")
                continue
            if line == "/clear-approvals":
                config.approved_tools.clear()
                print("ok")
                continue
            if line == "/status":
                result = await state.agent.run(_shell_request("status", AgentShellConfig(mode=CollectionAgentMode.status)))
                _print_agent_result(result)
                continue

            command, _, value = line.partition(" ")
            value = value.strip()
            if command == "/mode":
                config.mode = CollectionAgentMode(value)
                print(f"mode={config.mode}")
                continue
            if command == "/phase":
                config.phase = None if value in {"", "-"} else value
                print(f"phase={config.phase or '-'}")
                continue
            if command == "/keyword" and value:
                config.keywords.append(value)
                print("ok")
                continue
            if command == "/seed" and value:
                config.seed_channels.append(value.strip("@"))
                print("ok")
                continue
            if command == "/task":
                config.task_name = None if value in {"", "-"} else value
                print(f"task={config.task_name or '-'}")
                continue
            if command == "/limit":
                config.limit = int(value)
                print(f"limit={config.limit}")
                continue
            if command == "/depth":
                config.depth = int(value)
                print(f"depth={config.depth}")
                continue
            if command == "/pages":
                config.pages_per_channel = int(value)
                print(f"pages={config.pages_per_channel}")
                continue
            if command == "/crawl":
                config.crawl = value.casefold() in CLI_RULES["truthy_values"]
                print(f"crawl={config.crawl}")
                continue
            if command == "/steps":
                config.max_steps = int(value)
                print(f"max_steps={config.max_steps}")
                continue
            if command == "/approve" and value:
                config.approved_tools.append(value)
                print("ok")
                continue
            if command == "/tool" and value:
                tool_name, _, raw_args = value.partition(" ")
                config.tool_name = tool_name
                config.tool_args = json.loads(raw_args or "{}")
                result = await state.agent.run(_shell_request(f"tool {tool_name}", config))
                config.tool_name = None
                config.tool_args = {}
                _print_agent_result(result)
                continue

            goal = value if command == "/run" else line
            result = await state.agent.run(_shell_request(goal, config))
            _print_agent_result(result)
    finally:
        await _close_state(state)


async def _close_state(state) -> None:
    with suppress(asyncio.CancelledError):
        await state.auto_search.stop()
    with suppress(asyncio.CancelledError):
        await state.engine.dispose()


def agent_shell_main() -> None:
    parser = argparse.ArgumentParser()
    with suppress(KeyboardInterrupt):
        asyncio.run(agent_shell(parser.parse_args()))
