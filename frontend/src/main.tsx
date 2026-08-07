import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertCircle,
  Bot,
  Brain,
  Check,
  ChevronDown,
  Clock,
  Database,
  ExternalLink,
  FileText,
  Loader2,
  Play,
  Radar,
  RefreshCcw,
  Search,
  Send,
  Settings,
  Sparkles,
  Square,
  Terminal,
  Wrench,
  X,
} from "lucide-react";
import "./styles.css";

type Mode = "auto" | "status" | "discover" | "ingest";
type ToolMode = Mode | "create_task" | "run_task";
type Phase = "understand" | "plan" | "edit" | "verify" | "fix" | "finalize";
type Role = "user" | "agent" | "system";
type ActiveTab = "agent" | "data" | "writing";

type Candidate = {
  username: string;
  source: string;
  reason: string;
  score: number;
  depth: number;
  task_name?: string;
  state?: string;
  quality_score?: number;
  quality_reasons?: string[];
};
type SearchHit = {
  message_id: string;
  channel_username: string;
  channel_title?: string;
  url: string;
  posted_at?: string;
  text: string;
  score: number;
  highlights?: string[];
  why_matched?: string[];
  score_reasons?: string[];
  cluster_size?: number;
  pain_score?: number;
  pain_type?: string;
};
type PainInsight = {
  message_id: string;
  channel_username: string;
  channel_title?: string;
  url: string;
  posted_at?: string;
  text: string;
  pain_score: number;
  pain_type: string;
  intent: string;
  reasons: string[];
  highlights: string[];
};
type PainReport = { topic: string; total: number; groups: { name: string; count: number }[]; items: PainInsight[] };
type ResearchTopic = {
  slug: string;
  title?: string;
  keywords: string[];
  negative_keywords: string[];
  seed_channels: string[];
  enabled: boolean;
  channels: number;
  messages: number;
  pain_items: number;
};
type SearchTask = {
  name: string;
  keywords: string[];
  seed_channels: string[];
  depth: number;
  limit: number;
  pages_per_channel: number;
  crawl: boolean;
  enabled: boolean;
  running: boolean;
  last_error?: string;
  total_runs: number;
  total_candidates: number;
  total_channels_crawled: number;
  total_messages_saved: number;
  last_candidates_found: number;
  last_channels_crawled: number;
  last_messages_saved: number;
  last_run_at?: string;
  next_run_at?: string;
};
type ContentCard = {
  id: number;
  topic_slug: string;
  card_type: string;
  title: string;
  summary: string;
  body: string;
  source_channel?: string;
  source_url?: string;
  score: number;
  status: string;
  tags: string[];
  payload: Record<string, unknown>;
};
type ArticleDraft = { id: number; version: number; status: string; format: string; text: string; score: number };
type ArticleBlock = { id: number; position: number; block_type: string; title?: string; text: string; status: string };
type ArticleSession = {
  id: number;
  topic_slug: string;
  title: string;
  audience?: string;
  angle?: string;
  status: string;
  source_card_ids: number[];
  blocks: ArticleBlock[];
  drafts: ArticleDraft[];
};
type ContentBoard = { topic_slug: string; cards: ContentCard[]; articles: ArticleSession[] };
type ToolSpec = { name: string; description: string; risk_level?: string; input_schema?: Record<string, unknown>; args_schema?: Record<string, unknown> };
type AgentAction = { type: string; tool_name?: string; args: Record<string, unknown>; reason: string; expected_result?: string };
type AgentObservation = { tool_name: string; ok: boolean; summary: string; data: Record<string, unknown> };
type AgentStep = { step: number; action: AgentAction; observation?: AgentObservation };
type AgentRun = {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "timeout";
  final?: string;
  error?: string;
  started_at: string;
  finished_at?: string;
  steps: AgentStep[];
};
type AgentSyncResponse = { run_id?: string; final: string; steps: AgentStep[]; context?: Record<string, unknown> };
type CoreStatus = {
  engine_running: boolean;
  agent_worker_running?: boolean;
  content_factory_enabled: boolean;
  llm_enabled: boolean;
  last_eval_score?: number | null;
  last_eval_at?: string | null;
};
type TranscriptItem = { id: string; role: Role; text: string; runId?: string; createdAt: number };

const DEFAULT_GOAL =
  "Работаем по готовому топику vibe-coding-ai. Собери Telegram-сигналы: хайп, боли, тренды, конфликты, повторяющиеся тезисы. Подготовь тезис, evidence map, заголовки и черновик статьи. Не выдумывай факты.";

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const token = import.meta.env.VITE_TG_RADAR_API_TOKEN || window.localStorage.getItem("tgRadarApiToken") || "";
  const headers = {
    "content-type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options?.headers || {}),
  };
  const response = await fetch(`/api${path}`, { ...options, headers });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

function splitLines(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim().replace(/^@/, ""))
    .filter(Boolean);
}

function normalizeTopic(value: string): string {
  return value.toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}

function slugifyTopic(value: string): string {
  const latin = value
    .toLowerCase()
    .match(/[a-z0-9][a-z0-9_-]{1,}/g);
  if (latin?.length) return latin.slice(0, 6).join("-").replace(/_+/g, "-").slice(0, 64);
  return value
    .toLowerCase()
    .replace(/[^a-zа-яё0-9]+/gi, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 64) || "ad-hoc-research";
}

function explicitTopic(goal: string): string {
  const match = goal.match(/(?:готов(?:ому|ый)?\s+топик(?:у|ом)?|topic)\s+([a-z0-9][a-z0-9_-]{2,})/i);
  return match ? slugifyTopic(match[1]) : "";
}

function looksLikeResearchGoal(goal: string): boolean {
  return /(telegram|телег|сигнал|аналит|исслед|собер|тренд|боли|pain|hype|draft|article|стать|чернов|outline|evidence|заголов|agent|агент|tui|coding|кодинг|cursor|claude|langgraph)/i.test(goal);
}

function looksLikeWritingGoal(goal: string): boolean {
  return /(стать|article|draft|чернов|outline|evidence map|тезис|заголов|final article|финальн)/i.test(goal);
}

function keywordsFromGoal(goal: string, fallback: string[]): string[] {
  const tokens = Array.from(new Set(goal.match(/[a-zа-яё0-9][a-zа-яё0-9_-]{2,}/gi) || []));
  const stop = new Set([
    "собери", "собрать", "аналитику", "анализ", "лучшему", "среди", "всех", "агентов", "готовому", "топику", "подготовь", "telegram", "телеги", "телеграм",
    "the", "and", "for", "with", "from", "article", "draft", "best", "among", "agents", "research", "analysis",
  ]);
  const filtered = tokens.filter((item) => !stop.has(item.toLowerCase())).slice(0, 8);
  return filtered.length ? filtered : fallback.length ? fallback : [goal.slice(0, 80)];
}

function formatDate(value?: string): string {
  if (!value) return "never";
  return new Intl.DateTimeFormat("ru-RU", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function formatNumber(value?: number): string {
  return new Intl.NumberFormat("ru-RU").format(value ?? 0);
}

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function metric(data: Record<string, unknown> | undefined, key: string): number {
  const value = data?.[key];
  return typeof value === "number" ? value : 0;
}

function listCount(data: Record<string, unknown> | undefined, key: string): number {
  const value = data?.[key];
  return Array.isArray(value) ? value.length : 0;
}

function App() {
  const [topic, setTopic] = useState("vibe-coding-ai");
  const [articleFocus, setArticleFocus] = useState("");
  const [keywordsText, setKeywordsText] = useState("vibe coding AI\nvibe-coding\nAI coding agents\nCursor Claude Code\nвайбкодинг");
  const [seedText, setSeedText] = useState("aostrikov_ai_agents");
  const [mode, setMode] = useState<Mode>("auto");
  const [phase, setPhase] = useState<Phase | "">("");
  const [limit, setLimit] = useState(40);
  const [depth, setDepth] = useState(1);
  const [pages, setPages] = useState(1);
  const [crawl, setCrawl] = useState(true);
  const [maxSteps, setMaxSteps] = useState(8);
  const [approvedTools, setApprovedTools] = useState<string[]>([]);
  const [composer, setComposer] = useState(DEFAULT_GOAL);
  const [activeTab, setActiveTab] = useState<ActiveTab>("agent");

  const [core, setCore] = useState<CoreStatus | null>(null);
  const [tasks, setTasks] = useState<SearchTask[]>([]);
  const [topics, setTopics] = useState<ResearchTopic[]>([]);
  const [tools, setTools] = useState<ToolSpec[]>([]);
  const [activeRunId, setActiveRunId] = useState("");
  const [run, setRun] = useState<AgentRun | null>(null);
  const [transcript, setTranscript] = useState<TranscriptItem[]>([
    { id: "hello", role: "system", text: "Agent UI ready. Use /help or run a goal.", createdAt: Date.now() },
  ]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedChannel, setSelectedChannel] = useState("");
  const [messages, setMessages] = useState<SearchHit[]>([]);
  const [pain, setPain] = useState<PainReport | null>(null);
  const [contentCards, setContentCards] = useState<ContentCard[]>([]);
  const [articles, setArticles] = useState<ArticleSession[]>([]);
  const [selectedCardIds, setSelectedCardIds] = useState<number[]>([]);
  const [selectedArticle, setSelectedArticle] = useState<ArticleSession | null>(null);
  const [articleInstruction, setArticleInstruction] = useState("Рейджбейт, но доказательно: конфликт, боли, цитируемые Telegram-сигналы, вывод без выдуманных фактов.");

  const finalPosted = useRef<Set<string>>(new Set());
  const activeRunTopic = useRef(topic);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const keywords = useMemo(() => splitLines(keywordsText), [keywordsText]);
  const seedChannels = useMemo(() => splitLines(seedText), [seedText]);
  const latestDraft = selectedArticle?.drafts?.[0] || null;

  useEffect(() => {
    refreshCore();
    const id = window.setInterval(refreshCore, 5000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!activeRunId) return;
    let cancelled = false;
    async function tick() {
      const next = await api<AgentRun>(`/core/agent/runs/${encodeURIComponent(activeRunId)}`);
      if (cancelled) return;
      setRun(next);
      if (!["queued", "running"].includes(next.status)) {
        setBusy("");
        setActiveRunId("");
        if (!finalPosted.current.has(next.run_id)) {
          finalPosted.current.add(next.run_id);
          push("agent", next.error || next.final || next.status, next.run_id);
          await refreshEvidence(activeRunTopic.current || topic);
        }
      }
    }
    tick().catch((err) => setError(String(err)));
    const id = window.setInterval(() => tick().catch((err) => setError(String(err))), 900);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [activeRunId]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ block: "end" });
  }, [transcript, run?.steps.length]);

  useEffect(() => {
    if (activeTab === "data") void refreshEvidence();
    if (activeTab === "writing") void loadContent(topic);
  }, [activeTab]);

  useEffect(() => {
    if (core?.content_factory_enabled) void loadContent(topic);
  }, [core?.content_factory_enabled, topic]);

  async function refreshCore() {
    try {
      const status = await api<CoreStatus>("/core/status");
      setCore(status);
      setTasks(await api<SearchTask[]>("/core/tasks"));
      setTopics(await api<ResearchTopic[]>("/core/topics"));
      setTools(await api<ToolSpec[]>("/core/agent/tools"));
    } catch {
      // API can still be starting.
    }
  }

  async function refreshEvidence(topicName = topic) {
    await Promise.allSettled([loadCandidates(topicName), loadPain(topicName), loadContent(topicName), refreshCore()]);
  }

  async function loadCandidates(name: string) {
    if (!name) return;
    const saved = await api<{ candidates: Candidate[] }>(`/core/tasks/${encodeURIComponent(name)}/candidates?limit=200`);
    setCandidates(saved.candidates);
    const first = selectedChannel || saved.candidates[0]?.username || "";
    if (first) await loadChannelMessages(first);
  }

  async function loadChannelMessages(username: string) {
    setSelectedChannel(username);
    const result = await api<{ messages: SearchHit[] }>(`/core/channels/${encodeURIComponent(username)}/messages?limit=50`);
    setMessages(result.messages);
  }

  async function loadPain(topicName: string) {
    try {
      setPain(await api<PainReport>(`/core/topics/${encodeURIComponent(topicName)}/pain?limit=80`));
    } catch {
      setPain(null);
    }
  }

  async function loadContent(topicName: string) {
    if (!core?.content_factory_enabled) {
      setContentCards([]);
      setArticles([]);
      setSelectedArticle(null);
      return;
    }
    const board = await api<ContentBoard>(`/content/board?topic=${encodeURIComponent(topicName)}`);
    setContentCards(board.cards);
    setArticles(board.articles);
    setSelectedArticle((current) => board.articles.find((item) => item.id === current?.id) || board.articles[0] || null);
  }

  function resolveRunScope(goal: string, overrides: Partial<Record<string, unknown>> = {}) {
    const overrideTopic = typeof overrides.topic === "string" ? overrides.topic : "";
    const overrideKeywords = Array.isArray(overrides.keywords) ? overrides.keywords.map(String) : null;
    const modeValue = String(overrides.mode || mode);
    if (modeValue === "status" || overrides.tool_name) {
      return { topicName: overrideTopic || topic, keywordList: overrideKeywords || keywords, seedList: seedChannels, shouldSyncUi: false };
    }
    const explicit = explicitTopic(goal);
    const currentMentioned = normalizeTopic(goal).includes(normalizeTopic(topic));
    if (!looksLikeResearchGoal(goal)) {
      return { topicName: "", keywordList: [], seedList: [], shouldSyncUi: false };
    }
    if (!explicit && currentMentioned) {
      return { topicName: overrideTopic || topic, keywordList: overrideKeywords || keywords, seedList: seedChannels, shouldSyncUi: false };
    }
    const topicName = overrideTopic || explicit || slugifyTopic(goal);
    return {
      topicName,
      keywordList: overrideKeywords || keywordsFromGoal(goal, keywords),
      seedList: seedChannels,
      shouldSyncUi: topicName !== topic,
    };
  }

  function requestPayload(goal: string, overrides: Partial<Record<string, unknown>> = {}) {
    const scope = resolveRunScope(goal, overrides);
    return {
      goal,
      mode,
      phase: phase || null,
      topic: scope.topicName,
      article_focus: articleFocus || (looksLikeWritingGoal(goal) ? goal : null),
      focus_keywords: [],
      keywords: scope.keywordList,
      seed_channels: scope.seedList,
      task_name: null,
      depth,
      limit,
      pages_per_channel: pages,
      crawl,
      approved_tools: approvedTools,
      max_steps: maxSteps,
      ...overrides,
    };
  }

  async function startRun(goal: string, overrides: Partial<Record<string, unknown>> = {}) {
    setError("");
    setBusy("agent");
    push("user", goal);
    const scope = resolveRunScope(goal, overrides);
    if (scope.shouldSyncUi) {
      setTopic(scope.topicName);
      setKeywordsText(scope.keywordList.join("\n"));
      setMessages([]);
      setPain(null);
      setCandidates([]);
    }
    activeRunTopic.current = scope.topicName;
    const payload = requestPayload(goal, overrides);
    try {
      const started = await api<{ run_id: string; status: string }>("/core/agent/runs", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setActiveRunId(started.run_id);
      setRun({
        run_id: started.run_id,
        status: started.status as AgentRun["status"],
        started_at: new Date().toISOString(),
        steps: [],
      });
    } catch (err) {
      const message = String(err);
      if (!message.includes("Not Found") && !message.includes("404")) throw err;
      const fallback = await api<AgentSyncResponse>("/core/agent/run", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const runId = fallback.run_id || `sync-${Date.now()}`;
      setRun({
        run_id: runId,
        status: "completed",
        final: fallback.final,
        started_at: new Date().toISOString(),
        finished_at: new Date().toISOString(),
        steps: fallback.steps,
      });
      setActiveRunId("");
      push("agent", fallback.final, runId);
      await refreshEvidence(scope.topicName);
    } finally {
      setBusy("");
    }
  }

  function push(role: Role, text: string, runId?: string) {
    setTranscript((items) => [...items, { id: `${Date.now()}-${Math.random()}`, role, text, runId, createdAt: Date.now() }]);
  }

  async function handleSubmit(event?: FormEvent) {
    event?.preventDefault();
    const line = composer.trim();
    if (!line) return;
    setComposer("");
    try {
      await handleCommand(line);
    } catch (err) {
      setBusy("");
      setError(String(err));
      push("system", String(err));
    }
  }

  async function handleCommand(line: string) {
    const [command, ...restParts] = line.split(" ");
    const rest = restParts.join(" ").trim();
    if (command === "/help") return push("system", helpText());
    if (command === "/config") return push("system", pretty(requestPayload("preview")));
    if (command === "/keywords") return push("system", keywords.join("\n") || "no keywords");
    if (command === "/seeds") return push("system", seedChannels.map((seed) => `@${seed}`).join("\n") || "no seeds");
    if (command === "/approvals") return push("system", approvedTools.join("\n") || "no approvals");
    if (command === "/clear-keywords") return setKeywordsText("");
    if (command === "/clear-seeds") return setSeedText("");
    if (command === "/clear-approvals") return setApprovedTools([]);
    if (command === "/keyword" && rest) return setKeywordsText((value) => [value, rest].filter(Boolean).join("\n"));
    if (command === "/seed" && rest) return setSeedText((value) => [value, rest.replace(/^@/, "")].filter(Boolean).join("\n"));
    if (command === "/approve" && rest) return setApprovedTools((items) => Array.from(new Set([...items, rest])));
    if (command === "/mode") return setMode((rest || "auto") as Mode);
    if (command === "/phase") return setPhase(rest === "-" ? "" : (rest as Phase));
    if (command === "/topic" && rest) return setTopic(slugifyTopic(rest));
    if (command === "/focus") return setArticleFocus(rest);
    if (command === "/limit") return setLimit(Number(rest || limit));
    if (command === "/depth") return setDepth(Number(rest || depth));
    if (command === "/pages") return setPages(Number(rest || pages));
    if (command === "/steps") return setMaxSteps(Number(rest || maxSteps));
    if (command === "/crawl") return setCrawl(["on", "true", "1", "yes"].includes(rest.toLowerCase()));
    if (command === "/status") return startRun("status", { mode: "status" });
    if (command === "/run") return startRun(rest || DEFAULT_GOAL);
    if (command === "/tool" && rest) {
      const [toolName, ...rawParts] = rest.split(" ");
      const raw = rawParts.join(" ").trim();
      return startRun(`tool ${toolName}`, { tool_name: toolName, tool_args: raw ? JSON.parse(raw) : {} });
    }
    return startRun(line);
  }

  async function saveTopic() {
    await api<ResearchTopic>("/core/topics", {
      method: "POST",
      body: JSON.stringify({ slug: topic, title: topic, keywords, negative_keywords: [], seed_channels: seedChannels, enabled: true }),
    });
    await refreshCore();
  }

  async function startEngine() {
    setBusy("engine");
    try {
      await saveTopic();
      await api("/core/engine/start", { method: "POST" });
      await refreshCore();
    } finally {
      setBusy("");
    }
  }

  async function stopEngine() {
    setBusy("engine");
    try {
      await api("/core/engine/stop", { method: "POST" });
      await refreshCore();
    } finally {
      setBusy("");
    }
  }

  async function cancelRun() {
    if (!run) return;
    setBusy("agent");
    try {
      const next = await api<AgentRun>(`/core/agent/runs/${encodeURIComponent(run.run_id)}/cancel`, { method: "POST" });
      setRun(next);
      setActiveRunId("");
    } finally {
      setBusy("");
    }
  }

  async function retryRun() {
    if (!run) return;
    setBusy("agent");
    try {
      const started = await api<{ run_id: string; status: string }>(`/core/agent/runs/${encodeURIComponent(run.run_id)}/retry`, { method: "POST" });
      setRun({ ...run, status: started.status as AgentRun["status"], final: undefined, error: undefined, finished_at: undefined, steps: [] });
      setActiveRunId(started.run_id);
    } finally {
      setBusy("");
    }
  }

  async function continueRun() {
    const goal = composer.trim() || run?.final || DEFAULT_GOAL;
    await startRun(goal, { phase: phase || "edit", topic: activeRunTopic.current || topic });
  }

  async function quick(modeValue: ToolMode) {
    const goal = modeValue === "status" ? "status" : composer.trim() || DEFAULT_GOAL;
    await startRun(goal, { mode: modeValue });
  }

  async function generateCards() {
    setBusy("cards");
    try {
      const cards = await api<ContentCard[]>("/content/cards/generate", {
        method: "POST",
        body: JSON.stringify({ topic_slug: topic, limit: 80, include_llm: false }),
      });
      setContentCards(cards);
    } finally {
      setBusy("");
    }
  }

  async function createArticle() {
    setBusy("article");
    try {
      const article = await api<ArticleSession>("/content/articles", {
        method: "POST",
        body: JSON.stringify({
          topic_slug: topic,
          title: `${topic}: draft`,
          audience: "developers, tech leads, founders in AI/LLM/vibe coding",
          angle: articleInstruction,
          card_ids: selectedCardIds,
        }),
      });
      setSelectedArticle(article);
      await loadContent(topic);
    } finally {
      setBusy("");
    }
  }

  async function generateOutline() {
    if (!selectedArticle) return;
    setBusy("outline");
    try {
      const article = await api<ArticleSession>(`/content/articles/${selectedArticle.id}/outline`, {
        method: "POST",
        body: JSON.stringify({ style: "provocative_engineering", instruction: articleInstruction, use_llm: true }),
      });
      setSelectedArticle(article);
      await loadContent(topic);
    } finally {
      setBusy("");
    }
  }

  async function generateDraft() {
    if (!selectedArticle) return;
    setBusy("draft");
    try {
      await api<ArticleDraft>(`/content/articles/${selectedArticle.id}/draft`, {
        method: "POST",
        body: JSON.stringify({ style: "provocative_engineering", instruction: articleInstruction, use_llm: true }),
      });
      const article = await api<ArticleSession>(`/content/articles/${selectedArticle.id}`);
      setSelectedArticle(article);
      await loadContent(topic);
    } finally {
      setBusy("");
    }
  }

  async function formatTelegram() {
    if (!selectedArticle) return;
    setBusy("telegram");
    try {
      await api<ArticleDraft>(`/content/articles/${selectedArticle.id}/format-telegram`, { method: "POST" });
      const article = await api<ArticleSession>(`/content/articles/${selectedArticle.id}`);
      setSelectedArticle(article);
      await loadContent(topic);
    } finally {
      setBusy("");
    }
  }

  const selectedStep = run?.steps[run.steps.length - 1] || null;
  const runStatus = run?.status || "idle";

  return (
    <main className="agent-shell">
      <aside className="sidebar">
        <div className="brand">
          <Radar size={24} />
          <div>
            <h1>TG Radar</h1>
            <span>agent workspace</span>
          </div>
        </div>

        <StatusPill
          running={core?.engine_running || false}
          agentWorker={core?.agent_worker_running || false}
          llm={core?.llm_enabled || false}
          factory={core?.content_factory_enabled || false}
          evalScore={core?.last_eval_score}
        />

        <section className="side-section">
          <h2><Settings size={15} /> Context</h2>
          <label><span>Topic</span><input value={topic} onChange={(event) => setTopic(event.target.value)} /></label>
          <label><span>Article focus</span><textarea value={articleFocus} onChange={(event) => setArticleFocus(event.target.value)} placeholder="specific article angle inside the topic corpus" /></label>
          <label><span>Keywords</span><textarea value={keywordsText} onChange={(event) => setKeywordsText(event.target.value)} /></label>
          <label><span>Seeds</span><textarea value={seedText} onChange={(event) => setSeedText(event.target.value)} /></label>
          <div className="control-grid">
            <label><span>Mode</span><select value={mode} onChange={(event) => setMode(event.target.value as Mode)}>{["auto", "status", "discover", "ingest"].map((item) => <option key={item}>{item}</option>)}</select></label>
            <label><span>Phase</span><select value={phase} onChange={(event) => setPhase(event.target.value as Phase | "")}><option value="">auto</option>{["understand", "plan", "edit", "verify", "fix", "finalize"].map((item) => <option key={item}>{item}</option>)}</select></label>
            <NumberField label="Limit" value={limit} setValue={setLimit} />
            <NumberField label="Depth" value={depth} setValue={setDepth} />
            <NumberField label="Pages" value={pages} setValue={setPages} />
            <NumberField label="Steps" value={maxSteps} setValue={setMaxSteps} />
          </div>
          <label className="toggle"><input type="checkbox" checked={crawl} onChange={(event) => setCrawl(event.target.checked)} /> crawl posts</label>
          <div className="button-row compact-actions">
            <button onClick={startEngine} disabled={!!busy}><Play size={15} /> Start loop</button>
            <button onClick={stopEngine} disabled={!!busy}><Square size={15} /> Stop</button>
          </div>
        </section>

        <section className="side-section compact">
          <h2><Database size={15} /> Topics</h2>
          <div className="pill-list">
            {topics.slice(0, 8).map((item) => (
              <button key={item.slug} className={item.slug === topic ? "pill selected" : "pill"} onClick={() => {
                setTopic(item.slug);
                setKeywordsText(item.keywords.join("\n"));
                setSeedText(item.seed_channels.join("\n"));
                void refreshEvidence(item.slug);
              }}>
                <b>{item.slug}</b>
                <span>{formatNumber(item.channels)} ch · {formatNumber(item.messages)} msg · {formatNumber(item.pain_items)} pain</span>
              </button>
            ))}
          </div>
        </section>
      </aside>

      <section className="workspace">
        <header className="workspace-top">
          <div>
            <h2>{activeTab === "agent" ? "Agent" : activeTab === "data" ? "Data" : "Writing"}</h2>
            <p>{activeRunId || "no active run"} · {runStatus}</p>
          </div>
          <div className="segment-nav" role="tablist" aria-label="Workspace sections">
            <button className={activeTab === "agent" ? "active" : ""} onClick={() => setActiveTab("agent")} type="button"><Bot size={16} /> Agent</button>
            <button className={activeTab === "data" ? "active" : ""} onClick={() => setActiveTab("data")} type="button"><Database size={16} /> Data</button>
            <button className={activeTab === "writing" ? "active" : ""} onClick={() => setActiveTab("writing")} type="button"><FileText size={16} /> Writing</button>
          </div>
        </header>

        {error && <div className="error-line"><AlertCircle size={16} />{error}</div>}

        {activeTab === "agent" && (
          <section className="agent-grid">
            <div className="transcript-panel">
              <div className="panel-toolbar">
                <button onClick={() => quick("status")} disabled={!!busy}><Activity size={16} /> Status</button>
                <button onClick={() => setComposer(helpText())} type="button"><Terminal size={16} /> Commands</button>
              </div>
              {latestDraft && selectedArticle && (
                <LatestDraftCard
                  article={selectedArticle}
                  draft={latestDraft}
                  onOpen={() => {
                    setActiveTab("writing");
                    void loadContent(topic);
                  }}
                />
              )}
              <div className="transcript">
                {transcript.map((item) => <Transcript key={item.id} item={item} />)}
                {run && <RunCard run={run} onCancel={cancelRun} onRetry={retryRun} onContinue={continueRun} onOpenWriting={() => {
                  setActiveTab("writing");
                  void loadContent(topic);
                }} />}
                <div ref={transcriptEndRef} />
              </div>
              <form className="composer" onSubmit={handleSubmit}>
                <Terminal size={18} />
                <textarea value={composer} onChange={(event) => setComposer(event.target.value)} onKeyDown={(event) => {
                  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) void handleSubmit();
                }} />
                <button type="submit" disabled={!!busy}>{busy === "agent" ? <Loader2 className="spin" size={18} /> : <Send size={18} />} Run</button>
              </form>
            </div>

            <aside className="inspector">
              <PanelTitle icon={<Wrench size={16} />} title="Steps" count={run?.steps.length || 0} />
              <div className="step-list">
                {(run?.steps || []).map((step) => <StepRow key={`${run?.run_id || "run"}-${step.step}`} step={step} />)}
                {!run?.steps.length && <Empty text="Run /status to see tool calls" />}
              </div>
              <details className="json-box">
                <summary><ChevronDown size={14} /> Current payload</summary>
                <pre>{pretty(requestPayload("preview"))}</pre>
              </details>
              <details className="json-box">
                <summary><ChevronDown size={14} /> Last action</summary>
                <pre>{selectedStep ? pretty(selectedStep) : "no step"}</pre>
              </details>
              <details className="json-box">
                <summary><ChevronDown size={14} /> Tools</summary>
                <pre>{pretty(tools.map((tool) => ({ name: tool.name, risk: tool.risk_level, description: tool.description })))}</pre>
              </details>
            </aside>
          </section>
        )}

        {activeTab === "data" && (
          <section className="data-panel">
            <div className="panel-toolbar">
              <button onClick={() => quick("discover")} disabled={!!busy}><Search size={16} /> Discover</button>
              <button onClick={() => quick("ingest")} disabled={!!busy}><Database size={16} /> Ingest</button>
              <button onClick={() => refreshEvidence()} disabled={!!busy}><RefreshCcw size={16} /> Refresh</button>
            </div>
            <section className="evidence-grid">
              <EvidenceColumn title="Channels" count={candidates.length}>
                {candidates.map((candidate) => (
                  <button className={candidate.username === selectedChannel ? "candidate-line selected" : "candidate-line"} key={candidate.username} onClick={() => loadChannelMessages(candidate.username)}>
                    <b>@{candidate.username}</b>
                    <span>{candidate.state || "raw"} · q {Math.round((candidate.quality_score || 0) * 100)} · {candidate.source}</span>
                    <small>{candidate.quality_reasons?.slice(0, 3).join(" · ") || candidate.reason}</small>
                  </button>
                ))}
              </EvidenceColumn>

              <EvidenceColumn title={selectedChannel ? `@${selectedChannel}` : "Messages"} count={messages.length}>
                {messages.map((hit) => <MessageRow key={hit.message_id} hit={hit} />)}
              </EvidenceColumn>

              <EvidenceColumn title="Pain" count={pain?.total || 0}>
                <div className="tag-row">{(pain?.groups || []).slice(0, 6).map((group) => <span key={group.name}>{group.name} {group.count}</span>)}</div>
                {(pain?.items || []).map((item) => <PainRow key={item.message_id} item={item} />)}
              </EvidenceColumn>
            </section>
          </section>
        )}

        {activeTab === "writing" && (
          <section className="content-panel">
          <div className="content-head">
            <div>
              <h2><FileText size={17} /> Content Factory</h2>
              <p>{core?.content_factory_enabled ? "cards -> article -> outline -> draft" : "content factory disabled"}</p>
            </div>
            <div className="quick-actions">
              <button onClick={generateCards} disabled={!core?.content_factory_enabled || !!busy}><Sparkles size={16} /> Cards</button>
              <button onClick={createArticle} disabled={!core?.content_factory_enabled || selectedCardIds.length === 0 || !!busy}><FileText size={16} /> Article</button>
              <button onClick={generateOutline} disabled={!selectedArticle || !!busy}><Brain size={16} /> Outline</button>
              <button onClick={generateDraft} disabled={!selectedArticle || !!busy}><Bot size={16} /> Draft</button>
              <button onClick={formatTelegram} disabled={!latestDraft || !!busy}><Check size={16} /> Telegram</button>
            </div>
          </div>
          <textarea className="article-prompt" value={articleInstruction} onChange={(event) => setArticleInstruction(event.target.value)} />
          {core?.content_factory_enabled ? (
            <div className="content-grid">
              <div className="content-list">
                <PanelTitle icon={<Sparkles size={16} />} title="Cards" count={contentCards.length} />
                {contentCards.slice(0, 80).map((card) => (
                  <article className={selectedCardIds.includes(card.id) ? "content-card selected" : "content-card"} key={card.id}>
                    <button onClick={() => setSelectedCardIds((ids) => ids.includes(card.id) ? ids.filter((id) => id !== card.id) : [...ids, card.id])}>
                      <b>{card.title}</b>
                      <span>{card.card_type} · {card.status} · {Math.round(card.score * 100)}</span>
                    </button>
                    <p>{card.summary}</p>
                    {card.source_url && <a href={card.source_url} target="_blank" rel="noreferrer">source <ExternalLink size={13} /></a>}
                  </article>
                ))}
              </div>
              <div className="content-list">
                <PanelTitle icon={<FileText size={16} />} title="Articles" count={articles.length} />
                {articles.map((article) => (
                  <button className={article.id === selectedArticle?.id ? "article-line selected" : "article-line"} key={article.id} onClick={() => setSelectedArticle(article)}>
                    <b>{article.title}</b>
                    <span>{article.status} · {article.blocks.length} blocks · {article.drafts.length} drafts</span>
                  </button>
                ))}
                {selectedArticle?.blocks.map((block) => (
                  <article className="block-line" key={block.id}>
                    <span>{block.position + 1}. {block.block_type}</span>
                    <b>{block.title || "Untitled"}</b>
                    <p>{block.text}</p>
                  </article>
                ))}
              </div>
              <div className="draft-preview">
                <PanelTitle icon={<Bot size={16} />} title="Draft" count={latestDraft?.version || 0} />
                {latestDraft ? <textarea value={latestDraft.text} readOnly /> : <Empty text="Draft appears here" />}
              </div>
            </div>
          ) : (
            <Empty text="Set TG_RADAR_CONTENT_FACTORY_ENABLED=true to enable article workflow" />
          )}
          </section>
        )}
      </section>
    </main>
  );
}

function NumberField({ label, value, setValue }: { label: string; value: number; setValue: (value: number) => void }) {
  return <label><span>{label}</span><input type="number" value={value} onChange={(event) => setValue(Number(event.target.value))} /></label>;
}

function StatusPill({ running, agentWorker, llm, factory, evalScore }: { running: boolean; agentWorker: boolean; llm: boolean; factory: boolean; evalScore?: number | null }) {
  return (
    <div className="status-stack">
      <span className={running ? "ok" : ""}>engine {running ? "running" : "stopped"}</span>
      <span className={agentWorker ? "ok" : ""}>runs {agentWorker ? "worker" : "no worker"}</span>
      <span className={llm ? "ok" : ""}>llm {llm ? "ready" : "off"}</span>
      <span className={factory ? "ok" : ""}>factory {factory ? "on" : "off"}</span>
      <span className={typeof evalScore === "number" ? "ok" : ""}>eval {typeof evalScore === "number" ? Math.round(evalScore * 100) : "none"}</span>
    </div>
  );
}

function Transcript({ item }: { item: TranscriptItem }) {
  return (
    <article className={`transcript-item ${item.role}`}>
      <span>{item.role}</span>
      <p>{item.text}</p>
    </article>
  );
}

function RunCard({ run, onCancel, onRetry, onContinue, onOpenWriting }: { run: AgentRun; onCancel: () => void; onRetry: () => void; onContinue: () => void; onOpenWriting: () => void }) {
  const last = run.steps[run.steps.length - 1];
  const result = articleResult(run);
  const failed = failedTool(run);
  const weak = weakEvidence(run);
  const artifacts = artifactSummary(run);
  return (
    <article className={`run-card ${run.status}`}>
      <div>
        <b>{run.status}</b>
        <span>{run.run_id}</span>
      </div>
      <p>{last ? `${last.action.type} ${last.action.tool_name || ""}: ${last.action.reason}` : "waiting for first action"}</p>
      {weak && <p className="run-warning">weak evidence: {weak}</p>}
      {failed && <p className="run-warning">failed tool: {failed}</p>}
      {artifacts.length > 0 && <div className="tag-row">{artifacts.map((item) => <span key={item}>{item}</span>)}</div>}
      <div className="button-row compact-actions">
        {["queued", "running"].includes(run.status) && <button type="button" onClick={onCancel}><X size={14} /> Cancel</button>}
        {["failed", "cancelled", "timeout"].includes(run.status) && <button type="button" onClick={onRetry}><RefreshCcw size={14} /> Retry</button>}
        {artifacts.length > 0 && <button type="button" onClick={onContinue}><Play size={14} /> Continue</button>}
        {result && <button type="button" onClick={onOpenWriting}><FileText size={14} /> Open draft</button>}
      </div>
      {result && (
        <section className="result-card">
          <div>
            <b>Result</b>
            <span>article {result.articleId} · draft {result.draftId}</span>
          </div>
          <p>{result.preview}</p>
          <button type="button" onClick={onOpenWriting}><FileText size={14} /> Open draft</button>
        </section>
      )}
    </article>
  );
}

function failedTool(run: AgentRun): string {
  const step = [...run.steps].reverse().find((item) => item.observation && !item.observation.ok);
  if (step?.observation) return `${step.observation.tool_name}: ${step.observation.summary}`;
  return run.error || "";
}

function weakEvidence(run: AgentRun): string {
  const step = [...run.steps].reverse().find((item) => item.observation?.tool_name === "review_evidence" || item.observation?.tool_name === "verify_claims");
  const data = step?.observation?.data;
  if (!data) return "";
  const reason = data.weak_evidence_reason || data.reason || data.failure_reason;
  if (typeof reason === "string") return reason;
  const usable = typeof data.usable === "number" ? data.usable : undefined;
  const min = typeof data.min_required === "number" ? data.min_required : undefined;
  if (usable !== undefined && min !== undefined && usable < min) return `usable ${usable} below min ${min}`;
  return "";
}

function artifactSummary(run: AgentRun): string[] {
  const out = new Set<string>();
  for (const step of run.steps) {
    const data = step.observation?.data;
    if (!data) continue;
    if (data.evidence_map_card) out.add(`evidence map ${data.evidence_map_card}`);
    if (data.outline_blocks) out.add(`outline ${data.outline_blocks} blocks`);
    if (data.article_id) out.add(`article ${data.article_id}`);
    if (data.draft_id) out.add(`draft ${data.draft_id}`);
    if (data.freshness_days) out.add(`fresh ${data.freshness_days}d`);
    if (data.newest_message_at) out.add(`newest ${formatDate(String(data.newest_message_at))}`);
    if (data.source_count) out.add(`${data.source_count} sources`);
    if (data.message_count) out.add(`${data.message_count} messages`);
  }
  return Array.from(out).slice(0, 8);
}

function LatestDraftCard({ article, draft, onOpen }: { article: ArticleSession; draft: ArticleDraft; onOpen: () => void }) {
  return (
    <section className="latest-draft">
      <div>
        <b>Latest draft</b>
        <span>article {article.id} · draft {draft.id}</span>
      </div>
      <strong>{article.title}</strong>
      <p>{draft.text.slice(0, 900)}</p>
      <button type="button" onClick={onOpen}><FileText size={14} /> Open draft</button>
    </section>
  );
}

function articleResult(run: AgentRun): { articleId: string; draftId: string; preview: string } | null {
  const step = [...run.steps].reverse().find((item) => ["final_article", "write_draft", "write_article"].includes(item.observation?.tool_name || ""));
  const data = step?.observation?.data;
  const preview = typeof data?.final_preview === "string" ? data.final_preview.trim() : typeof data?.draft_preview === "string" ? data.draft_preview.trim() : "";
  if (!data || !preview) return null;
  return {
    articleId: String(data.article_id || "-"),
    draftId: String(data.draft_id || "-"),
    preview,
  };
}

function StepRow({ step }: { step: AgentStep }) {
  const data = step.observation?.data;
  const errors = Array.isArray(data?.errors) ? data.errors : [];
  const line = stepMetricLine(step.observation?.tool_name, data);
  return (
    <article className="step-row">
      <div className="step-head">
        <span>#{step.step}</span>
        <b>{step.action.type}</b>
        <em>{step.action.tool_name || "-"}</em>
      </div>
      <p>{step.action.reason}</p>
      {step.observation && (
        <div className={step.observation.ok ? "observation ok" : "observation fail"}>
          <b>{step.observation.summary}</b>
          <span>{line}</span>
        </div>
      )}
      <details>
        <summary>args</summary>
        <pre>{pretty(step.action.args)}</pre>
      </details>
      {errors.length > 0 && (
        <details>
          <summary>errors</summary>
          <pre>{pretty(errors)}</pre>
        </details>
      )}
    </article>
  );
}

function stepMetricLine(toolName?: string, data?: Record<string, unknown>): string {
  if (toolName === "write_article") {
    return `usable ${metric(data, "usable_evidence")} · rejected ${metric(data, "rejected_evidence")} · cards ${metric(data, "cards")} · draft ${metric(data, "draft_chars")} chars`;
  }
  if (toolName === "review_evidence") {
    return `usable ${metric(data, "usable")} · rejected ${metric(data, "rejected")} · min ${metric(data, "min_required")} · focus ${String(data?.focus_key || "topic")}`;
  }
  if (toolName === "infer_signal_taxonomy") {
    return `taxonomy ${listCount(data, "taxonomy")} · usable ${metric(data, "usable")} · rejected ${metric(data, "rejected")} · card ${metric(data, "taxonomy_card")}`;
  }
  if (toolName === "cluster_signals") {
    return `signals ${listCount(data, "signals")} · usable ${metric(data, "usable")} · rejected ${metric(data, "rejected")}`;
  }
  if (toolName === "build_evidence_map") {
    return `claims ${listCount(data, "claims")} · signals ${listCount(data, "signals")} · card ${metric(data, "evidence_map_card")}`;
  }
  if (toolName === "propose_angles") {
    return `angles ${listCount(data, "angles")} · card ${metric(data, "angle_card")}`;
  }
  if (toolName === "build_outline") {
    return `article ${metric(data, "article_id")} · blocks ${metric(data, "outline_blocks")}`;
  }
  if (toolName === "write_draft") {
    return `article ${metric(data, "article_id")} · draft ${metric(data, "draft_id")} · ${metric(data, "draft_chars")} chars`;
  }
  if (toolName === "verify_claims") {
    return `checks ${listCount(data, "checks")} · unsupported ${metric(data, "unsupported_claims")} · missing refs ${metric(data, "missing_draft_refs")} · focus mismatch ${metric(data, "focus_mismatches")} · weak ${metric(data, "weak_generalizations")}`;
  }
  if (toolName === "final_article") {
    return `final ${String(Boolean(data?.final))} · article ${metric(data, "article_id")} · ${metric(data, "final_chars")} chars`;
  }
  if (data?.freshness_days || data?.newest_message_at || data?.source_count) {
    return `fresh ${String(data?.freshness_days || "-")}d · newest ${formatDate(String(data?.newest_message_at || ""))} · sources ${metric(data, "source_count")} · messages ${metric(data, "message_count")}`;
  }
  return `candidates ${metric(data, "candidates_found")} · crawled ${metric(data, "channels_crawled")} · messages ${metric(data, "messages_saved")} · cached ${metric(data, "cached_messages_used")}`;
}

function PanelTitle({ icon, title, count }: { icon?: React.ReactNode; title: string; count: number }) {
  return <div className="panel-title">{icon}<h2>{title}</h2><span>{formatNumber(count)}</span></div>;
}

function EvidenceColumn({ title, count, children }: { title: string; count: number; children: React.ReactNode }) {
  return (
    <section className="evidence-column">
      <PanelTitle title={title} count={count} />
      <div className="evidence-scroll">{children}</div>
    </section>
  );
}

function MessageRow({ hit }: { hit: SearchHit }) {
  return (
    <article className="message-line">
      <div><b>@{hit.channel_username}</b><span>{formatDate(hit.posted_at)}</span><a href={hit.url} target="_blank" rel="noreferrer"><ExternalLink size={14} /></a></div>
      <p>{hit.text}</p>
      <div className="tag-row">
        {(hit.why_matched || hit.score_reasons || []).slice(0, 4).map((item) => <span key={item}>{item}</span>)}
        {(hit.pain_score || 0) > 0 && <span>{hit.pain_type} {Math.round((hit.pain_score || 0) * 100)}</span>}
      </div>
    </article>
  );
}

function PainRow({ item }: { item: PainInsight }) {
  return (
    <article className="message-line pain">
      <div><b>@{item.channel_username}</b><span>{formatDate(item.posted_at)}</span><a href={item.url} target="_blank" rel="noreferrer"><ExternalLink size={14} /></a></div>
      <p>{item.text}</p>
      <div className="tag-row"><span>{item.pain_type}</span><span>{item.intent}</span><span>{Math.round(item.pain_score * 100)}</span></div>
    </article>
  );
}

function Empty({ text }: { text: string }) {
  return <div className="empty">{text}</div>;
}

function helpText() {
  return [
    "/run <goal>",
    "/status",
    "/keyword <text> | /clear-keywords | /keywords",
    "/seed <channel> | /clear-seeds | /seeds",
    "/focus <article focus>",
    "/mode <auto|status|discover|ingest>",
    "/phase <understand|plan|edit|verify|fix|finalize|->",
    "/limit <n> | /depth <n> | /pages <n> | /steps <n> | /crawl <on|off>",
    "/tool <name> <json>",
    "/approve <tool> | /approvals | /clear-approvals",
    "/config",
  ].join("\n");
}

createRoot(document.getElementById("root")!).render(<App />);
