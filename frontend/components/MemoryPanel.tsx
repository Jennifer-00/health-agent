"use client";

import { useEffect, useState } from "react";
import type { ToolEvent } from "@/lib/useAgentChat";
import {
  fetchMemories, deleteMemory, downloadReportPdf, importMemory, fetchProfile,
} from "@/lib/api";
import type { MemoryItem, ProfileItem } from "@/lib/api";

interface Props {
  events: ToolEvent[];
  toolResultMap: Record<string, string>;
}

type Tab = "events" | "profile" | "episodic" | "import";

const TOOL_LABELS: Record<string, string> = {
  search_memory:   "记忆检索",
  search_rag:      "知识库",
  web_search:      "联网搜索",
  generate_report: "健康报告",
};

// ── 健康档案（Neo4j）配色 ─────────────────────────────────────────────────────
const PROFILE_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  诊断:  { bg: "bg-indigo-50",  text: "text-indigo-700",  border: "border-indigo-200" },
  用药:  { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
  过敏:  { bg: "bg-rose-50",    text: "text-rose-700",    border: "border-rose-200" },
  病史:  { bg: "bg-amber-50",   text: "text-amber-700",   border: "border-amber-200" },
  家族史: { bg: "bg-violet-50", text: "text-violet-700",  border: "border-violet-200" },
};

const PROFILE_CATEGORY_ORDER = ["诊断", "用药", "过敏", "病史", "家族史"];

// ── 语义记忆（Qdrant）category 配色 ──────────────────────────────────────────
const EPISODIC_TAG_STYLES: Record<string, string> = {
  症状: "bg-rose-100 text-rose-700",
  检查: "bg-blue-100 text-blue-700",
  生活: "bg-teal-100 text-teal-700",
};

export default function MemoryPanel({ events, toolResultMap }: Props) {
  const [tab, setTab] = useState<Tab>("events");

  // 健康档案（Neo4j）
  const [profile, setProfile]       = useState<ProfileItem[]>([]);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError]     = useState<string | null>(null);

  // 语义记忆（Qdrant / mem0）
  const [memories, setMemories]     = useState<MemoryItem[]>([]);
  const [memLoading, setMemLoading] = useState(false);
  const [memError, setMemError]     = useState<string | null>(null);

  // 下载报告
  const [downloading, setDownloading] = useState(false);

  // 导入
  const [importText, setImportText] = useState("");
  const [importing, setImporting]   = useState(false);
  const [importMsg, setImportMsg]   = useState<{ ok: boolean; text: string } | null>(null);

  // 工具调用展开
  const [expandedTool, setExpandedTool] = useState<string | null>(null);

  // 组件挂载时立即后台预取，点 tab 时数据已就绪
  useEffect(() => {
    setProfileLoading(true);
    fetchProfile()
      .then(setProfile)
      .catch((e: Error) => setProfileError(e.message ?? "加载失败"))
      .finally(() => setProfileLoading(false));

    setMemLoading(true);
    fetchMemories()
      .then(setMemories)
      .catch((e: Error) => setMemError(e.message ?? "加载失败"))
      .finally(() => setMemLoading(false));
  }, []);

  function refreshProfile() {
    setProfileLoading(true);
    setProfileError(null);
    fetchProfile()
      .then(setProfile)
      .catch((e: Error) => setProfileError(e.message ?? "加载失败"))
      .finally(() => setProfileLoading(false));
  }

  function refreshMemories() {
    setMemLoading(true);
    setMemError(null);
    fetchMemories()
      .then(setMemories)
      .catch((e: Error) => setMemError(e.message ?? "加载失败"))
      .finally(() => setMemLoading(false));
  }

  async function handleDelete(item: MemoryItem) {
    await deleteMemory(item.id);
    setMemories((prev) => prev.filter((m) => m.id !== item.id));
  }

  async function handleImport() {
    if (!importText.trim()) return;
    setImporting(true);
    setImportMsg(null);
    try {
      await importMemory(importText.trim());
      setImportMsg({ ok: true, text: "已提交，正在后台写入记忆库（约10-30秒）" });
      setImportText("");
    } catch {
      setImportMsg({ ok: false, text: "导入失败，请检查后端是否运行" });
    } finally {
      setImporting(false);
    }
  }

  // 按 category 分组 profile
  const profileGroups = PROFILE_CATEGORY_ORDER.reduce<Record<string, ProfileItem[]>>(
    (acc, cat) => {
      const items = profile.filter((p) => p.category === cat);
      if (items.length) acc[cat] = items;
      return acc;
    },
    {}
  );

  const tabs: { id: Tab; label: string }[] = [
    { id: "events",   label: "工具调用" },
    { id: "profile",  label: "健康档案" },
    { id: "episodic", label: "语义记忆" },
    { id: "import",   label: "导入记忆" },
  ];

  return (
    <div className="flex h-full flex-col bg-white/80">
      {/* 标签切换 */}
      <div className="flex border-b overflow-x-auto">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 whitespace-nowrap py-2 text-xs font-medium transition-colors ${
              tab === t.id
                ? "border-b-2 border-emerald-500 text-emerald-700"
                : "text-slate-400 hover:text-slate-600"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4">

        {/* ── 工具调用 ────────────────────────────────────────── */}
        {tab === "events" && (
          <>
            {events.length === 0 && (
              <p className="text-xs text-slate-400">当前会话还没有触发工具调用。</p>
            )}
            <ul className="space-y-2">
              {events.map((ev, i) => {
                const result = toolResultMap[ev.tool];
                const isOpen = expandedTool === `${ev.tool}-${i}`;
                return (
                  <li key={i} className="rounded-xl border border-emerald-100 bg-emerald-50 text-xs overflow-hidden">
                    <div className="flex items-start gap-2 p-3">
                      <div className="flex-1 min-w-0">
                        <span className="font-semibold text-emerald-700">
                          {TOOL_LABELS[ev.tool] ?? ev.tool}
                        </span>
                        <p className="mt-0.5 text-slate-500 truncate">{ev.summary}</p>
                      </div>
                      {result && (
                        <button
                          onClick={() => setExpandedTool(isOpen ? null : `${ev.tool}-${i}`)}
                          className="shrink-0 rounded-md bg-emerald-100 px-2 py-0.5 text-[10px] font-medium text-emerald-600 hover:bg-emerald-200 transition-colors"
                        >
                          {isOpen ? "收起" : "查看结果"}
                        </button>
                      )}
                    </div>
                    {isOpen && result && (
                      <div className="border-t border-emerald-100 bg-white px-3 py-2">
                        <p className="whitespace-pre-wrap leading-5 text-slate-600 text-[11px]">
                          {result}
                        </p>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </>
        )}

        {/* ── 健康档案（Neo4j 图记忆）─────────────────────────── */}
        {tab === "profile" && (
          <>
            <div className="mb-3 flex items-center justify-between gap-2">
              <p className="text-[10px] text-slate-400">结构化档案 · Neo4j 图存储</p>
              <div className="flex gap-1.5">
                <button onClick={refreshProfile} disabled={profileLoading}
                  className="rounded-lg border border-slate-200 px-2.5 py-1 text-[10px] font-medium text-slate-500 hover:bg-slate-50 disabled:opacity-40">
                  刷新
                </button>
                <button
                  onClick={async () => { setDownloading(true); try { await downloadReportPdf(); } finally { setDownloading(false); } }}
                  disabled={downloading}
                  className="rounded-lg bg-emerald-500 px-3 py-1 text-[10px] font-medium text-white hover:bg-emerald-600 disabled:opacity-50">
                  {downloading ? "生成中…" : "下载报告"}
                </button>
              </div>
            </div>

            {profileLoading && (
              <div className="flex justify-center py-6">
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-emerald-500" />
              </div>
            )}
            {!profileLoading && profileError && (
              <p className="text-xs text-red-400">{profileError}</p>
            )}
            {!profileLoading && !profileError && Object.keys(profileGroups).length === 0 && (
              <p className="text-xs text-slate-400">暂无健康档案。对话后档案将自动更新。</p>
            )}

            <div className="space-y-4">
              {PROFILE_CATEGORY_ORDER.filter((cat) => profileGroups[cat]).map((cat) => {
                const style = PROFILE_STYLES[cat] ?? { bg: "bg-slate-50", text: "text-slate-700", border: "border-slate-200" };
                return (
                  <div key={cat}>
                    <div className={`mb-1.5 inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${style.bg} ${style.text} ${style.border}`}>
                      {cat}
                    </div>
                    <ul className="space-y-1.5">
                      {profileGroups[cat].map((item, idx) => (
                        <li key={idx} className={`rounded-xl border p-3 text-xs ${style.bg} ${style.border}`}>
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <span className={`font-semibold ${style.text}`}>{item.name}</span>
                              {item.detail && (
                                <span className="ml-1.5 text-slate-500">{item.detail}</span>
                              )}
                            </div>
                            <div className="flex shrink-0 items-center gap-1.5">
                              {item.confirmed && (
                                <span className="rounded-full bg-white/80 border border-current px-1.5 py-0.5 text-[9px] font-medium text-emerald-600">
                                  医生确认
                                </span>
                              )}
                              <ConfidenceBar value={item.confidence} />
                            </div>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {/* ── 语义记忆（Qdrant / mem0）────────────────────────── */}
        {tab === "episodic" && (
          <>
            <div className="mb-3 flex items-center justify-between">
              <p className="text-[10px] text-slate-400">情景记忆 · Qdrant 向量存储</p>
              <button onClick={refreshMemories} disabled={memLoading}
                className="rounded-lg border border-slate-200 px-2.5 py-1 text-[10px] font-medium text-slate-500 hover:bg-slate-50 disabled:opacity-40">
                刷新
              </button>
            </div>

            {memLoading && (
              <div className="flex justify-center py-6">
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-emerald-500" />
              </div>
            )}
            {!memLoading && memError && <p className="text-xs text-red-400">{memError}</p>}
            {!memLoading && !memError && memories.length === 0 && (
              <p className="text-xs text-slate-400">暂无语义记忆。</p>
            )}
            <ul className="space-y-2">
              {memories.map((m) => {
                const tagStyle = m.category
                  ? (EPISODIC_TAG_STYLES[m.category] ?? "bg-slate-100 text-slate-600")
                  : null;
                return (
                  <li key={`${m.source}-${m.id}`} className="rounded-xl border border-slate-100 bg-slate-50 p-3 text-xs">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        {tagStyle && (
                          <span className={`mb-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${tagStyle}`}>
                            {m.category}
                          </span>
                        )}
                        <p className="mt-1 leading-5 text-slate-600">{m.content}</p>
                      </div>
                      <button
                        onClick={() => handleDelete(m)}
                        className="shrink-0 text-slate-300 hover:text-red-400"
                        title="删除"
                      >
                        ✕
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          </>
        )}

        {/* ── 导入记忆 ────────────────────────────────────────── */}
        {tab === "import" && (
          <div className="space-y-3">
            <p className="text-xs text-slate-500 leading-5">
              粘贴任意文字，AI 会自动提取健康事实写入记忆库。
              适合批量导入历史记录或就诊摘要。
            </p>
            <textarea
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
              rows={8}
              placeholder={"例：\n患者有高血压病史三年，长期服用氨氯地平5mg\n2024年体检空腹血糖6.2mmol/L\n对阿司匹林过敏"}
              className="w-full rounded-xl border border-slate-200 bg-white p-3 text-xs text-slate-700 outline-none focus:border-emerald-400 resize-none leading-5"
            />
            <button
              onClick={handleImport}
              disabled={importing || !importText.trim()}
              className="flex w-full items-center justify-center gap-1.5 rounded-lg bg-emerald-500 py-2 text-xs font-medium text-white hover:bg-emerald-600 disabled:opacity-50 transition-colors"
            >
              {importing ? (
                <>
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                  导入中…
                </>
              ) : "写入记忆库"}
            </button>
            {importMsg && (
              <p className={`text-xs ${importMsg.ok ? "text-emerald-600" : "text-red-400"}`}>
                {importMsg.text}
              </p>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 85 ? "bg-emerald-400" : pct >= 70 ? "bg-amber-400" : "bg-slate-300";
  return (
    <div className="flex items-center gap-1" title={`置信度 ${pct}%`}>
      <div className="h-1.5 w-10 rounded-full bg-slate-200 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[9px] text-slate-400">{pct}%</span>
    </div>
  );
}
