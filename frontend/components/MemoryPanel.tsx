"use client";

import { useEffect, useState } from "react";
import type { ToolEvent } from "@/lib/useAgentChat";
import { fetchMemories, deleteMemory, consolidateMemories, fetchHealthReport } from "@/lib/api";
import type { MemoryItem } from "@/lib/api";

interface Props {
  events: ToolEvent[];
}

type Tab = "events" | "archive";

export default function MemoryPanel({ events }: Props) {
  const [tab, setTab] = useState<Tab>("events");
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [consolidating, setConsolidating] = useState(false);
  const [consolidateMsg, setConsolidateMsg] = useState("");
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    if (tab !== "archive") return;
    setLoading(true);
    fetchMemories()
      .then(setMemories)
      .finally(() => setLoading(false));
  }, [tab]);

  async function handleDelete(item: MemoryItem) {
    await deleteMemory(item.id);
    setMemories((prev) => prev.filter((m) => m.id !== item.id));
  }

  async function handleExportReport() {
    setExporting(true);
    try {
      const summary = await fetchHealthReport();
      const date = new Date().toLocaleDateString("zh-CN", {
        year: "numeric", month: "long", day: "numeric",
      });
      const reportMd = [
        "# 个人健康记录报告",
        "",
        `> 生成时间：${date}`,
        "> 本报告由健康管理 AI Agent 自动整理，仅供就医参考，不构成医疗诊断。",
        "",
        "---",
        "",
        "## 健康记录摘要",
        "",
        summary || "暂无健康记录。",
        "",
        "---",
        "",
        "*就医时请将本报告交给医生参考*",
      ].join("\n");

      const blob = new Blob([reportMd], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `健康报告_${date}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("导出失败，请稍后重试");
    } finally {
      setExporting(false);
    }
  }

  async function handleConsolidate() {
    setConsolidating(true);
    setConsolidateMsg("");
    try {
      const result = await consolidateMemories();
      setConsolidateMsg(result || "整理完成");
      // 刷新记忆列表
      const updated = await fetchMemories();
      setMemories(updated);
    } catch {
      setConsolidateMsg("整理失败，请稍后重试");
    } finally {
      setConsolidating(false);
    }
  }

  return (
    <div className="flex h-full flex-col bg-white/80">
      {/* 标签切换 */}
      <div className="flex border-b">
        <button
          onClick={() => setTab("events")}
          className={`flex-1 py-2 text-xs font-medium ${
            tab === "events"
              ? "border-b-2 border-emerald-500 text-emerald-700"
              : "text-slate-400"
          }`}
        >
          工具事件
        </button>
        <button
          onClick={() => setTab("archive")}
          className={`flex-1 py-2 text-xs font-medium ${
            tab === "archive"
              ? "border-b-2 border-emerald-500 text-emerald-700"
              : "text-slate-400"
          }`}
        >
          记忆档案
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {/* 工具事件 */}
        {tab === "events" && (
          <>
            {events.length === 0 && (
              <p className="text-xs text-slate-400">当前会话还没有触发记忆写入或其他工具事件。</p>
            )}
            <ul className="space-y-2">
              {events.map((ev, i) => (
                <li key={i} className="rounded-xl border border-emerald-100 bg-emerald-50 p-3 text-xs">
                  <span className="font-medium text-emerald-700">{ev.tool}</span>
                  <p className="mt-1 leading-5 text-slate-600">{ev.summary}</p>
                </li>
              ))}
            </ul>
          </>
        )}

        {/* 记忆档案 */}
        {tab === "archive" && (
          <>
            {/* 工具栏 */}
            <div className="mb-3 flex items-center gap-2">
              <button
                onClick={handleConsolidate}
                disabled={consolidating || loading}
                className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-emerald-300 hover:text-emerald-600 disabled:opacity-50"
              >
                {consolidating ? (
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-300 border-t-emerald-500" />
                ) : (
                  "✦"
                )}
                整理记忆
              </button>
              <button
                onClick={handleExportReport}
                disabled={exporting || loading}
                className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-blue-300 hover:text-blue-600 disabled:opacity-50"
              >
                {exporting ? (
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-300 border-t-blue-500" />
                ) : (
                  "↓"
                )}
                导出报告
              </button>
              {consolidateMsg && (
                <span className="ml-auto text-[10px] text-slate-400">{consolidateMsg}</span>
              )}
            </div>

            {/* 加载中转圈 */}
            {loading && (
              <div className="flex justify-center py-6">
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-emerald-500" />
              </div>
            )}
            {!loading && memories.length === 0 && (
              <p className="text-xs text-slate-400">暂无健康记忆。</p>
            )}
            <ul className="space-y-2">
              {memories.map((m) => (
                <li
                  key={`${m.source}-${m.id}`}
                  className="rounded-xl border border-slate-100 bg-slate-50 p-3 text-xs"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1">
                      {m.category && (
                        <span className="mb-1 inline-block rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] text-emerald-700">
                          {m.category}
                        </span>
                      )}
                      {m.record_date && (
                        <span className="ml-1 text-[10px] text-slate-400">{m.record_date}</span>
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
                  <span className="mt-1 block text-[10px] text-slate-300">{m.source}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
