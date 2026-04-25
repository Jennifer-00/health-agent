"use client";

import { useEffect, useState } from "react";
import type { ToolEvent } from "@/lib/useAgentChat";
import { fetchMemories, deleteMemory, downloadReportPdf } from "@/lib/api";
import type { MemoryItem } from "@/lib/api";

interface Props {
  events: ToolEvent[];
}

type Tab = "events" | "archive";

export default function MemoryPanel({ events }: Props) {
  const [tab, setTab] = useState<Tab>("events");
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (tab !== "archive") return;
    setLoading(true);
    setFetchError(null);
    fetchMemories()
      .then(setMemories)
      .catch((e: Error) => setFetchError(e.message ?? "加载失败，请稍后重试"))
      .finally(() => setLoading(false));
  }, [tab]);

  async function handleDelete(item: MemoryItem) {
    await deleteMemory(item.id);
    setMemories((prev) => prev.filter((m) => m.id !== item.id));
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
              <p className="text-xs text-slate-400">当前会话还没有触发工具调用事件。</p>
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
            <button
              onClick={async () => {
                setDownloading(true);
                try { await downloadReportPdf(); }
                finally { setDownloading(false); }
              }}
              disabled={downloading}
              className="mb-3 flex w-full items-center justify-center gap-1.5 rounded-lg bg-emerald-500 py-1.5 text-xs font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
            >
              {downloading ? "生成中…" : "下载健康报告 PDF"}
            </button>
            {loading && (
              <div className="flex justify-center py-6">
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-emerald-500" />
              </div>
            )}
            {!loading && fetchError && (
              <p className="text-xs text-red-400">{fetchError}</p>
            )}
            {!loading && !fetchError && memories.length === 0 && (
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
