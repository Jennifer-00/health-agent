"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { Message } from "@/lib/useAgentChat";

interface Props {
  messages: Message[];
  isStreaming: boolean;
  statusText?: string;
  isConsulting?: boolean;
  onSend: (text: string) => void;
}

const COMMANDS = [
  { name: "/consult", desc: "进入问诊模式，AI 将主动追问你的症状" },
  { name: "/recommend", desc: "根据健康记录生成个性化健康建议" },
];

export default function ChatWindow({ messages, isStreaming, statusText, isConsulting, onSend }: Props) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedIdx, setSelectedIdx] = useState(0);

  const filteredCommands = input.startsWith("/")
    ? COMMANDS.filter((c) => c.name.startsWith(input.toLowerCase()))
    : [];
  const showMenu = filteredCommands.length > 0;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSend(input.trim());
    setInput("");
    setSelectedIdx(0);
  };

  const selectCommand = (name: string) => {
    setInput(name);
    setSelectedIdx(0);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showMenu) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, filteredCommands.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Tab" || e.key === "Enter") {
      if (showMenu) {
        e.preventDefault();
        selectCommand(filteredCommands[selectedIdx].name);
      }
    } else if (e.key === "Escape") {
      setInput("");
    }
  };

  // 重置选中项
  useEffect(() => {
    setSelectedIdx(0);
  }, [input]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
        <div>
          <h1 className="text-lg font-semibold text-ink">健康对话</h1>
          <p className="text-sm text-slate-500">记录症状、追踪变化、查看工具调用</p>
        </div>
        {isConsulting && (
          <span className="flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-600 ring-1 ring-emerald-200">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            问诊中
          </span>
        )}
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="rounded-2xl border border-dashed border-slate-300 bg-white/70 p-5 text-sm text-slate-500">
            可以直接输入"我这两天头痛"或"帮我回顾最近的健康记录"开始测试。
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-xl rounded-2xl px-4 py-3 text-sm shadow-sm ${
                msg.role === "user"
                  ? "bg-accent text-white"
                  : "border border-slate-200 bg-white text-ink"
              }`}
            >
              {msg.role === "user" ? (
                <p className="whitespace-pre-wrap leading-6">{msg.content}</p>
              ) : (
                <div className="prose prose-sm max-w-none leading-6
                  prose-p:my-1 prose-ul:my-1 prose-ol:my-1
                  prose-li:my-0.5 prose-strong:font-semibold
                  prose-headings:font-semibold prose-headings:my-2">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ href, children }) => (
                        <a href={href} target="_blank" rel="noopener noreferrer">
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        ))}

        {isStreaming && (
          <div className="flex items-center gap-1.5 text-xs text-slate-400 animate-pulse">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent animate-bounce" />
            <span>{statusText || "Agent 正在思考..."}</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="border-t border-slate-200 bg-white/80 p-4">
        {/* 斜杠命令菜单 */}
        {showMenu && (
          <div className="mb-2 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg">
            {filteredCommands.map((cmd, i) => (
              <button
                key={cmd.name}
                type="button"
                onClick={() => selectCommand(cmd.name)}
                className={`flex w-full items-baseline gap-3 px-4 py-2.5 text-left text-sm transition ${
                  i === selectedIdx
                    ? "bg-accent/10 text-accent"
                    : "text-ink hover:bg-slate-50"
                }`}
              >
                <span className="font-mono font-semibold">{cmd.name}</span>
                <span className="text-xs text-slate-400">{cmd.desc}</span>
              </button>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <input
            ref={inputRef}
            id="chat-message"
            name="chatMessage"
            className="flex-1 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-accent"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="描述症状，或输入 / 查看可用命令"
            autoComplete="off"
          />
          <button
            type="submit"
            disabled={isStreaming}
            className="rounded-xl bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            发送
          </button>
        </div>
      </form>
    </div>
  );
}
