import { useCallback, useEffect, useState } from "react";
import { authHeaders, clearToken } from "@/lib/auth";

export interface Message {
  role: "user" | "assistant";
  content: string;
  intent?: string;
}

export interface ToolEvent {
  tool: string;
  summary: string;
}

export interface ToolResult {
  tool: string;
  preview: string;
}

export function useAgentChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [toolResultMap, setToolResultMap] = useState<Record<string, string>>({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [statusText, setStatusText] = useState<string>("");
  const [isConsulting, setIsConsulting] = useState(false);
  const [emergencyText, setEmergencyText] = useState<string | null>(null);
  const [sessionId] = useState(() => `web-${crypto.randomUUID()}`);

  // 页面加载时触发后端记忆预热，消除首条消息的冷启动延迟
  useEffect(() => {
    fetch("/api/chat/warmup", { method: "POST", headers: authHeaders() }).catch(() => {});
  }, []);

  // 关闭页面或组件卸载时，将剩余未写入的对话 flush 到 Mem0
  useEffect(() => {
    const flushSession = () => {
      fetch("/api/chat/end", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ session_id: sessionId }),
        keepalive: true,
      }).catch(() => {});
    };

    window.addEventListener("beforeunload", flushSession);
    return () => {
      window.removeEventListener("beforeunload", flushSession);
      flushSession();
    };
  }, [sessionId]);

  const sendMessage = useCallback(
    async (text: string) => {
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setIsStreaming(true);
      if (text.trim() === "/consult") setIsConsulting(true);

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (res.status === 401) {
        clearToken();
        window.location.replace("/login");
        return;
      }

      if (!res.body) {
        setIsStreaming(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";
      let buffer = "";

      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        buffer = buffer.replace(/\r\n/g, "\n");
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const rawEvent of events) {
          const line = rawEvent
            .split("\n")
            .find((candidate) => candidate.startsWith("data: "));
          if (!line) continue;

          try {
            const payload = JSON.parse(line.slice(6));
            console.debug("chat sse payload", payload);
            if (payload.type === "emergency") {
              setEmergencyText(payload.text ?? "");
              setIsStreaming(false);
            } else if (payload.type === "mode") {
              setIsConsulting(payload.mode === "consult");
            } else if (payload.type === "status") {
              setStatusText(payload.text ?? "");
            } else if (payload.type === "text") {
              setStatusText("");
              assistantText += payload.delta;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: "assistant",
                  content: assistantText,
                };
                return updated;
              });
            } else if (payload.type === "intent") {
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === "assistant") {
                  updated[updated.length - 1] = { ...last, intent: payload.intent };
                }
                return updated;
              });
            } else if (payload.type === "tool_call") {
              setToolEvents((prev) => [
                ...prev,
                { tool: payload.tool, summary: payload.summary },
              ]);
            } else if (payload.type === "tool_result") {
              setToolResultMap((prev) => ({
                ...prev,
                [payload.tool]: payload.preview,
              }));
            } else if (payload.type === "done") {
              setStatusText("");
              setIsStreaming(false);
            }
          } catch {
            // 忽略解析失败的事件
          }
        }
      }

      buffer += decoder.decode();
      buffer = buffer.replace(/\r\n/g, "\n");
      if (buffer.trim()) {
        const line = buffer
          .split("\n")
          .find((candidate) => candidate.startsWith("data: "));
        if (line) {
          try {
            const payload = JSON.parse(line.slice(6));
            console.debug("chat sse tail payload", payload);
            if (payload.type === "status") {
              setStatusText(payload.text ?? "");
            } else if (payload.type === "text") {
              setStatusText("");
              assistantText += payload.delta;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: "assistant",
                  content: assistantText,
                };
                return updated;
              });
            } else if (payload.type === "tool_call") {
              setToolEvents((prev) => [
                ...prev,
                { tool: payload.tool, summary: payload.summary },
              ]);
            }
          } catch {
            // 忽略尾部不完整事件
          }
        }
      }

      setIsStreaming(false);
    },
    [sessionId],
  );

  const dismissEmergency = useCallback(() => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant" && last.content === "" && emergencyText) {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: emergencyText };
        return updated;
      }
      return prev;
    });
    setEmergencyText(null);
  }, [emergencyText]);

  // 手动立即写入记忆：flush 当前 session 的 pending 轮次，轮数清零重新计
  const flushMemory = useCallback(async () => {
    await fetch("/api/chat/end", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ session_id: sessionId }),
    });
  }, [sessionId]);

  return { messages, toolEvents, toolResultMap, sendMessage, isStreaming, statusText, isConsulting, emergencyText, dismissEmergency, flushMemory };
}
