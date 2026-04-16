import { useCallback, useEffect, useState } from "react";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface ToolEvent {
  tool: string;
  summary: string;
}

export function useAgentChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [statusText, setStatusText] = useState<string>("");
  const [isConsulting, setIsConsulting] = useState(false);
  const [emergencyText, setEmergencyText] = useState<string | null>(null);
  const [sessionId] = useState(() => `web-${Date.now()}`);

  // 页面加载时触发后端记忆预热，消除首条消息的冷启动延迟
  useEffect(() => {
    fetch("/api/chat/warmup", { method: "POST" }).catch(() => {});
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setIsStreaming(true);
      if (text.trim() === "/consult") setIsConsulting(true);

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

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
            } else if (payload.type === "tool_call") {
              setToolEvents((prev) => [
                ...prev,
                { tool: payload.tool, summary: payload.summary },
              ]);
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

  return { messages, toolEvents, sendMessage, isStreaming, statusText, isConsulting, emergencyText, dismissEmergency };
}
