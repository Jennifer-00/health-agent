"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getToken, clearToken } from "@/lib/auth";
import ChatWindow from "@/components/ChatWindow";
import EmergencyModal from "@/components/EmergencyModal";
import MemoryPanel from "@/components/MemoryPanel";
import { useAgentChat } from "@/lib/useAgentChat";

export default function Home() {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
    } else {
      setReady(true);
    }
  }, [router]);

  const { messages, toolEvents, toolResultMap, sendMessage, isStreaming, statusText, isConsulting, emergencyText, dismissEmergency, flushMemory } =
    useAgentChat();

  function handleLogout() {
    clearToken();
    router.replace("/login");
  }

  if (!ready) return null;

  return (
    <div className="flex h-screen">
      <main className="flex-1 flex flex-col">
        <ChatWindow
          messages={messages}
          isStreaming={isStreaming}
          statusText={statusText}
          isConsulting={isConsulting}
          onSend={sendMessage}
          onLogout={handleLogout}
          onRemember={flushMemory}
        />
      </main>
      <aside className="w-80 border-l">
        <MemoryPanel events={toolEvents} toolResultMap={toolResultMap} />
      </aside>

      {emergencyText && (
        <EmergencyModal text={emergencyText} onDismiss={dismissEmergency} />
      )}
    </div>
  );
}
