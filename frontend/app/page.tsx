"use client";

import ChatWindow from "@/components/ChatWindow";
import EmergencyModal from "@/components/EmergencyModal";
import MemoryPanel from "@/components/MemoryPanel";
import { useAgentChat } from "@/lib/useAgentChat";

export default function Home() {
  const { messages, toolEvents, sendMessage, isStreaming, statusText, isConsulting, emergencyText, dismissEmergency } = useAgentChat();

  return (
    <div className="flex h-screen">
      <main className="flex-1 flex flex-col">
        <ChatWindow
          messages={messages}
          isStreaming={isStreaming}
          statusText={statusText}
          isConsulting={isConsulting}
          onSend={sendMessage}
        />
      </main>
      <aside className="w-80 border-l">
        <MemoryPanel events={toolEvents} />
      </aside>

      {emergencyText && (
        <EmergencyModal text={emergencyText} onDismiss={dismissEmergency} />
      )}
    </div>
  );
}
