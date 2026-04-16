// REST client for backend API

export interface MemoryItem {
  id: string;
  content: string;
  category: string | null;
  record_date: string | null;
  source: "mem0" | "db";
}

export async function fetchMemories(): Promise<MemoryItem[]> {
  const res = await fetch("/api/memory");
  if (!res.ok) throw new Error("Failed to fetch memories");
  return res.json();
}

export async function deleteMemory(memoryId: string): Promise<void> {
  const res = await fetch(`/api/memory/${memoryId}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete memory");
}

export async function consolidateMemories(): Promise<string> {
  const res = await fetch("/api/memory", { method: "POST" });
  if (!res.ok) throw new Error("Failed to consolidate memories");
  const data = await res.json();
  return data.result ?? "";
}

export async function fetchHealthReport(): Promise<string> {
  const res = await fetch("/api/memory/summary");
  if (!res.ok) throw new Error("Failed to fetch health report");
  const data = await res.json();
  return data.summary ?? "";
}
