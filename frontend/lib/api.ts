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

export async function downloadReportPdf(): Promise<void> {
  const res = await fetch("/api/report/pdf");
  if (!res.ok) throw new Error("Failed to generate report");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `health_report_${new Date().toISOString().slice(0, 10)}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}

