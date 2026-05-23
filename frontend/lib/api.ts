// REST client for backend API
import { authHeaders } from "@/lib/auth";

export interface MemoryItem {
  id: string;
  content: string;
  category: string | null;
  record_date: string | null;
  source: "mem0" | "db";
}

export interface ProfileItem {
  category: string;   // 诊断 / 用药 / 过敏 / 病史 / 家族史
  name: string;
  detail: string;
  confidence: number;
  confirmed: boolean; // 医生确认
}

export async function fetchMemories(): Promise<MemoryItem[]> {
  const res = await fetch("/api/memory", { headers: authHeaders() });
  if (!res.ok) throw new Error("Failed to fetch memories");
  return res.json();
}

export async function fetchProfile(): Promise<ProfileItem[]> {
  const res = await fetch("/api/memory/profile", { headers: authHeaders() });
  if (!res.ok) throw new Error("Failed to fetch profile");
  return res.json();
}

export async function deleteMemory(memoryId: string): Promise<void> {
  const res = await fetch(`/api/memory/${memoryId}`, { method: "DELETE", headers: authHeaders() });
  if (!res.ok) throw new Error("Failed to delete memory");
}

export async function importMemory(text: string): Promise<void> {
  const res = await fetch("/api/memory/import", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("导入失败");
}

export async function downloadReportPdf(): Promise<void> {
  const res = await fetch("/api/report/pdf", { headers: authHeaders() });
  if (!res.ok) throw new Error("Failed to generate report");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `health_report_${new Date().toISOString().slice(0, 10)}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}

