import { proxyToBackend } from "@/lib/serverProxy";

export async function GET(): Promise<Response> {
  const upstream = await proxyToBackend("/memory/summary", { method: "GET" });
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
