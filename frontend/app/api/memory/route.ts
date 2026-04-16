import { proxyToBackend } from "@/lib/serverProxy";

export async function GET(): Promise<Response> {
  const upstream = await proxyToBackend("/memory", { method: "GET" });
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function POST(): Promise<Response> {
  const upstream = await proxyToBackend("/memory/consolidate", { method: "POST" });
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
