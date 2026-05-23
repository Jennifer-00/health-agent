import { proxyToBackend } from "@/lib/serverProxy";

export async function GET(req: Request): Promise<Response> {
  const upstream = await proxyToBackend("/memory", { method: "GET" }, req);
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function POST(req: Request): Promise<Response> {
  const upstream = await proxyToBackend("/memory/consolidate", { method: "POST" }, req);
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
