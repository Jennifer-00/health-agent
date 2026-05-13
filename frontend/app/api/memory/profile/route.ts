import { proxyToBackend } from "@/lib/serverProxy";

export async function GET(req: Request): Promise<Response> {
  const upstream = await proxyToBackend("/memory/profile", { method: "GET" }, req);
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
