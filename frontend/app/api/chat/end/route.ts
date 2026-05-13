import { proxyToBackend } from "@/lib/serverProxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  const body = await req.text();
  const upstream = await proxyToBackend("/chat/end", {
    method: "POST",
    body,
    headers: { "Content-Type": "application/json" },
  }, req);
  return new Response(upstream.body, { status: upstream.status });
}
