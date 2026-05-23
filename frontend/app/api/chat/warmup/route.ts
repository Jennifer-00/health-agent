import { proxyToBackend } from "@/lib/serverProxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  const upstream = await proxyToBackend("/chat/warmup", { method: "POST" }, req);
  return new Response(upstream.body, { status: upstream.status });
}
