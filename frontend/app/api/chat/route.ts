import { proxyToBackend } from "@/lib/serverProxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  const body = await req.text();
  const contentType = req.headers.get("content-type") ?? "application/json";

  const upstream = await proxyToBackend("/chat", {
    method: "POST",
    body,
    headers: {
      "Content-Type": contentType,
    },
  });

  if (!upstream.body) {
    return new Response(
      JSON.stringify({ error: "Chat upstream returned no response body" }),
      {
        status: 502,
        headers: {
          "Content-Type": "application/json",
        },
      },
    );
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
