import { proxyToBackend } from "@/lib/serverProxy";

export async function POST(req: Request): Promise<Response> {
  const body = await req.text();
  const upstream = await proxyToBackend(
    "/memory/import",
    { method: "POST", body, headers: { "Content-Type": "application/json" } },
    req,
  );
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
