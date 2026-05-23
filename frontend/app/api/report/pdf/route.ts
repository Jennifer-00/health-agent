import { proxyToBackend } from "@/lib/serverProxy";

export async function GET(req: Request): Promise<Response> {
  const upstream = await proxyToBackend("/report/pdf", { method: "GET" }, req);
  const contentDisposition =
    upstream.headers.get("Content-Disposition") ??
    `attachment; filename="health_report.pdf"`;
  return new Response(await upstream.arrayBuffer(), {
    status: upstream.status,
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": contentDisposition,
    },
  });
}
