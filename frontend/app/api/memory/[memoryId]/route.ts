import { proxyToBackend } from "@/lib/serverProxy";

type Params = {
  params: { memoryId: string };
};

export async function PATCH(req: Request, { params }: Params): Promise<Response> {
  const { memoryId } = params;
  const body = await req.text();
  const contentType = req.headers.get("content-type") ?? "application/json";

  const upstream = await proxyToBackend(`/memory/${memoryId}`, {
    method: "PATCH",
    body,
    headers: {
      "Content-Type": contentType,
    },
  }, req);

  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}

export async function DELETE(req: Request, { params }: Params): Promise<Response> {
  const { memoryId } = params;
  const { searchParams } = new URL(req.url);
  const source = searchParams.get("source") ?? "mem0";

  const upstream = await proxyToBackend(`/memory/${memoryId}?source=${source}`, {
    method: "DELETE",
  }, req);

  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}
