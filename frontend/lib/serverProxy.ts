import { createDevJwt } from "@/lib/serverAuth";

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL ?? "http://127.0.0.1:8000";

export async function proxyToBackend(
  path: string,
  init: RequestInit = {},
  req?: Request,
): Promise<Response> {
  const headers = new Headers(init.headers);

  const clientAuth = req?.headers.get("authorization");
  if (clientAuth) {
    headers.set("Authorization", clientAuth);
  } else {
    headers.set("Authorization", `Bearer ${createDevJwt()}`);
  }

  return fetch(`${BACKEND_BASE_URL}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
}
