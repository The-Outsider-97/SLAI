/** Client for the shared SLAI server envelope: `{ok:true,data}` or `{ok:false,error:{code,message,hint?,details?}}`.
 *  The server sets an HttpOnly session cookie when the page loads; mutations carry the `X-SLAI-<App>` marker header. */

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly hint: string | undefined;
  readonly details: unknown;
  constructor(status: number, code: string, message: string, hint?: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.hint = hint;
    this.details = details;
  }
  /** The message plus its hint, ready for a toast or notice. */
  get friendly(): string { return this.hint ? `${this.message} ${this.hint}` : this.message; }
}

interface Envelope<T> { ok: boolean; data?: T; error?: { code?: string; message?: string; hint?: string; details?: unknown } }

export interface ClientOptions {
  /** e.g. "splanner": the marker header becomes `X-SLAI-Splanner`. */
  slug: string;
  base?: string;
  /** Called for every failure, including network ones, before the promise rejects (for a global offline banner). */
  onError?: (error: ApiError) => void;
}

export interface Client {
  get<T>(path: string, query?: Query): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  put<T>(path: string, body?: unknown): Promise<T>;
  patch<T>(path: string, body?: unknown): Promise<T>;
  del<T>(path: string): Promise<T>;
  /** Fetch a file response (export, backup) as a Blob with the server's suggested name. */
  download(path: string, query?: Query): Promise<{ blob: Blob; filename: string }>;
}

export type Query = Record<string, string | number | boolean | null | undefined>;

export function queryString(query?: Query): string {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
  const text = params.toString();
  return text ? `?${text}` : "";
}

export function markerHeader(slug: string): string {
  return `X-SLAI-${slug.charAt(0).toUpperCase()}${slug.slice(1)}`;
}

function filenameFrom(disposition: string | null, fallback: string): string {
  const match = disposition ? /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition) : null;
  if (!match || !match[1]) return fallback;
  try { return decodeURIComponent(match[1]); } catch { return match[1]; }
}

export function createClient(options: ClientOptions): Client {
  const base = options.base ?? "/api";
  const marker = markerHeader(options.slug);

  const fail = (error: ApiError): never => { options.onError?.(error); throw error; };

  async function request(method: string, path: string, body?: unknown, query?: Query): Promise<Response> {
    const headers: Record<string, string> = { Accept: "application/json", [marker]: "1" };
    const init: RequestInit = { method, headers, credentials: "same-origin", cache: "no-store" };
    if (body !== undefined) { headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
    try {
      return await fetch(`${base}${path}${queryString(query)}`, init);
    } catch (cause) {
      return fail(new ApiError(0, "offline", "Can't reach the local server.", "Check that the application is still running, then try again.", String(cause)));
    }
  }

  async function json<T>(method: string, path: string, body?: unknown, query?: Query): Promise<T> {
    const response = await request(method, path, body, query);
    let parsed: Envelope<T> | null = null;
    try { parsed = (await response.json()) as Envelope<T>; } catch { parsed = null; }
    if (parsed && parsed.ok) return parsed.data as T;
    const error = parsed?.error;
    return fail(new ApiError(response.status, error?.code ?? "http_error", error?.message ?? `The server answered ${response.status}.`, error?.hint, error?.details));
  }

  return {
    get: (path, query) => json("GET", path, undefined, query),
    post: (path, body) => json("POST", path, body ?? {}),
    put: (path, body) => json("PUT", path, body ?? {}),
    patch: (path, body) => json("PATCH", path, body ?? {}),
    del: (path) => json("DELETE", path),
    async download(path, query) {
      const response = await request("GET", path, undefined, query);
      if (!response.ok) {
        let parsed: Envelope<unknown> | null = null;
        try { parsed = (await response.json()) as Envelope<unknown>; } catch { parsed = null; }
        const error = parsed?.error;
        return fail(new ApiError(response.status, error?.code ?? "http_error", error?.message ?? `The server answered ${response.status}.`, error?.hint, error?.details));
      }
      return { blob: await response.blob(), filename: filenameFrom(response.headers.get("Content-Disposition"), "download") };
    },
  };
}
