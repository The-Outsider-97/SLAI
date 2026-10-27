import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ApiError, createClient, markerHeader, queryString } from "../web/src/api.js";
import { DEFAULT_APPEARANCE, normaliseAppearance } from "../web/src/theme.js";
import { clamp, compareText, debounce, Emitter, formatBytes, plural, uid } from "../web/src/util.js";
import { hasGlyph, registerGlyph } from "../web/src/icons.js";

interface Seen { url: string; init: RequestInit }

function fakeFetch(reply: (url: string, init: RequestInit) => Response | Promise<Response>, seen: Seen[] = []): Seen[] {
  (globalThis as { fetch: unknown }).fetch = async (url: string, init: RequestInit = {}) => { seen.push({ url, init }); return reply(url, init); };
  return seen;
}

const jsonResponse = (body: unknown, status = 200, headers: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

describe("api client", () => {
  it("names the marker header after the application", () => {
    assert.equal(markerHeader("splanner"), "X-SLAI-Splanner");
    assert.equal(markerHeader("scical"), "X-SLAI-Scical");
  });

  it("builds query strings and skips empty values", () => {
    assert.equal(queryString({ q: "a b", n: 3, flag: false, none: null, empty: "", gone: undefined }), "?q=a+b&n=3&flag=false");
    assert.equal(queryString({}), "");
  });

  it("unwraps data and sends the marker and JSON body on mutations", async () => {
    const seen = fakeFetch(() => jsonResponse({ ok: true, data: { id: "1" } }));
    const client = createClient({ slug: "splanner" });
    assert.deepEqual(await client.post("/tasks", { title: "x" }), { id: "1" });
    const sent = seen[0];
    assert.ok(sent);
    assert.equal(sent.url, "/api/tasks");
    const headers = sent.init.headers as Record<string, string>;
    assert.equal(headers["X-SLAI-Splanner"], "1");
    assert.equal(headers["Content-Type"], "application/json");
    assert.equal(sent.init.body, JSON.stringify({ title: "x" }));
    assert.equal(sent.init.credentials, "same-origin");
  });

  it("turns the error envelope into an ApiError with its hint", async () => {
    fakeFetch(() => jsonResponse({ ok: false, error: { code: "bad_request", message: "Title is required.", hint: "Type one." } }, 400));
    const failures: ApiError[] = [];
    const client = createClient({ slug: "splanner", onError: (e) => failures.push(e) });
    await assert.rejects(() => client.get("/x"), (error: unknown) => error instanceof ApiError && error.code === "bad_request" && error.status === 400 && error.friendly === "Title is required. Type one.");
    assert.equal(failures.length, 1);
  });

  it("reports an unreachable server as offline and a non-JSON failure as http_error", async () => {
    (globalThis as { fetch: unknown }).fetch = async () => { throw new TypeError("fetch failed"); };
    const client = createClient({ slug: "splanner" });
    await assert.rejects(() => client.get("/x"), (error: unknown) => error instanceof ApiError && error.code === "offline" && error.status === 0);
    fakeFetch(() => new Response("<html>boom</html>", { status: 502 }));
    await assert.rejects(() => client.get("/x"), (error: unknown) => error instanceof ApiError && error.code === "http_error" && error.status === 502);
  });

  it("downloads a file with the server's filename", async () => {
    fakeFetch(() => new Response("hello", { status: 200, headers: { "Content-Disposition": 'attachment; filename="splanner-backup.json"' } }));
    const client = createClient({ slug: "splanner" });
    const file = await client.download("/export");
    assert.equal(file.filename, "splanner-backup.json");
    assert.equal(await file.blob.text(), "hello");
  });
});

describe("appearance", () => {
  it("falls back to defaults and clamps text size", () => {
    assert.deepEqual(normaliseAppearance(null), DEFAULT_APPEARANCE);
    assert.deepEqual(normaliseAppearance({ theme: "neon" as never, scale: Number.NaN, motion: "x" as never }), DEFAULT_APPEARANCE);
    assert.equal(normaliseAppearance({ scale: 9 }).scale, 1.4);
    assert.equal(normaliseAppearance({ scale: 0.1 }).scale, 0.85);
    assert.equal(normaliseAppearance({ theme: "contrast", motion: "reduce" }).theme, "contrast");
  });
});

describe("util", () => {
  it("clamps, formats and sorts", () => {
    assert.equal(clamp(5, 1, 3), 3);
    assert.equal(clamp(-5, 1, 3), 1);
    assert.equal(formatBytes(512), "512 B");
    assert.equal(formatBytes(1536), "1.5 KB");
    assert.equal(formatBytes(5 * 1024 * 1024), "5.0 MB");
    assert.equal(formatBytes(-1), "—");
    assert.ok(compareText("Note 2", "note 10") < 0);
  });

  it("makes 32-hex ids", () => {
    const a = uid();
    assert.ok(/^[0-9a-f]{32}$/.test(a));
    assert.notEqual(a, uid());
  });

  it("emits to subscribers and unsubscribes", () => {
    const emitter = new Emitter<number>();
    const seen: number[] = [];
    const off = emitter.on((n) => seen.push(n));
    emitter.emit(1);
    off();
    emitter.emit(2);
    assert.deepEqual(seen, [1]);
    assert.equal(emitter.size, 0);
  });

  it("debounces to the last call, and flush runs it immediately", async () => {
    const calls: number[] = [];
    (globalThis as { window?: unknown }).window = globalThis;
    const fn = debounce((n: number) => calls.push(n), 10);
    fn(1); fn(2); fn(3);
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.deepEqual(calls, [3]);
    fn(4);
    fn.flush();
    assert.deepEqual(calls, [3, 4]);
    fn(5);
    fn.cancel();
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.deepEqual(calls, [3, 4]);
  });
});

describe("icons", () => {
  it("lets applications register their own glyphs", () => {
    assert.ok(hasGlyph("calendar"));
    assert.ok(!hasGlyph("splanner-test"));
    registerGlyph("splanner-test", ["M0 0h1"]);
    assert.ok(hasGlyph("splanner-test"));
  });
});

describe("plural", () => {
  it("counts with the right noun", () => {
    assert.equal(plural(0, "note"), "0 notes");
    assert.equal(plural(1, "note"), "1 note");
    assert.equal(plural(2, "sticky note"), "2 sticky notes");
    assert.equal(plural(2, "box", "boxes"), "2 boxes");
    assert.equal(plural(1, "box", "boxes"), "1 box");
  });
});
