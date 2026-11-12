import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import vm from "node:vm";

/** The shared start-up loader (web/startup/loader.js) run against a hand-made page, so the rule that matters is tested without a browser:
 *  only the application's own code may fail its start-up, and the ready signal always clears the screen. */
const source = readFileSync(new URL("../web/startup/loader.js", import.meta.url), "utf8");
const ORIGIN = "http://127.0.0.1:8771";

class FakeElement {
  children: FakeElement[] = []; attrs: Record<string, string> = {}; className = ""; textContent = ""; open = false; removed = false; onclick: (() => void) | null = null;
  constructor(readonly tag: string) {}
  setAttribute(name: string, value: string): void { this.attrs[name] = value; }
  getAttribute(name: string): string | null { return this.attrs[name] ?? null; }
  append(...kids: FakeElement[]): void { this.children.push(...kids); }
  addEventListener(): void { /* the loader only listens for cancel here */ }
  querySelector(selector: string): FakeElement | null { return this.children.find((c) => c.tag === selector || (selector === "[role=alert]" && c.attrs["role"] === "alert")) ?? null; }
  get classList() { return { add: (n: string) => { this.className += ` ${n}`; }, contains: (n: string) => this.className.split(" ").includes(n) }; }
  showModal(): void { this.open = true; }
  close(): void { this.open = false; }
  remove(): void { this.removed = true; }
}

function page() {
  const listeners = new Map<string, Array<(event: unknown) => void>>();
  const panels: FakeElement[] = [];
  class FakeScript {}
  class FakeErrorEvent { constructor(readonly message: string, readonly filename: string) {} }
  const sandbox: Record<string, unknown> = {
    URL, AbortController, setTimeout, clearTimeout, HTMLScriptElement: FakeScript, ErrorEvent: FakeErrorEvent,
    Image: class { alt = ""; src = ""; addEventListener(): void {} getAttribute(): string { return this.src; } remove(): void {} },
    location: { origin: ORIGIN, href: `${ORIGIN}/`, reload: () => {} },
    document: {
      currentScript: { dataset: { name: "SPADRA", logo: "/assets/logo.png" } },
      createElement: (tag: string) => { const el = new FakeElement(tag); if (tag === "dialog") panels.push(el); return el; },
      body: { append: () => {} },
    },
    addEventListener: (type: string, fn: (event: unknown) => void) => { listeners.set(type, [...(listeners.get(type) ?? []), fn]); },
  };
  sandbox["window"] = sandbox;
  vm.runInNewContext(source, sandbox);
  const panel = panels[0] as FakeElement;
  const fire = (type: string, event: object = {}): void => { for (const fn of listeners.get(type) ?? []) fn(event); };
  const script = (src: string): object => Object.assign(new FakeScript(), { src });
  const status = (): FakeElement => panel.children.find((c) => c.tag === "p") as FakeElement;
  return { panel, fire, status, error: (message: string, filename: string): void => fire("error", new FakeErrorEvent(message, filename)), failedScript: (src: string): void => fire("error", { target: script(src) }) };
}

describe("start-up loader", () => {
  it("opens as soon as the script runs, saying what is loading", () => {
    const p = page();
    assert.equal(p.panel.open, true);
    assert.equal(p.status().textContent, "Loading SPADRA");
  });

  it("is not stopped by a browser extension throwing on the page (the wallet error seen on 127.0.0.1)", () => {
    const p = page();
    p.error("Uncaught TypeError: Cannot assign to read only property 'ethereum' of object '#<Window>'", "chrome-extension://abcdef/inpage.js");
    p.error("Uncaught TypeError: Cannot assign to read only property 'ethereum' of object '#<Window>'", `${ORIGIN}/`);   // injected inline: the page's own address
    p.error("Script error.", "");
    p.fire("unhandledrejection", { reason: new Error("extension noise") });
    assert.equal(p.panel.classList.contains("failed"), false);
    assert.equal(p.status().textContent, "Loading SPADRA");
    p.fire("slai:ready");
    assert.equal(p.panel.removed, true);
  });

  it("is stopped by the application's own script failing, and says why", () => {
    const p = page();
    p.error("boom", `${ORIGIN}/js/SPADRA/src/client/main.js`);
    assert.equal(p.panel.classList.contains("failed"), true);
    assert.equal(p.status().textContent, "boom");
    assert.ok(p.panel.querySelector("button"));
  });

  it("is stopped by the application's own script not loading, but not by someone else's", () => {
    const other = page();
    other.failedScript("https://cdn.example.org/js/lib.js");
    other.failedScript(`${ORIGIN}/somewhere/else.js`);
    assert.equal(other.panel.classList.contains("failed"), false);
    const own = page();
    own.failedScript(`${ORIGIN}/js/SPADRA/src/client/main.js`);
    assert.equal(own.panel.classList.contains("failed"), true);
  });

  it("counts an unhandled rejection from the application's code, not one from elsewhere", () => {
    const p = page();
    p.fire("unhandledrejection", { reason: { message: "elsewhere", stack: "Error: elsewhere\n    at x (chrome-extension://abc/a.js:1:1)" } });
    assert.equal(p.panel.classList.contains("failed"), false);
    p.fire("unhandledrejection", { reason: { message: "ours", stack: `Error: ours\n    at start (${ORIGIN}/js/SPADRA/src/client/main.js:9:3)` } });
    assert.equal(p.panel.classList.contains("failed"), true);
  });

  it("keeps the explicit start-up error of the application, and ready does not hide it", () => {
    const p = page();
    p.fire("slai:startup-error", { detail: "The server did not answer" });
    assert.equal(p.status().textContent, "The server did not answer");
    p.fire("slai:ready");
    assert.equal(p.panel.removed, false);
  });
});
