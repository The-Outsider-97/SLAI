/** DOM helpers shared by every SLAI application. Elements are built with createElement and textContent only: no innerHTML anywhere. */

export type Child = Node | string | number | null | undefined | false | Child[];

export interface Props {
  class?: string;
  text?: string;
  /** Inline presentation without `style=""` (the CSP forbids it): each entry goes through style.setProperty. */
  css?: Record<string, string>;
  data?: Record<string, string | undefined>;
  /** Attributes; undefined, null and false are skipped, true becomes the empty string. */
  attrs?: Record<string, string | number | boolean | null | undefined>;
  on?: { [K in keyof HTMLElementEventMap]?: (event: HTMLElementEventMap[K]) => void };
  [name: string]: unknown;
}

const SVG_NS = "http://www.w3.org/2000/svg";
const PROPERTY_FIRST = new Set(["value", "checked", "disabled", "hidden", "selected", "open", "indeterminate", "readOnly", "required", "min", "max", "step", "type", "placeholder"]);

function append(parent: Node, child: Child): void {
  if (child === null || child === undefined || child === false) return;
  if (Array.isArray(child)) { for (const item of child) append(parent, item); return; }
  parent.appendChild(typeof child === "object" ? child : document.createTextNode(String(child)));
}

function apply(el: Element, props: Props | null | undefined): void {
  if (!props) return;
  for (const [name, value] of Object.entries(props)) {
    if (value === undefined || value === null) continue;
    if (name === "class") el.setAttribute("class", String(value));
    else if (name === "text") el.textContent = String(value);
    else if (name === "css") for (const [prop, v] of Object.entries(value as Record<string, string>)) (el as HTMLElement).style.setProperty(prop, v);
    else if (name === "data") for (const [key, v] of Object.entries(value as Record<string, string | undefined>)) { if (v !== undefined) (el as HTMLElement).dataset[key] = v; }
    else if (name === "attrs") for (const [attr, v] of Object.entries(value as Record<string, unknown>)) {
      if (v === undefined || v === null || v === false) continue;
      el.setAttribute(attr, v === true ? "" : String(v));
    } else if (name === "on") for (const [type, handler] of Object.entries(value as Record<string, EventListener>)) el.addEventListener(type, handler);
    else if (PROPERTY_FIRST.has(name) || name in el) (el as unknown as Record<string, unknown>)[name] = value;
    else el.setAttribute(name, String(value));
  }
}

/** Build an HTML element. `h("button", {class: "btn", on: {click}}, "Save")`. */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, props?: Props | null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  apply(el, props);
  for (const child of children) append(el, child);
  return el;
}

/** Build an SVG element (namespace-aware). Attributes go in `attrs`. */
export function s<K extends keyof SVGElementTagNameMap>(tag: K, props?: Props | null, ...children: Child[]): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  apply(el, props);
  for (const child of children) append(el, child);
  return el;
}

export function clear(node: Node): void { while (node.firstChild) node.removeChild(node.firstChild); }

export function replace(node: Node, ...children: Child[]): void {
  clear(node);
  for (const child of children) append(node, child);
}

/** querySelector that throws when the element is missing, so a typo fails loudly instead of at the first use. */
export function $<T extends Element = HTMLElement>(selector: string, root: ParentNode = document): T {
  const found = root.querySelector<T>(selector);
  if (!found) throw new Error(`Missing element: ${selector}`);
  return found;
}

export function $$<T extends Element = HTMLElement>(selector: string, root: ParentNode = document): T[] {
  return Array.from(root.querySelectorAll<T>(selector));
}

let counter = 0;
/** A document-unique id for label/aria wiring. */
export function nextId(prefix = "id"): string { counter += 1; return `${prefix}-${counter}`; }

/** Wrap a control in the shared `.field` layout: label above, optional hint and error below, aria wired up. */
export function labelled(label: string, control: HTMLElement, options: { hint?: string; error?: string; className?: string } = {}): HTMLElement {
  const id = control.id || nextId("f");
  control.id = id;
  const wrap = h("div", { class: `field${options.className ? ` ${options.className}` : ""}` }, h("label", { attrs: { for: id }, text: label }), control);
  if (options.hint) {
    const hint = h("span", { class: "hint", text: options.hint, attrs: { id: `${id}-hint` } });
    control.setAttribute("aria-describedby", hint.id);
    wrap.appendChild(hint);
  }
  if (options.error) {
    const error = h("span", { class: "error", text: options.error, attrs: { id: `${id}-error`, role: "alert" } });
    control.setAttribute("aria-invalid", "true");
    control.setAttribute("aria-describedby", error.id);
    wrap.appendChild(error);
  }
  return wrap;
}

let live: HTMLElement | null = null;
/** Say something to screen readers without moving focus. */
export function announce(message: string, politeness: "polite" | "assertive" = "polite"): void {
  if (!live || !live.isConnected) {
    live = h("div", { class: "visually-hidden", attrs: { "aria-live": politeness, "aria-atomic": "true", role: politeness === "assertive" ? "alert" : "status" } });
    document.body.appendChild(live);
  }
  live.setAttribute("aria-live", politeness);
  live.textContent = "";
  window.setTimeout(() => { if (live) live.textContent = message; }, 30);
}

/** Focusable descendants in tab order, for focus traps and roving menus. */
export function focusable(root: ParentNode): HTMLElement[] {
  return $$<HTMLElement>('a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])', root)
    .filter((el) => !el.hidden && el.getClientRects().length > 0);
}
