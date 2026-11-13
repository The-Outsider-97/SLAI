/** Shared interface behaviour: toasts, modal dialogs, confirm/prompt, popup menus, clipboard and file saving.
 *  Everything here is built on the platform (<dialog>, fixed positioning) and styled by slai-ui.css. */
import { announce, focusable, h, labelled, nextId, replace, type Child } from "./dom.js";
import { glyph } from "./icons.js";

/* ------------------------------------------------------------------ toasts */

export interface ToastOptions {
  kind?: "info" | "ok" | "error";
  /** 0 keeps the toast until dismissed. Errors stay longer by default. */
  timeoutMs?: number;
  action?: { label: string; run: () => void };
}

let toastHost: HTMLElement | null = null;

export function toast(message: string, options: ToastOptions = {}): () => void {
  if (!toastHost || !toastHost.isConnected) {
    toastHost = h("div", { class: "toasts", attrs: { role: "region", "aria-label": "Notifications", popover: "manual" } });
    document.body.appendChild(toastHost);
  }
  const kind = options.kind ?? "info";
  const host = toastHost;
  let timer: number | undefined;
  const dismiss = (): void => { if (timer !== undefined) window.clearTimeout(timer); el.remove(); };
  const el = h("div", { class: `toast${kind === "info" ? "" : ` ${kind}`}`, attrs: { role: kind === "error" ? "alert" : "status" } },
    h("span", { class: "grow", text: message }),
    options.action ? h("button", { class: "textbtn act", type: "button", text: options.action.label, on: { click: () => { options.action?.run(); dismiss(); } } }) : null,
    h("button", { class: "iconbtn tiny", type: "button", attrs: { "aria-label": "Dismiss" }, on: { click: dismiss } }, glyph("close")),
  );
  host.appendChild(el);
  while (host.children.length > 4) host.firstElementChild?.remove();
  // The browser's top layer, so a toast is readable above a modal dialog and its backdrop (re-shown to be topmost again).
  try { host.hidePopover?.(); host.showPopover?.(); } catch { /* an old browser: the toast stays in the page, which is fine */ }
  const ms = options.timeoutMs ?? (kind === "error" ? 9000 : options.action ? 8000 : 4500);
  if (ms > 0) timer = window.setTimeout(dismiss, ms);
  return dismiss;
}

/* ------------------------------------------------------------------ dialogs */

export interface DialogOptions<R> {
  title: string;
  eyebrow?: string;
  size?: "narrow" | "normal" | "wide";
  /** Content of the body. Build it with `h`. */
  body: Child;
  /** Footer buttons. Return a value from `run` to close with it, `undefined` to stay open (validation failed), or `null` to close with no result. */
  actions?: Array<{ label: string; kind?: "primary" | "danger" | "plain"; run: () => R | null | undefined | Promise<R | null | undefined> }>;
  /** Hide the default Cancel button. */
  noCancel?: boolean;
  cancelLabel?: string;
  /** Element to focus on open; defaults to the first field in the body. */
  focus?: () => HTMLElement | null;
}

export interface DialogHandle<R> { el: HTMLDialogElement; result: Promise<R | null>; close(value?: R | null): void }

export function openDialog<R = true>(options: DialogOptions<R>): DialogHandle<R> {
  const titleId = nextId("dlg-title");
  const dialog = h("dialog", { class: `dialog${options.size === "wide" ? " wide" : options.size === "narrow" ? " narrow" : ""}`, attrs: { "aria-labelledby": titleId } });
  const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  let finish: (value: R | null) => void = () => undefined;
  const result = new Promise<R | null>((resolve) => { finish = resolve; });
  let settled = false;
  const close = (value: R | null = null): void => {
    if (settled) return;
    settled = true;
    if (dialog.open) dialog.close();
    dialog.remove();
    if (opener && opener.isConnected) opener.focus();
    finish(value);
  };

  const buttons: HTMLButtonElement[] = [];
  if (!options.noCancel) buttons.push(h("button", { class: "btn", type: "button", text: options.cancelLabel ?? "Cancel", on: { click: () => close(null) } }));
  for (const action of options.actions ?? []) {
    const button: HTMLButtonElement = h("button", {
      class: `btn${action.kind === "primary" ? " primary" : action.kind === "danger" ? " danger" : ""}`, type: "button", text: action.label,
      on: { click: () => {
        button.disabled = true;
        Promise.resolve(action.run()).then((value) => { if (value !== undefined) close(value); }).catch((error: unknown) => {
          toast(error instanceof Error ? error.message : "Something went wrong.", { kind: "error" });
        }).finally(() => { button.disabled = false; });
      } },
    });
    buttons.push(button);
  }

  replace(dialog,
    h("div", { class: "dialog-head" },
      h("div", {}, options.eyebrow ? h("div", { class: "eyebrow", text: options.eyebrow }) : null, h("h2", { id: titleId, text: options.title })),
      h("button", { class: "iconbtn tiny", type: "button", attrs: { "aria-label": "Close" }, on: { click: () => close(null) } }, glyph("close"))),
    h("div", { class: "dialog-body" }, options.body),
    buttons.length ? h("div", { class: "dialog-actions" }, buttons) : null,
  );

  dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(null); });
  // Clicking the backdrop (the dialog element itself, outside its box) dismisses, like Escape.
  dialog.addEventListener("mousedown", (event) => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) close(null);
  });
  // Enter in a single-line field triggers the primary action.
  dialog.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    const target = event.target;
    if (!(target instanceof HTMLInputElement) || ["checkbox", "radio", "button", "range"].includes(target.type)) return;
    const primary = dialog.querySelector<HTMLButtonElement>(".dialog-actions .btn.primary, .dialog-actions .btn.danger");
    if (primary && !primary.disabled) { event.preventDefault(); primary.click(); }
  });

  document.body.appendChild(dialog);
  dialog.showModal();
  const first = options.focus?.() ?? dialog.querySelector<HTMLElement>(".dialog-body input:not([type=hidden]), .dialog-body textarea, .dialog-body select") ?? focusable(dialog).find((el) => !el.closest(".dialog-head")) ?? null;
  first?.focus();
  return { el: dialog, result, close };
}

export function confirmDialog(options: { title: string; message: string; confirmLabel?: string; danger?: boolean; detail?: string }): Promise<boolean> {
  let dialogEl: HTMLElement | null = null;
  const handle: DialogHandle<true> = openDialog<true>({
    title: options.title, size: "narrow",
    body: [h("p", { text: options.message }), options.detail ? h("p", { class: "muted", text: options.detail }) : null],
    actions: [{ label: options.confirmLabel ?? "Confirm", kind: options.danger ? "danger" : "primary", run: () => true }],
    // Focus the confirm button for plain confirmations, but the safe Cancel button for destructive ones.
    focus: () => dialogEl?.querySelector<HTMLElement>(options.danger ? ".dialog-actions .btn:first-child" : ".dialog-actions .btn:last-child") ?? null,
  });
  dialogEl = handle.el;
  handle.el.querySelector<HTMLElement>(options.danger ? ".dialog-actions .btn:first-child" : ".dialog-actions .btn:last-child")?.focus();
  return handle.result.then((value) => value === true);
}

export function promptDialog(options: { title: string; label: string; value?: string; placeholder?: string; confirmLabel?: string; maxLength?: number; validate?: (value: string) => string | null }): Promise<string | null> {
  const input = h("input", { type: "text", value: options.value ?? "", placeholder: options.placeholder ?? "", attrs: { maxlength: options.maxLength ?? 200, autocomplete: "off" } });
  const slot = h("div", {});
  const render = (error?: string): void => { replace(slot, labelled(options.label, input, error ? { error } : {})); };
  render();
  const handle = openDialog<string>({
    title: options.title, size: "narrow", body: slot, focus: () => input,
    actions: [{ label: options.confirmLabel ?? "Save", kind: "primary", run: () => {
      const value = input.value.trim();
      const problem = value ? options.validate?.(value) ?? null : "Enter a value.";
      if (problem) { render(problem); input.focus(); announce(problem, "assertive"); return undefined; }
      return value;
    } }],
  });
  return handle.result;
}

/* ------------------------------------------------------------------ menus */

export interface MenuItem {
  label: string;
  icon?: string;
  danger?: boolean;
  disabled?: boolean;
  run: () => void;
}
export type MenuEntry = MenuItem | "separator";

let closeOpenMenu: (() => void) | null = null;

/** Open a popup menu next to `anchor`. Arrow keys move, Enter/Space choose, Escape or an outside click closes and returns focus. */
export function openMenu(anchor: HTMLElement, entries: MenuEntry[], options: { label?: string; align?: "start" | "end" } = {}): void {
  closeOpenMenu?.();
  const menu = h("div", { class: "menu", attrs: { role: "menu", "aria-label": options.label ?? "Actions" } });
  const items: HTMLButtonElement[] = [];
  for (const entry of entries) {
    if (entry === "separator") { menu.appendChild(h("div", { class: "menu-sep", attrs: { role: "separator" } })); continue; }
    const button = h("button", { class: `menu-item${entry.danger ? " danger" : ""}`, type: "button", disabled: entry.disabled === true, attrs: { role: "menuitem", tabindex: "-1" },
      on: { click: () => { close(); entry.run(); } } }, entry.icon ? glyph(entry.icon) : null, h("span", { text: entry.label }));
    items.push(button);
    menu.appendChild(button);
  }
  const close = (): void => {
    document.removeEventListener("mousedown", outside, true);
    document.removeEventListener("keydown", keys, true);
    window.removeEventListener("resize", close);
    window.removeEventListener("scroll", close, true);
    menu.remove();
    if (closeOpenMenu === close) closeOpenMenu = null;
    if (anchor.isConnected) anchor.focus();
  };
  const outside = (event: MouseEvent): void => { if (!menu.contains(event.target as Node) && !anchor.contains(event.target as Node)) close(); };
  const keys = (event: KeyboardEvent): void => {
    const enabled = items.filter((item) => !item.disabled);
    const at = enabled.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "Escape") { event.preventDefault(); close(); }
    else if (event.key === "ArrowDown") { event.preventDefault(); enabled[(at + 1) % enabled.length]?.focus(); }
    else if (event.key === "ArrowUp") { event.preventDefault(); enabled[(at - 1 + enabled.length) % enabled.length]?.focus(); }
    else if (event.key === "Home") { event.preventDefault(); enabled[0]?.focus(); }
    else if (event.key === "End") { event.preventDefault(); enabled[enabled.length - 1]?.focus(); }
    else if (event.key === "Tab") close();
  };
  // A modal <dialog> lives in the browser's top layer and covers everything outside it, so a menu opened from inside one must be inside it too.
  (anchor.closest("dialog[open]") ?? document.body).appendChild(menu);
  const a = anchor.getBoundingClientRect();
  const m = menu.getBoundingClientRect();
  const margin = 8;
  let left = options.align === "end" ? a.right - m.width : a.left;
  left = Math.max(margin, Math.min(left, window.innerWidth - m.width - margin));
  let top = a.bottom + 6;
  if (top + m.height > window.innerHeight - margin) top = Math.max(margin, a.top - m.height - 6);
  menu.style.setProperty("left", `${Math.round(left)}px`);
  menu.style.setProperty("top", `${Math.round(top)}px`);
  document.addEventListener("mousedown", outside, true);
  document.addEventListener("keydown", keys, true);
  window.addEventListener("resize", close);
  window.addEventListener("scroll", close, true);
  closeOpenMenu = close;
  items.find((item) => !item.disabled)?.focus();
}

/* ------------------------------------------------------------------ clipboard and files */

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = h("textarea", { value: text, attrs: { "aria-hidden": "true", readonly: true }, css: { position: "fixed", left: "-9999px" } });
    document.body.appendChild(area);
    area.select();
    let done = false;
    try { done = document.execCommand("copy"); } catch { done = false; }
    area.remove();
    return done;
  }
}

/** Hand a Blob to the browser's download flow. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = h("a", { href: url, download: filename, css: { display: "none" } });
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
