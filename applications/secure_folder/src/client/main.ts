/** SLAI Secure Folder client. The vault token exists only in JS memory, never localStorage/cookies. */
import { $, announce, applyAppearance, appearanceControls, confirmDialog, formatBytes, glyph, h,
         labelled, markerHeader, openDialog, openMenu, promptDialog, registerGlyph, replace,
         saveBlob, toast, type Appearance, type DialogHandle } from "./kit.js";

type Kind = "file" | "folder" | "note";
interface Entry {
  id: string; kind: Kind; parent: string | null; name: string; size: number; mime: string;
  favorite: boolean; trashed: boolean; created: string; updated: string;
  chunks?: number; sha256?: string;
}
type View = "vault" | "favorites" | "notes" | "photos" | "trash" | "security";
interface Status { initialized: boolean; locked: boolean; cipher: string; kdf: string; idleSeconds: number; }
interface UploadTicket { uploadId: string; chunkSize: number }

const app = $("#app");
const marker = markerHeader("secure_folder");
let vaultToken: string | null = null;
let entries: Entry[] = [];
let folder: string | null = null;
let view: View = "vault";
let searchText = "";
let lastActivity = Date.now();
let busy = false;
let filePickerActive = false;
let generation = 0;
let locking = false;
let contentHost: HTMLElement | null = null;
let navHost: HTMLElement | null = null;
const activeDialogs = new Set<DialogHandle<unknown>>();
const previewUrls = new Set<string>();

registerGlyph("folder", ["M3.5 6.5A1.5 1.5 0 0 1 5 5h5l2.2 2.3H19a1.5 1.5 0 0 1 1.5 1.5v9.5A1.5 1.5 0 0 1 19 20H5a1.5 1.5 0 0 1-1.5-1.5z"]);
registerGlyph("shield", ["M12 2.5l8 3v5.8c0 5.1-3.1 8.5-8 10.2-4.9-1.7-8-5.1-8-10.2V5.5z", "M9 12.1l2.1 2.1 4.4-4.5"]);
registerGlyph("file", ["M7 3h7l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z", "M14 3v5h5"]);
registerGlyph("image", ["M4 5h16v14H4z", "M7 15l3.5-4 3 3 2-2.5L19 16", "M8 8h.01"]);

const appearanceKey = "slai-secure-folder-appearance";
function savedAppearance(): Appearance {
  try { return JSON.parse(localStorage.getItem(appearanceKey) ?? "null") as Appearance ?? { theme: "system", scale: 1, motion: "system" }; }
  catch { return { theme: "system", scale: 1, motion: "system" }; }
}
let appearance = savedAppearance();
applyAppearance(appearance, "soft");

function track<T>(dialog: DialogHandle<T>): DialogHandle<T> {
  activeDialogs.add(dialog as DialogHandle<unknown>);
  void dialog.result.finally(() => activeDialogs.delete(dialog as DialogHandle<unknown>));
  return dialog;
}

function failure(error: unknown): void {
  toast(error instanceof Error ? error.message : "Something went wrong.", { kind: "error" });
}

async function api<T>(method: string, path: string, body?: unknown, forcedToken?: string | null): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json", [marker]: "1" };
  const token = forcedToken === undefined ? vaultToken : forcedToken;
  if (token) headers["X-Vault-Session"] = token;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response: Response;
  try {
    response = await fetch(`/api${path}`, { method, headers, credentials: "same-origin", cache: "no-store",
                                             body: body === undefined ? undefined : JSON.stringify(body) });
  } catch { throw new Error("Cannot reach the local Secure Folder service."); }
  const envelope = await response.json().catch(() => null) as { ok?: boolean; data?: T; error?: { message?: string } } | null;
  if (response.status === 401 && token && path !== "/unlock") void localLock(false);
  if (!response.ok || !envelope?.ok) throw new Error(envelope?.error?.message ?? `Request failed (${response.status}).`);
  lastActivity = Date.now();
  return envelope.data as T;
}

async function chunk(entry: Entry, index: number): Promise<Uint8Array> {
  if (!vaultToken) throw new Error("The vault is locked.");
  const response = await fetch(`/api/files/${entry.id}/chunks/${index}`, {
    method: "GET", cache: "no-store", credentials: "same-origin",
    headers: { [marker]: "1", "X-Vault-Session": vaultToken },
  });
  if (response.status === 401) { void localLock(false); throw new Error("Your session has expired."); }
  if (!response.ok) throw new Error("The encrypted file could not be opened.");
  lastActivity = Date.now();
  return new Uint8Array(await response.arrayBuffer());
}

function clearSensitiveUi(): void {
  for (const dlg of Array.from(activeDialogs)) dlg.close(null);
  for (const url of previewUrls) URL.revokeObjectURL(url);
  previewUrls.clear();
  entries = [];
  folder = null;
  searchText = "";
  contentHost = null;
  navHost = null;
  replace(app, h("div", { class: "boot", text: "Locking your private space…" }));
}

async function localLock(sendRequest = true): Promise<void> {
  if (locking) return;
  locking = true;
  const old = vaultToken;
  vaultToken = null;
  generation += 1;
  clearSensitiveUi();
  if (old && sendRequest) {
    try { await api("POST", "/lock", {}, old); } catch { /* server also expires its volatile session */ }
  }
  try { await showAuth(); } catch (err) { failure(err); }
  finally { locking = false; }
}

const brand = (): HTMLElement => h("div", { class: "brand" },
  h("span", { class: "brand-mark" }, glyph("lock")),
  h("span", { class: "brand-label" }, h("strong", { text: "Secure Folder" }), h("small", { text: "SLAI · Private space" })));

function authForm(initial: boolean): HTMLElement {
  const pw = h("input", { type: "password", attrs: { autocomplete: initial ? "new-password" : "current-password", required: true,
                                          minlength: initial ? 12 : undefined, maxlength: 1024 }, placeholder: "Enter your passphrase" });
  const confirm = h("input", { type: "password", attrs: { autocomplete: "new-password", minlength: 12, maxlength: 1024 },
                                placeholder: "Confirm your passphrase" });
  const message = h("p", { class: "auth-error", attrs: { role: "alert" } });
  const submit = h("button", { class: "btn primary auth-submit", type: "submit" },
                   glyph(initial ? "shield" : "lock"), initial ? "Create encrypted vault" : "Unlock secure folder");
  const form = h("form", { class: "auth-form" },
    labelled(initial ? "Create a strong password or passphrase" : "Password or passphrase", pw,
             initial ? { hint: "At least 12 characters. Short PINs are not supported on this platform." } : {}),
    initial ? labelled("Confirm passphrase", confirm) : null,
    initial ? h("p", { class: "security-note", text: "No recovery backdoor exists. Losing your passphrase may make your data unrecoverable." }) : null,
    message, submit);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    submit.disabled = true;
    message.textContent = "";
    try {
      if (initial) {
        await api("POST", "/setup", { password: pw.value, confirmation: confirm.value });
        pw.value = ""; confirm.value = "";
        toast("Vault created. Enter your passphrase to unlock it.", { kind: "ok" });
        await showAuth();
      } else {
        const result = await api<{ sessionToken: string }>("POST", "/unlock", { password: pw.value });
        pw.value = "";
        vaultToken = result.sessionToken;
        generation += 1;
        await refresh();
        showWorkspace();
      }
    } catch (err) { message.textContent = err instanceof Error ? err.message : "Cannot unlock the vault."; }
    finally { submit.disabled = false; }
  });
  window.setTimeout(() => pw.focus(), 0);
  return form;
}

async function showAuth(): Promise<void> {
  const status = await api<Status>("GET", "/status", undefined, null);
  replace(app, h("div", { class: "auth-screen" },
    h("div", { class: "auth-top" }, brand(), h("span", { class: "private-chip" }, glyph("shield"), "Local and private")),
    h("section", { class: "auth-card" },
      h("div", { class: "auth-hero" }, glyph("lock")),
      h("p", { class: "eyebrow", text: status.initialized ? "Welcome back" : "Your private space begins here" }),
      h("h1", { text: status.initialized ? "A safe place for what matters." : "Protect what's personal." }),
      h("p", { class: "auth-lead", text: status.initialized
        ? "Enter your passphrase every time you unlock. Your files stay encrypted at rest."
        : "Store documents, photos and private notes behind a passphrase-derived encryption key." }),
      authForm(!status.initialized)),
    h("p", { class: "auth-footer" }, glyph("shield"), "AES-256-GCM encryption · Offline-first · No AI access"),
  ));
}

async function refresh(): Promise<void> {
  const result = await api<{ entries: Entry[] }>("GET", "/items");
  entries = result.entries;
  if (folder && !entries.some((e) => e.id === folder && !e.trashed)) folder = null;
  if (contentHost) renderPage();
}

function navigate(next: View, nextFolder: string | null = null): void {
  view = next;
  folder = nextFolder;
  searchText = "";
  const search = document.querySelector<HTMLInputElement>(".global-search");
  if (search) search.value = "";
  renderPage();
  renderNav();
  document.querySelector(".sf-shell")?.classList.remove("sidebar-open");
}

const NAV: { id: View; name: string; icon: string }[] = [
  { id: "vault", name: "My vault", icon: "folder" },
  { id: "favorites", name: "Favorites", icon: "star" },
  { id: "notes", name: "Private notes", icon: "note" },
  { id: "photos", name: "Photos", icon: "image" },
  { id: "trash", name: "Trash", icon: "trash" },
  { id: "security", name: "Security & privacy", icon: "shield" },
];

function renderNav(): void {
  if (!navHost) return;
  replace(navHost, h("div", { class: "side-brand" }, brand(),
                       h("span", { class: "encrypted-tag" }, glyph("check-circle"), "Encrypted")),
    h("div", { class: "nav-label", text: "WORKSPACE" }),
    h("nav", { class: "side-links", attrs: { "aria-label": "Secure Folder navigation" } },
      NAV.map((item) => h("button", { class: `side-link ${view === item.id ? "active" : ""}`, type: "button",
          attrs: { "aria-current": view === item.id ? "page" : undefined },
          on: { click: () => navigate(item.id) } }, glyph(item.icon), item.name))),
    h("div", { class: "side-bottom" },
      h("div", { class: "side-privacy" }, glyph("shield"), h("div", {},
        h("strong", { text: "Private by default" }), h("p", { text: "Only your passphrase unlocks your data." }))),
      h("button", { class: "side-lock", type: "button", on: { click: () => void localLock() } }, glyph("lock"), "Lock vault")));
}

function showWorkspace(): void {
  const search = h("input", { class: "global-search", type: "search", placeholder: "Search encrypted items…",
                              attrs: { "aria-label": "Search your vault", autocomplete: "off" },
                              on: { input: () => { searchText = search.value.trim().toLowerCase(); renderPage(); } } });
  navHost = h("aside", { class: "sf-sidebar" });
  contentHost = h("main", { class: "sf-content", id: "main", attrs: { tabindex: "-1" } });
  const shell = h("div", { class: "sf-shell" }, navHost,
    h("div", { class: "sf-main-col" },
      h("header", { class: "sf-topbar" },
        h("button", { class: "iconbtn menu-button", type: "button", attrs: { "aria-label": "Toggle navigation" },
          on: { click: () => shell.classList.toggle("sidebar-open") } }, glyph("menu")),
        h("div", { class: "search-field" }, glyph("search"), search),
        h("span", { class: "local-badge" }, h("span", { class: "status-dot" }), "Local vault"),
        h("button", { class: "btn top-lock", type: "button", on: { click: () => void localLock() } }, glyph("lock"), "Lock")),
      contentHost,
      h("nav", { class: "mobile-tabs", attrs: { "aria-label": "Mobile navigation" } },
        NAV.slice(0, 5).map((item) => h("button", { type: "button", on: { click: () => navigate(item.id) } },
          glyph(item.icon), h("span", { text: item.name === "Private notes" ? "Notes" : item.name === "My vault" ? "Files" : item.name }))))));
  replace(app, shell);
  renderNav();
  renderPage();
}

function breadcrumbs(): HTMLElement {
  const row = h("div", { class: "breadcrumbs" },
    h("button", { class: "textbtn", type: "button", on: { click: () => navigate("vault") } }, "My vault"));
  const parents: Entry[] = [];
  let current = folder;
  while (current) {
    const e = entries.find((item) => item.id === current);
    if (!e || parents.length > 100) break;
    parents.unshift(e); current = e.parent;
  }
  for (const e of parents) row.append(glyph("chevron-right"),
    h("button", { class: "textbtn", type: "button", on: { click: () => navigate("vault", e.id) } }, e.name));
  return row;
}

function filtered(): Entry[] {
  let result = entries.filter((item) => view === "trash" ? item.trashed : !item.trashed);
  if (view === "vault") result = result.filter((item) => item.parent === folder);
  else if (view === "favorites") result = result.filter((item) => item.favorite);
  else if (view === "notes") result = result.filter((item) => item.kind === "note");
  else if (view === "photos") result = result.filter((item) => item.kind === "file" && /^image\//.test(item.mime));
  if (searchText) result = result.filter((item) => item.name.toLowerCase().includes(searchText));
  return result.sort((a, b) => Number(b.kind === "folder") - Number(a.kind === "folder") || a.name.localeCompare(b.name));
}

function iconFor(e: Entry): string {
  if (e.kind === "folder") return "folder";
  if (e.kind === "note") return "note";
  if (/^image\//.test(e.mime)) return "image";
  return "file";
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" }).format(new Date(value));
}

function itemMenu(anchor: HTMLElement, e: Entry): void {
  openMenu(anchor, e.trashed ? [
    { label: "Restore", icon: "undo", run: () => void updateWith("POST", `/items/${e.id}/restore`) },
    { label: "Delete permanently", icon: "trash", danger: true, run: () => void purge(e) },
  ] : [
    { label: "Open", icon: "file", run: () => void openEntry(e) },
    { label: e.favorite ? "Remove favorite" : "Add favorite", icon: "star", run: () => void updateWith("PATCH", `/items/${e.id}`, { favorite: !e.favorite }) },
    { label: "Rename", icon: "edit", run: () => void renameEntry(e) },
    { label: "Move", icon: "folder", run: () => void moveEntry(e) },
    { label: "Export", icon: "download", disabled: e.kind === "folder", run: () => void downloadEntry(e) },
    "separator",
    { label: "Move to Trash", icon: "trash", danger: true, run: () => void deleteEntry(e) },
  ], { label: `Actions for ${e.name}` });
}

function fileCard(e: Entry): HTMLElement {
  const menu = h("button", { class: "iconbtn tiny", type: "button", attrs: { "aria-label": `Actions for ${e.name}` },
    on: { click: () => itemMenu(menu, e) } }, glyph("more-vertical"));
  const action = h("button", { class: "item-open", type: "button", on: { click: () => void openEntry(e) } },
    h("span", { class: `item-symbol ${e.kind}` }, glyph(iconFor(e))),
    h("span", { class: "item-text" }, h("strong", { text: e.name }),
      h("small", { text: e.kind === "folder" ? "Folder" : `${formatBytes(e.size)} · ${formatDate(e.updated)}` })));
  return h("div", { class: "item-card" }, action,
    e.favorite ? h("span", { class: "favorite-mark", attrs: { title: "Favorite" } }, glyph("star")) : null,
    menu);
}

function renderPage(): void {
  if (!contentHost) return;
  if (view === "security") { replace(contentHost, securityPage()); return; }
  const labels: Record<View, string> = { vault: folder ? entries.find((e) => e.id === folder)?.name ?? "Folder" : "My vault",
    favorites: "Favorites", notes: "Private notes", photos: "Photos", trash: "Trash", security: "Security & privacy" };
  const chosen = filtered();
  const allActive = entries.filter((e) => !e.trashed);
  const totalBytes = allActive.reduce((sum, e) => sum + (e.kind === "folder" ? 0 : e.size), 0);
  replace(contentHost,
    h("div", { class: "page-intro" },
      h("div", {}, h("p", { class: "eyebrow", text: "YOUR PRIVATE WORKSPACE" }),
        h("h1", { text: labels[view] }),
        h("p", { class: "subtitle", text: view === "trash" ? "Deleted items stay encrypted until permanently removed." :
           "Organize everything important, all in one protected place." })),
      h("div", { class: "quick-actions" },
        view === "vault" ? h("button", { class: "btn", type: "button", on: { click: () => void newFolder() } }, glyph("folder"), "New folder") : null,
        view === "vault" || view === "notes" ? h("button", { class: "btn", type: "button", on: { click: () => void noteEditor() } }, glyph("note"), "New note") : null,
        view === "vault" ? h("button", { class: "btn primary", type: "button", on: { click: chooseFiles } }, glyph("upload"), "Add files") : null)),
    view === "vault" && folder ? breadcrumbs() : null,
    view === "vault" && !folder ? h("div", { class: "summary-grid" },
      h("div", { class: "summary-card" }, h("span", { class: "summary-icon" }, glyph("folder")),
        h("div", {}, h("strong", { text: String(allActive.length) }), h("span", { text: "Protected items" }))),
      h("div", { class: "summary-card" }, h("span", { class: "summary-icon" }, glyph("database")),
        h("div", {}, h("strong", { text: formatBytes(totalBytes) }), h("span", { text: "Encrypted content" }))),
      h("div", { class: "summary-card" }, h("span", { class: "summary-icon" }, glyph("shield")),
        h("div", {}, h("strong", { text: "Protected" }), h("span", { text: "Local-only access" })))) : null,
    h("section", { class: "files-section card" },
      h("div", { class: "section-heading" }, h("h2", { text: `Files & folders` }), h("span", { class: "count-tag", text: `${chosen.length} items` })),
      chosen.length ? h("div", { class: "items-grid" }, chosen.map(fileCard)) :
        h("div", { class: "empty-state" }, glyph(view === "trash" ? "trash" : "folder"),
          h("h3", { text: searchText ? "No matches found" : view === "trash" ? "Trash is empty" : "Nothing here yet" }),
          h("p", { text: searchText ? "Try another search term." : "Add files or create a folder to get started." }))));
}

async function updateWith(method: string, path: string, body?: unknown): Promise<void> {
  try { await api(method, path, body); await refresh(); toast("Changes saved.", { kind: "ok" }); }
  catch (err) { failure(err); }
}

async function newFolder(): Promise<void> {
  const name = await promptDialog({ title: "New folder", label: "Folder name", maxLength: 160 });
  if (name) await updateWith("POST", "/folders", { name, parent: folder });
}

async function renameEntry(e: Entry): Promise<void> {
  const name = await promptDialog({ title: "Rename item", label: "Name", value: e.name, maxLength: 160 });
  if (name && name !== e.name) await updateWith("PATCH", `/items/${e.id}`, { name });
}

async function moveEntry(e: Entry): Promise<void> {
  const folders = entries.filter((item) => item.kind === "folder" && !item.trashed && item.id !== e.id);
  const select = h("select", {}, h("option", { value: "root", text: "My vault (root)" }),
    folders.map((f) => h("option", { value: f.id, text: f.name })));
  select.value = e.parent ?? "root";
  const result = await track(openDialog<string>({ title: `Move ${e.name}`, size: "narrow", body: labelled("Destination", select),
    actions: [{ label: "Move", kind: "primary", run: () => select.value }] })).result;
  if (result !== null) await updateWith("PATCH", `/items/${e.id}`, { parent: result });
}

async function deleteEntry(e: Entry): Promise<void> {
  if (await confirmDialog({ title: "Move to Trash?", message: `${e.name} will remain encrypted and can be restored.`,
                            confirmLabel: "Move to Trash", danger: true })) await updateWith("DELETE", `/items/${e.id}`);
}

async function passwordPrompt(title: string, description: string): Promise<string | null> {
  const input = h("input", { type: "password", attrs: { autocomplete: "current-password" }, placeholder: "Your passphrase" });
  const result = await track(openDialog<string>({ title, size: "narrow",
    body: h("div", {}, h("p", { class: "muted", text: description }), labelled("Password or passphrase", input)),
    actions: [{ label: "Continue", kind: "primary", run: () => input.value || undefined }] })).result;
  input.value = "";
  return result;
}

async function purge(e: Entry): Promise<void> {
  const password = await passwordPrompt("Permanently delete?", "This removes the encrypted file from the vault. Recovery is not guaranteed.");
  if (!password) return;
  if (!await confirmDialog({ title: "Confirm permanent deletion", message: `Permanently remove ${e.name}?`,
                            danger: true, confirmLabel: "Delete permanently" })) return;
  await updateWith("POST", `/items/${e.id}/purge`, { password });
}

function chooseFiles(): void {
  filePickerActive = true;
  const input = h("input", { type: "file", attrs: { multiple: true }, on: { change: () => {
    filePickerActive = false;
    if (input.files) void importFiles(Array.from(input.files)); input.remove();
  } } });
  input.addEventListener("cancel", () => { filePickerActive = false; input.remove(); });
  input.hidden = true;
  document.body.append(input);
  input.click();
}

function base64(bytes: Uint8Array): string {
  let value = "";
  for (let i = 0; i < bytes.length; i += 16384) value += String.fromCharCode(...bytes.subarray(i, i + 16384));
  return btoa(value);
}

async function importFiles(files: File[]): Promise<void> {
  if (busy) { toast("Finish the current import first."); return; }
  busy = true;
  const progress = h("div", { class: "progress-panel", attrs: { role: "status" } }, "Preparing file import…");
  document.body.append(progress);
  try {
    for (const file of files) {
      const ticket = await api<UploadTicket>("POST", "/uploads", { name: file.name, size: file.size, mime: file.type, parent: folder });
      try {
        for (let offset = 0, index = 0; offset < file.size; index += 1) {
          const part = new Uint8Array(await file.slice(offset, offset + ticket.chunkSize).arrayBuffer());
          await api("POST", `/uploads/${ticket.uploadId}/chunks`, { index, data: base64(part) });
          offset += part.byteLength;
          progress.textContent = `Encrypting ${file.name} · ${Math.round(100 * offset / file.size)}%`;
        }
        await api("POST", `/uploads/${ticket.uploadId}/finish`, {});
      } catch (err) { await api("POST", `/uploads/${ticket.uploadId}/cancel`, {}).catch(() => undefined); throw err; }
    }
    await refresh();
    toast(`${files.length} file${files.length === 1 ? "" : "s"} encrypted and saved.`, { kind: "ok" });
  } catch (err) { failure(err); }
  finally { progress.remove(); busy = false; }
}

async function readWhole(e: Entry): Promise<Blob> {
  if (e.size > 96 * 1024 * 1024) throw new Error("This browser limits previews and exports to 96 MiB. Your encrypted file remains safe.");
  const parts: Uint8Array<ArrayBuffer>[] = [];
  for (let i = 0; i < (e.chunks ?? 0); i += 1) {
    const part = await chunk(e, i);
    parts.push(new Uint8Array(part));
  }
  const blob = new Blob(parts, { type: "application/octet-stream" });
  if (blob.size !== e.size) throw new Error("The file is incomplete.");
  if (e.sha256) {
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", await blob.arrayBuffer()));
    const hex = Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (hex !== e.sha256) throw new Error("The file failed its integrity check.");
  }
  return blob;
}

async function downloadEntry(e: Entry): Promise<void> {
  if (e.kind === "folder") return;
  const password = await passwordPrompt("Confirm secure export", "Exported files are no longer protected by Secure Folder encryption.");
  if (!password) return;
  try {
    await api("POST", "/reauth", { password });
    const blob = await readWhole(e);
    saveBlob(blob, e.name);
    toast("Export prepared. Keep the downloaded copy private.", { kind: "ok" });
  } catch (err) { failure(err); }
}

async function openEntry(e: Entry): Promise<void> {
  const openedGeneration = generation;
  if (e.trashed) return;
  if (e.kind === "folder") { navigate("vault", e.id); return; }
  if (e.kind === "note") { await noteEditor(e); return; }
  try {
    if (e.size > 24 * 1024 * 1024) {
      toast("Preview is limited to 24 MiB. Use Export for larger files."); return;
    }
    const blob = await readWhole(e);
    if (!vaultToken || generation !== openedGeneration) return;
    const mime = e.mime;
    let body: HTMLElement;
    let previewUrl: string | null = null;
    if (/^image\/(?:png|jpeg|gif|webp|avif)$/.test(mime)) {
      previewUrl = URL.createObjectURL(new Blob([blob], { type: mime }));
      previewUrls.add(previewUrl);
      body = h("img", { class: "file-image", src: previewUrl, alt: e.name });
    } else if (mime === "text/plain" || e.name.toLowerCase().endsWith(".txt")) {
      body = h("pre", { class: "file-text", text: await blob.text() });
    } else {
      body = h("div", { class: "preview-unsupported" }, glyph("file"), h("h3", { text: "Preview unavailable" }),
               h("p", { text: "This file remains encrypted. Use Export to open it in an appropriate application." }));
    }
    const handle = track(openDialog({ title: e.name, size: "wide", body,
      actions: [{ label: "Export", kind: "primary", run: async () => { await downloadEntry(e); return null; } }] }));
    const urlToClean = previewUrl;
    void handle.result.finally(() => { if (urlToClean) { URL.revokeObjectURL(urlToClean); previewUrls.delete(urlToClean); } });
  } catch (err) { failure(err); }
}

async function noteEditor(existing?: Entry): Promise<void> {
  const openedGeneration = generation;
  let currentText = "";
  if (existing) {
    try { currentText = (await api<{ text: string }>("GET", `/notes/${existing.id}`)).text; }
    catch (err) { failure(err); return; }
  }
  if (!vaultToken || generation !== openedGeneration) return;
  const title = h("input", { type: "text", value: existing?.name ?? "Untitled note", attrs: { maxlength: 160 } });
  const area = h("textarea", { class: "note-input", value: currentText, attrs: { maxlength: 200000 },
                               placeholder: "Start writing. Everything is encrypted before storage…" });
  const result = await track(openDialog<true>({ title: existing ? "Edit private note" : "New private note", size: "wide",
    body: h("div", {}, labelled("Title", title), labelled("Encrypted note", area)),
    actions: [{ label: "Save note", kind: "primary", run: async () => {
      if (!title.value.trim()) { toast("Enter a note title.", { kind: "error" }); return undefined; }
      if (existing) {
        await api("PUT", `/notes/${existing.id}`, { content: area.value });
        if (title.value !== existing.name) await api("PATCH", `/items/${existing.id}`, { name: title.value });
      } else await api("POST", "/notes", { name: title.value, content: area.value, parent: folder });
      return true;
    } }] })).result;
  if (result) { await refresh(); toast("Encrypted note saved.", { kind: "ok" }); }
}

function securityPage(): HTMLElement {
  return h("div", { class: "security-page" },
    h("div", { class: "page-intro" }, h("div", {}, h("p", { class: "eyebrow", text: "CONTROL YOUR PRIVACY" }),
      h("h1", { text: "Security & privacy" }), h("p", { class: "subtitle", text: "Protection settings for your local encrypted vault." }))),
    h("div", { class: "security-grid" },
      h("section", { class: "card security-card" }, h("div", { class: "security-card-head" }, glyph("shield"),
        h("h2", { text: "Vault protection" })),
        h("div", { class: "security-setting" }, h("span", { text: "Encryption" }), h("strong", { text: "AES-256-GCM" })),
        h("div", { class: "security-setting" }, h("span", { text: "Password derivation" }), h("strong", { text: "scrypt" })),
        h("div", { class: "security-setting" }, h("span", { text: "Inactivity lock" }), h("strong", { text: "5 minutes" })),
        h("div", { class: "security-setting" }, h("span", { text: "Cloud sync" }), h("strong", { text: "Off" })),
        h("div", { class: "security-setting" }, h("span", { text: "AI access" }), h("strong", { text: "Disabled" }))),
      h("section", { class: "card security-card" }, h("div", { class: "security-card-head" }, glyph("lock"),
        h("h2", { text: "Authentication" })),
        h("p", { class: "muted", text: "Every new unlock requires your passphrase. Locking revokes the current session." }),
        h("button", { class: "btn", type: "button", on: { click: () => void changePassword() } }, glyph("edit"), "Change passphrase"),
        h("button", { class: "btn", type: "button", on: { click: () => void localLock() } }, glyph("lock"), "Lock now")),
      h("section", { class: "card security-card" }, h("div", { class: "security-card-head" }, glyph("palette"),
        h("h2", { text: "Appearance" })),
        appearanceControls(appearance, (next) => {
          appearance = next;
          try { localStorage.setItem(appearanceKey, JSON.stringify(next)); } catch { /* appearance only */ }
        })),
      h("section", { class: "card security-card" }, h("div", { class: "security-card-head" }, glyph("info"),
        h("h2", { text: "Privacy boundaries" })),
        h("p", { class: "muted", text: "Vault data is not sent to SLAI agents, external AI services, telemetry, or cloud storage." }),
        h("p", { class: "muted", text: "This browser interface cannot guarantee screenshot prevention or protection against malware with access to your unlocked device." }))));
}

async function changePassword(): Promise<void> {
  const oldPassword = h("input", { type: "password", attrs: { autocomplete: "current-password" } });
  const newPassword = h("input", { type: "password", attrs: { autocomplete: "new-password", minlength: 12 } });
  const confirmation = h("input", { type: "password", attrs: { autocomplete: "new-password" } });
  const result = await track(openDialog<true>({ title: "Change vault passphrase", size: "narrow",
    body: h("div", {}, labelled("Current passphrase", oldPassword), labelled("New passphrase", newPassword),
                     labelled("Confirm new passphrase", confirmation)),
    actions: [{ label: "Change passphrase", kind: "primary", run: async () => {
      if (newPassword.value !== confirmation.value) { toast("New passphrases do not match.", { kind: "error" }); return undefined; }
      await api("POST", "/password/change", { oldPassword: oldPassword.value, newPassword: newPassword.value });
      return true;
    } }] })).result;
  oldPassword.value = ""; newPassword.value = ""; confirmation.value = "";
  if (result) { toast("Passphrase changed. Unlock again with the new one.", { kind: "ok" }); await localLock(false); }
}

for (const name of ["click", "keydown", "pointerdown"] as const) {
  document.addEventListener(name, () => { if (vaultToken) lastActivity = Date.now(); }, { passive: true });
}
window.setInterval(() => { if (vaultToken && Date.now() - lastActivity > 5 * 60_000) void localLock(); }, 10_000);
document.addEventListener("visibilitychange", () => { if (document.hidden && vaultToken && !filePickerActive) void localLock(); });
window.addEventListener("pagehide", () => {
  if (vaultToken) {
    const old = vaultToken;
    vaultToken = null;
    void api("POST", "/lock", {}, old).catch(() => undefined);
  }
});

void showAuth().catch((error: unknown) => {
  replace(app, h("div", { class: "auth-screen" }, h("section", { class: "auth-card" },
    h("h1", { text: "Secure Folder could not start" }),
    h("p", { text: error instanceof Error ? error.message : "The local service is unavailable." }))));
});
