/** Icons. `glyph(name)` is the shared inline set (24x24 outline, currentColor); `assetIcon(url)` tints one of the shared brand
 *  icons in applications/assets with the accent colour. Applications add their own glyphs with `registerGlyph` instead of copying paths. */
import { s } from "./dom.js";

const GLYPHS: Record<string, string[]> = {
  calendar: ["M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5z", "M4 9.5h16", "M8.5 2.8v3", "M15.5 2.8v3"],
  "calendar-day": ["M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5z", "M4 9.5h16", "M8.5 2.8v3", "M15.5 2.8v3", "M9 14.2h2v2.3H9z"],
  sun: ["M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z", "M12 2.5v2", "M12 19.5v2", "M2.5 12h2", "M19.5 12h2", "M5.3 5.3l1.4 1.4", "M17.3 17.3l1.4 1.4", "M5.3 18.7l1.4-1.4", "M17.3 6.7l1.4-1.4"],
  moon: ["M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"],
  contrast: ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M12 3v18"],
  check: ["M5 12.5l4.5 4.5L19 7.5"],
  "check-circle": ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M8 12.3l3 3 5-6"],
  "check-square": ["M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5z", "M8 12.3l3 3 5-6"],
  note: ["M7 3h7l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z", "M14 3v5h5", "M8.5 13h7", "M8.5 16.5h5"],
  sticky: ["M5 4h14a1 1 0 0 1 1 1v9l-6 6H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z", "M14 20v-5a1 1 0 0 1 1-1h5"],
  bell: ["M6 16.5V11a6 6 0 0 1 12 0v5.5l1.5 1.5h-15z", "M10 20.5a2 2 0 0 0 4 0"],
  "bell-off": ["M6 16.5V11c0-.8.15-1.5.4-2.2", "M9.2 4.8A6 6 0 0 1 18 11v5.5l1.5 1.5H8", "M10 20.5a2 2 0 0 0 4 0", "M4 4l16 16"],
  plus: ["M12 5v14", "M5 12h14"],
  minus: ["M5 12h14"],
  close: ["M6 6l12 12", "M18 6L6 18"],
  "chevron-left": ["M14.5 5.5L8 12l6.5 6.5"],
  "chevron-right": ["M9.5 5.5L16 12l-6.5 6.5"],
  "chevron-down": ["M5.5 9.5L12 16l6.5-6.5"],
  "chevron-up": ["M5.5 14.5L12 8l6.5 6.5"],
  "arrow-right": ["M5 12h14", "M13 6l6 6-6 6"],
  search: ["M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13z", "M15.5 15.5L20 20"],
  more: ["M5 12h.01", "M12 12h.01", "M19 12h.01"],
  "more-vertical": ["M12 5h.01", "M12 12h.01", "M12 19h.01"],
  edit: ["M4 20h4L19 9a2.1 2.1 0 0 0-3-3L5 17z", "M14.5 7.5l3 3"],
  trash: ["M5 7h14", "M10 7V4.5h4V7", "M7 7l.8 12.2A1.5 1.5 0 0 0 9.3 20.5h5.4a1.5 1.5 0 0 0 1.5-1.3L17 7", "M10.5 11v6", "M13.5 11v6"],
  copy: ["M9 9h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1V10a1 1 0 0 1 1-1z", "M5 15V5a1 1 0 0 1 1-1h10"],
  download: ["M12 4v11", "M7.5 11l4.5 4.5 4.5-4.5", "M5 19.5h14"],
  upload: ["M12 16V5", "M7.5 9L12 4.5 16.5 9", "M5 19.5h14"],
  save: ["M5 4h11l3 3v12a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1z", "M8 4v5h7V4", "M8 20v-6h8v6"],
  refresh: ["M19 7a8 8 0 1 0 1.5 6", "M19.5 3.5V8H15"],
  undo: ["M9 7L4.5 11.5 9 16", "M5 11.5h8a6 6 0 0 1 6 6"],
  clock: ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M12 7.5V12l3 2"],
  tag: ["M3.5 12.2V4.5a1 1 0 0 1 1-1h7.7a1 1 0 0 1 .7.3l7.3 7.3a1 1 0 0 1 0 1.4l-7.4 7.4a1 1 0 0 1-1.4 0l-7.6-7.5a1 1 0 0 1-.3-.7z", "M8 8h.01"],
  pin: ["M9 4h6l-1 6 3.5 3.5v1.5h-5.5V21l-1 0v-6H5.5v-1.5L9 10z"],
  flag: ["M5 21V4", "M5 4h11l-2 4 2 4H5"],
  repeat: ["M4 11V9.5A2.5 2.5 0 0 1 6.5 7H19", "M16 4l3 3-3 3", "M20 13v1.5a2.5 2.5 0 0 1-2.5 2.5H5", "M8 20l-3-3 3-3"],
  list: ["M9 6.5h11", "M9 12h11", "M9 17.5h11", "M4.5 6.5h.01", "M4.5 12h.01", "M4.5 17.5h.01"],
  grid: ["M4 4h7v7H4z", "M13 4h7v7h-7z", "M4 13h7v7H4z", "M13 13h7v7h-7z"],
  settings: ["M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z", "M19.4 14.5a1.6 1.6 0 0 0 .3 1.7l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.7-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.7.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.7 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.7l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.7.3h0a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.7-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.7v0a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"],
  palette: ["M12 3a9 9 0 1 0 0 18c1.4 0 2-.9 2-1.8 0-.8-.5-1.2-.5-1.9 0-.9.7-1.6 1.6-1.6H17a4 4 0 0 0 4-4C21 6.8 17 3 12 3z", "M7.5 11.5h.01", "M10 7.8h.01", "M14.5 7.8h.01"],
  database: ["M12 4c4.4 0 8 1.1 8 2.5S16.4 9 12 9 4 7.9 4 6.5 7.6 4 12 4z", "M4 6.5v11C4 18.9 7.6 20 12 20s8-1.1 8-2.5v-11", "M4 12c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5"],
  lock: ["M6.5 11h11a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1z", "M8.5 11V8a3.5 3.5 0 0 1 7 0v3"],
  info: ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M12 11v5.5", "M12 7.7h.01"],
  alert: ["M12 3.5l9.5 16.5h-19z", "M12 10v4.5", "M12 17.3h.01"],
  external: ["M14 4h6v6", "M20 4l-9 9", "M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10"],
  menu: ["M4 7h16", "M4 12h16", "M4 17h16"],
  home: ["M4 11l8-7 8 7", "M6 9.5V20h12V9.5", "M10 20v-5.5h4V20"],
  user: ["M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8z", "M4.5 20a7.5 7.5 0 0 1 15 0"],
  star: ["M12 3.5l2.6 5.4 5.9.8-4.3 4.1 1 5.9L12 16.9 6.8 19.7l1-5.9L3.5 9.7l5.9-.8z"],
  filter: ["M4 5h16l-6 7.5V19l-4 1.5v-8z"],
  sort: ["M8 4v16", "M4.5 16.5L8 20l3.5-3.5", "M16 20V4", "M12.5 7.5L16 4l3.5 3.5"],
  link: ["M10 14a4 4 0 0 0 5.7 0l3.3-3.3a4 4 0 0 0-5.7-5.7l-1 1", "M14 10a4 4 0 0 0-5.7 0L5 13.3A4 4 0 0 0 10.7 19l1-1"],
  archive: ["M4 5h16v4H4z", "M5.5 9v9.5a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1V9", "M10 13h4"],
  laptop: ["M5.5 6A1.5 1.5 0 0 1 7 4.5h10A1.5 1.5 0 0 1 18.5 6v9h-13z", "M3 18.5h18", "M3 18.5l1-3.5h16l1 3.5"],
  layout: ["M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5z", "M10 4v16", "M4 10h6"],
  drag: ["M9 6h.01", "M9 12h.01", "M9 18h.01", "M15 6h.01", "M15 12h.01", "M15 18h.01"],
};

export type GlyphName = keyof typeof GLYPHS | (string & {});

/** Add or override a glyph (24x24 outline paths). Applications register theirs once at start-up. */
export function registerGlyph(name: string, paths: string[]): void { GLYPHS[name] = paths; }

export function hasGlyph(name: string): boolean { return name in GLYPHS; }

/** An inline icon that inherits the text colour. Decorative by default; pass `label` when the icon stands alone. */
export function glyph(name: GlyphName, label?: string): SVGSVGElement {
  const paths = GLYPHS[name] ?? GLYPHS["info"] ?? [];
  const svg = s("svg", { class: "glyph", attrs: { viewBox: "0 0 24 24", focusable: "false", ...(label ? { role: "img", "aria-label": label } : { "aria-hidden": "true" }) } });
  for (const d of paths) svg.appendChild(s("path", { attrs: { d } }));
  return svg;
}

/** A brand icon from applications/assets (served at /assets/...), drawn as a mask so it follows the accent colour. */
export function assetIcon(url: string, label?: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.className = "icon asset";
  span.style.setProperty("--icon", `url("${url.replace(/"/g, "%22")}")`);
  if (label) { span.setAttribute("role", "img"); span.setAttribute("aria-label", label); } else span.setAttribute("aria-hidden", "true");
  return span;
}
