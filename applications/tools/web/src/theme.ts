/** Appearance: theme, text size, reduced motion and skin. The same four settings exist in every SLAI application, so the control
 *  and the way it is applied live here. Persisting is the application's job (SPlanner stores it in its preferences file). */
import { h } from "./dom.js";
import { glyph } from "./icons.js";
import { clamp } from "./util.js";

export type Theme = "system" | "light" | "dark" | "contrast";
export type Motion = "system" | "reduce";
export type Skin = "raised" | "soft";

export interface Appearance { theme: Theme; scale: number; motion: Motion }

export const THEMES: readonly Theme[] = ["system", "light", "dark", "contrast"];
export const SCALE_RANGE = { low: 0.85, high: 1.4, step: 0.05 } as const;
export const DEFAULT_APPEARANCE: Appearance = { theme: "system", scale: 1, motion: "system" };

export function normaliseAppearance(value: Partial<Appearance> | null | undefined): Appearance {
  const theme = THEMES.includes(value?.theme as Theme) ? (value?.theme as Theme) : DEFAULT_APPEARANCE.theme;
  const rawScale = typeof value?.scale === "number" && Number.isFinite(value.scale) ? value.scale : DEFAULT_APPEARANCE.scale;
  const motion: Motion = value?.motion === "reduce" ? "reduce" : "system";
  return { theme, scale: clamp(rawScale, SCALE_RANGE.low, SCALE_RANGE.high), motion };
}

/** Write the appearance onto <html>. The skin is a property of the application, not a user choice. */
export function applyAppearance(appearance: Appearance, skin?: Skin, root: HTMLElement = document.documentElement): void {
  if (appearance.theme === "system") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", appearance.theme);
  if (appearance.motion === "reduce") root.setAttribute("data-motion", "reduce"); else root.removeAttribute("data-motion");
  root.style.setProperty("--font-scale", String(appearance.scale));
  if (skin) root.setAttribute("data-skin", skin);
}

const THEME_LABEL: Record<Theme, string> = { system: "System", light: "Light", dark: "Dark", contrast: "High contrast" };
const THEME_ICON: Record<Theme, string> = { system: "settings", light: "sun", dark: "moon", contrast: "contrast" };

/** The shared appearance panel: theme buttons, text size slider, reduced-motion switch. `onChange` receives the new full appearance. */
export function appearanceControls(current: Appearance, onChange: (next: Appearance) => void): HTMLElement {
  let state = { ...current };
  const emit = (patch: Partial<Appearance>): void => { state = normaliseAppearance({ ...state, ...patch }); applyAppearance(state); onChange(state); };

  const themeButtons = THEMES.map((theme) => h("button", {
    class: "btn small", type: "button", attrs: { "aria-pressed": String(state.theme === theme) },
    on: { click: () => { emit({ theme }); for (const [i, b] of themeButtons.entries()) b.setAttribute("aria-pressed", String(THEMES[i] === theme)); } },
  }, glyph(THEME_ICON[theme] ?? "settings"), THEME_LABEL[theme]));

  const scaleId = "appearance-scale";
  const readout = h("output", { class: "muted", text: `${Math.round(state.scale * 100)}%`, attrs: { for: scaleId } });
  const slider = h("input", {
    type: "range", id: scaleId, min: String(SCALE_RANGE.low), max: String(SCALE_RANGE.high), step: String(SCALE_RANGE.step), value: String(state.scale),
    on: { input: () => { const next = Number(slider.value); readout.textContent = `${Math.round(next * 100)}%`; emit({ scale: next }); } },
  });
  const motion = h("input", { type: "checkbox", checked: state.motion === "reduce", on: { change: () => emit({ motion: motion.checked ? "reduce" : "system" }) } });

  return h("div", { class: "appearance", css: { display: "grid", gap: "1rem" } },
    h("div", { class: "field" }, h("span", { class: "label", text: "Theme" }), h("div", { class: "chips", attrs: { role: "group", "aria-label": "Theme" } }, themeButtons)),
    h("div", { class: "field" }, h("label", { attrs: { for: scaleId }, text: "Text size" }), h("div", { class: "row" }, h("div", { class: "grow" }, slider), readout)),
    h("label", { class: "switch" }, motion, h("span", { text: "Reduce motion" })),
  );
}
