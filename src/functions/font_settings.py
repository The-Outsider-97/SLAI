"""
font_settings.py — Universal font/typography settings handler (Python).

Create, validate, normalize, serialize, and merge typographic settings.
Emit CSS declarations, full CSS rules, inline styles, or JSON. Includes
color parsing/conversion (with WCAG contrast), size conversion, and
theme presets.

Public API:
    FontSettings, FontSettingsError
    PRESETS, GENERIC_FAMILIES, NAMED_COLORS, SIZE_UNITS
    parse_color, format_color, is_valid_color
    contrast_ratio, ensure_contrast, lighten, darken, mix_colors
    parse_size, format_size, convert_size
    font_stack, theme

Example:
    from font_settings import FontSettings, PRESETS, theme

    fs = PRESETS["body"].merge(
        FontSettings(size="1.125rem", line_height=1.6, color="#1f2937")
    ).validate()

    print(fs.to_css(".body-text"))
    print(fs.to_inline())
    print(fs.to_json())

    # From a CSS block:
    fs2 = FontSettings.from_css("h2 { font-size: 24px; color: #333; }")
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import re
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Iterable, Optional, Tuple, Union

__all__ = [
    "FontSettings",
    "FontSettingsError",
    "PRESETS",
    "GENERIC_FAMILIES",
    "NAMED_COLORS",
    "SIZE_UNITS",
    "parse_color",
    "format_color",
    "is_valid_color",
    "contrast_ratio",
    "ensure_contrast",
    "lighten",
    "darken",
    "mix_colors",
    "parse_size",
    "format_size",
    "convert_size",
    "font_stack",
    "theme",
]


class FontSettingsError(ValueError):
    """Raised when typography settings are invalid."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_KEYWORDS = {"normal": 400, "bold": 700}
WEIGHT_RELATIVE = {"bolder", "lighter"}
STYLE_VALUES = {"normal", "italic", "oblique"}
STRETCH_VALUES = {
    "normal", "ultra-condensed", "extra-condensed", "condensed",
    "semi-condensed", "semi-expanded", "expanded", "extra-expanded",
    "ultra-expanded",
}
ALIGN_VALUES = {"left", "right", "center", "justify", "start", "end"}
VERTICAL_ALIGN_VALUES = {
    "baseline", "sub", "super", "text-top", "text-bottom", "middle",
    "top", "bottom",
}
TRANSFORM_VALUES = {"none", "capitalize", "uppercase", "lowercase"}
WHITE_SPACE_VALUES = {"normal", "nowrap", "pre", "pre-wrap", "pre-line", "break-spaces"}
DIRECTION_VALUES = {"ltr", "rtl"}
DECORATION_LINE_VALUES = {"none", "underline", "overline", "line-through", "blink"}
DECORATION_STYLE_VALUES = {"solid", "double", "dotted", "dashed", "wavy"}

GENERIC_FAMILIES = {
    "serif", "sans-serif", "monospace", "cursive", "fantasy",
    "system-ui", "ui-serif", "ui-sans-serif", "ui-monospace",
    "ui-rounded", "emoji", "math", "fangsong",
}

COLOR_KEYWORDS = {
    "transparent", "currentcolor", "currentColor",
    "inherit", "initial", "unset", "revert", "revert-layer",
}

SIZE_UNITS = {
    "px", "pt", "pc", "in", "cm", "mm", "q",
    "em", "rem", "ex", "ch", "%",
    "vw", "vh", "vmin", "vmax",
    "svw", "svh", "lvw", "lvh", "dvw", "dvh",
}

# Absolute px-per-unit factors (1 unit = N px)
_PX_FACTORS = {
    "px": 1.0,
    "in": 96.0,
    "cm": 96.0 / 2.54,
    "mm": 96.0 / 25.4,
    "q": 96.0 / 101.6,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
}

# Canonical property order for deterministic output
_FIELD_ORDER: Tuple[str, ...] = (
    "family", "size", "weight", "style", "stretch", "variant",
    "line_height", "letter_spacing", "word_spacing",
    "color", "background_color",
    "text_align", "vertical_align", "direction", "text_indent", "white_space",
    "text_decoration", "text_transform", "text_shadow",
    "font_feature_settings", "font_variant_ligatures", "opacity",
)

# CSS property name for each field
_CSS_NAME = {
    "family": "font-family",
    "size": "font-size",
    "weight": "font-weight",
    "style": "font-style",
    "stretch": "font-stretch",
    "variant": "font-variant",
    "line_height": "line-height",
    "letter_spacing": "letter-spacing",
    "word_spacing": "word-spacing",
    "color": "color",
    "background_color": "background-color",
    "text_align": "text-align",
    "vertical_align": "vertical-align",
    "direction": "direction",
    "text_indent": "text-indent",
    "white_space": "white-space",
    "text_decoration": "text-decoration",
    "text_transform": "text-transform",
    "text_shadow": "text-shadow",
    "font_feature_settings": "font-feature-settings",
    "font_variant_ligatures": "font-variant-ligatures",
    "opacity": "opacity",
}
_CSS_TO_FIELD = {v: k for k, v in _CSS_NAME.items()}


# ---------------------------------------------------------------------------
# Color engine
# ---------------------------------------------------------------------------

NAMED_COLORS: Dict[str, str] = {
    "aliceblue": "#f0f8ff", "antiquewhite": "#faebd7", "aqua": "#00ffff",
    "aquamarine": "#7fffd4", "azure": "#f0ffff", "beige": "#f5f5dc",
    "bisque": "#ffe4c4", "black": "#000000", "blanchedalmond": "#ffebcd",
    "blue": "#0000ff", "blueviolet": "#8a2be2", "brown": "#a52a2a",
    "burlywood": "#deb887", "cadetblue": "#5f9ea0", "chartreuse": "#7fff00",
    "chocolate": "#d2691e", "coral": "#ff7f50", "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc", "crimson": "#dc143c", "cyan": "#00ffff",
    "darkblue": "#00008b", "darkcyan": "#008b8b", "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9", "darkgreen": "#006400", "darkgrey": "#a9a9a9",
    "darkkhaki": "#bdb76b", "darkmagenta": "#8b008b", "darkolivegreen": "#556b2f",
    "darkorange": "#ff8c00", "darkorchid": "#9932cc", "darkred": "#8b0000",
    "darksalmon": "#e9967a", "darkseagreen": "#8fbc8f", "darkslateblue": "#483d8b",
    "darkslategray": "#2f4f4f", "darkslategrey": "#2f4f4f",
    "darkturquoise": "#00ced1", "darkviolet": "#9400d3", "deeppink": "#ff1493",
    "deepskyblue": "#00bfff", "dimgray": "#696969", "dimgrey": "#696969",
    "dodgerblue": "#1e90ff", "firebrick": "#b22222", "floralwhite": "#fffaf0",
    "forestgreen": "#228b22", "fuchsia": "#ff00ff", "gainsboro": "#dcdcdc",
    "ghostwhite": "#f8f8ff", "gold": "#ffd700", "goldenrod": "#daa520",
    "gray": "#808080", "green": "#008000", "greenyellow": "#adff2f",
    "grey": "#808080", "honeydew": "#f0fff0", "hotpink": "#ff69b4",
    "indianred": "#cd5c5c", "indigo": "#4b0082", "ivory": "#fffff0",
    "khaki": "#f0e68c", "lavender": "#e6e6fa", "lavenderblush": "#fff0f5",
    "lawngreen": "#7cfc00", "lemonchiffon": "#fffacd", "lightblue": "#add8e6",
    "lightcoral": "#f08080", "lightcyan": "#e0ffff",
    "lightgoldenrodyellow": "#fafad2", "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90", "lightgrey": "#d3d3d3", "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a", "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa", "lightslategray": "#778899",
    "lightslategrey": "#778899", "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0", "lime": "#00ff00", "limegreen": "#32cd32",
    "linen": "#faf0e6", "magenta": "#ff00ff", "maroon": "#800000",
    "mediumaquamarine": "#66cdaa", "mediumblue": "#0000cd",
    "mediumorchid": "#ba55d3", "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371", "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a", "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585", "midnightblue": "#191970",
    "mintcream": "#f5fffa", "mistyrose": "#ffe4e1", "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead", "navy": "#000080", "oldlace": "#fdf5e6",
    "olive": "#808000", "olivedrab": "#6b8e23", "orange": "#ffa500",
    "orangered": "#ff4500", "orchid": "#da70d6", "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98", "paleturquoise": "#afeeee",
    "palevioletred": "#db7093", "papayawhip": "#ffefd5", "peachpuff": "#ffdab9",
    "peru": "#cd853f", "pink": "#ffc0cb", "plum": "#dda0dd",
    "powderblue": "#b0e0e6", "purple": "#800080", "rebeccapurple": "#663399",
    "red": "#ff0000", "rosybrown": "#bc8f8f", "royalblue": "#4169e1",
    "saddlebrown": "#8b4513", "salmon": "#fa8072", "sandybrown": "#f4a460",
    "seagreen": "#2e8b57", "seashell": "#fff5ee", "sienna": "#a0522d",
    "silver": "#c0c0c0", "skyblue": "#87ceeb", "slateblue": "#6a5acd",
    "slategray": "#708090", "slategrey": "#708090", "snow": "#fffafa",
    "springgreen": "#00ff7f", "steelblue": "#4682b4", "tan": "#d2b48c",
    "teal": "#008080", "thistle": "#d8bfd8", "tomato": "#ff6347",
    "turquoise": "#40e0d0", "violet": "#ee82ee", "wheat": "#f5deb3",
    "white": "#ffffff", "whitesmoke": "#f5f5f5", "yellow": "#ffff00",
    "yellowgreen": "#9acd32",
}

RGBA = Tuple[float, float, float, float]  # r,g,b in 0-255, a in 0-1

_HEX_RE = re.compile(r"^#([0-9a-fA-F]+)$")
_RGB_RE = re.compile(
    r"^rgba?\(\s*([\d.]+%?)\s*[,\s]\s*([\d.]+%?)\s*[,\s]\s*([\d.]+%?)\s*"
    r"(?:[,/]\s*([\d.]+%?)\s*)?\)$",
    re.IGNORECASE,
)
_HSL_RE = re.compile(
    r"^hsla?\(\s*(-?[\d.]+)(?:deg|grad|rad|turn)?\s*[,\s]\s*([\d.]+)%\s*[,\s]\s*"
    r"([\d.]+)%\s*(?:[,/]\s*([\d.]+%?)\s*)?\)$",
    re.IGNORECASE,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _parse_alpha(s: Optional[str]) -> float:
    if not s:
        return 1.0
    if s.endswith("%"):
        return _clamp(float(s[:-1]) / 100.0, 0.0, 1.0)
    return _clamp(float(s), 0.0, 1.0)


def _parse_channel(s: str) -> float:
    if s.endswith("%"):
        return _clamp(float(s[:-1]) * 255.0 / 100.0, 0.0, 255.0)
    return _clamp(float(s), 0.0, 255.0)


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    h = (h % 360) / 360.0
    if s <= 0:
        v = l * 255.0
        return v, v, v
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q

    def hue(t: float) -> float:
        t = (t + 1) % 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    return hue(h + 1 / 3) * 255.0, hue(h) * 255.0, hue(h - 1 / 3) * 255.0


def parse_color(value: Any) -> Optional[RGBA]:
    """Parse a CSS color into (r, g, b, a). Returns None for keywords / invalid."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Named color
    low = s.lower()
    if low in NAMED_COLORS:
        s = NAMED_COLORS[low]

    # Hex
    if s.startswith("#"):
        m = _HEX_RE.match(s)
        if not m:
            return None
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h) + "ff"
        elif len(h) == 4:
            h = "".join(c * 2 for c in h)
        elif len(h) == 6:
            h += "ff"
        elif len(h) != 8:
            return None
        try:
            return (
                float(int(h[0:2], 16)),
                float(int(h[2:4], 16)),
                float(int(h[4:6], 16)),
                int(h[6:8], 16) / 255.0,
            )
        except ValueError:
            return None

    # rgb / rgba
    m = _RGB_RE.match(s)
    if m:
        try:
            return (
                _parse_channel(m.group(1)),
                _parse_channel(m.group(2)),
                _parse_channel(m.group(3)),
                _parse_alpha(m.group(4)),
            )
        except ValueError:
            return None

    # hsl / hsla
    m = _HSL_RE.match(s)
    if m:
        try:
            h = float(m.group(1))
            s_pct = _clamp(float(m.group(2)) / 100.0, 0.0, 1.0)
            l_pct = _clamp(float(m.group(3)) / 100.0, 0.0, 1.0)
            r, g, b = _hsl_to_rgb(h, s_pct, l_pct)
            return (r, g, b, _parse_alpha(m.group(4)))
        except ValueError:
            return None

    return None


def is_valid_color(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    if not s:
        return False
    if s in COLOR_KEYWORDS or s == "currentcolor":
        return True
    return parse_color(value) is not None


def format_color(rgba: RGBA, format: str = "hex") -> str:
    """Format an RGBA tuple as hex, rgb, rgba, or hsl string."""
    r, g, b, a = rgba
    ri, gi, bi = int(round(_clamp(r, 0, 255))), int(round(_clamp(g, 0, 255))), int(round(_clamp(b, 0, 255)))
    if format == "hex":
        if a >= 1.0:
            return f"#{ri:02x}{gi:02x}{bi:02x}"
        return f"#{ri:02x}{gi:02x}{bi:02x}{int(round(a * 255)):02x}"
    if format == "rgb":
        return f"rgb({ri}, {gi}, {bi})"
    if format == "rgba":
        return f"rgba({ri}, {gi}, {bi}, {round(a, 3)})"
    if format == "hsl":
        # RGB -> HSL
        rn, gn, bn = ri / 255.0, gi / 255.0, bi / 255.0
        mx, mn = max(rn, gn, bn), min(rn, gn, bn)
        l = (mx + mn) / 2
        if mx == mn:
            h_deg, s = 0.0, 0.0
        else:
            d = mx - mn
            s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
            if mx == rn:
                h_deg = ((gn - bn) / d + (6 if gn < bn else 0)) * 60
            elif mx == gn:
                h_deg = ((bn - rn) / d + 2) * 60
            else:
                h_deg = ((rn - gn) / d + 4) * 60
        if a >= 1.0:
            return f"hsl({round(h_deg)}, {round(s * 100)}%, {round(l * 100)}%)"
        return f"hsla({round(h_deg)}, {round(s * 100)}%, {round(l * 100)}%, {round(a, 3)})"
    raise FontSettingsError(f"Unknown color format: {format!r}")


def relative_luminance(rgba: RGBA) -> float:
    """WCAG 2.x relative luminance from an RGBA tuple."""
    def chan(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b, _ = rgba
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def contrast_ratio(fg: Union[str, RGBA], bg: Union[str, RGBA]) -> float:
    """WCAG contrast ratio between two colors (1.0 – 21.0)."""
    f = fg if isinstance(fg, tuple) else parse_color(fg)
    b = bg if isinstance(bg, tuple) else parse_color(bg)
    if f is None or b is None:
        raise FontSettingsError("contrast_ratio requires parseable colors")
    l1, l2 = relative_luminance(f), relative_luminance(b)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


def ensure_contrast(
    fg: Union[str, RGBA],
    bg: Union[str, RGBA],
    min_ratio: float = 4.5,
    format: str = "hex",
) -> str:
    """Adjust fg toward black or white until contrast_ratio(fg, bg) >= min_ratio."""
    f = fg if isinstance(fg, tuple) else parse_color(fg)
    b = bg if isinstance(bg, tuple) else parse_color(bg)
    if f is None or b is None:
        raise FontSettingsError("ensure_contrast requires parseable colors")
    if contrast_ratio(f, b) >= min_ratio:
        return format_color(f, format)
    target = (255.0, 255.0, 255.0) if relative_luminance(b) < 0.5 else (0.0, 0.0, 0.0)
    fr, fg_, fb, fa = f
    tr, tg, tb = target
    lo, hi = 0.0, 1.0
    # Binary search for the lightest adjustment reaching min_ratio
    for _ in range(24):
        mid = (lo + hi) / 2
        cur = (fr + (tr - fr) * mid, fg_ + (tg - fg_) * mid, fb + (tb - fb) * mid, fa)
        if contrast_ratio(cur, b) >= min_ratio:
            hi = mid
        else:
            lo = mid
    return format_color((fr + (tr - fr) * hi, fg_ + (tg - fg_) * hi, fb + (tb - fb) * hi, fa), format)


def lighten(color: Union[str, RGBA], amount: float = 0.1, format: str = "hex") -> str:
    c = color if isinstance(color, tuple) else parse_color(color)
    if c is None:
        raise FontSettingsError(f"Cannot lighten non-color: {color!r}")
    r, g, b, a = c
    return format_color(
        (r + (255 - r) * amount, g + (255 - g) * amount, b + (255 - b) * amount, a), format
    )


def darken(color: Union[str, RGBA], amount: float = 0.1, format: str = "hex") -> str:
    c = color if isinstance(color, tuple) else parse_color(color)
    if c is None:
        raise FontSettingsError(f"Cannot darken non-color: {color!r}")
    r, g, b, a = c
    return format_color((r * (1 - amount), g * (1 - amount), b * (1 - amount), a), format)


def mix_colors(a: Union[str, RGBA], b: Union[str, RGBA], t: float = 0.5, format: str = "hex") -> str:
    ca = a if isinstance(a, tuple) else parse_color(a)
    cb = b if isinstance(b, tuple) else parse_color(b)
    if ca is None or cb is None:
        raise FontSettingsError("mix_colors requires parseable colors")
    t = _clamp(t, 0.0, 1.0)
    return format_color(tuple(ca[i] * (1 - t) + cb[i] * t for i in range(4)), format)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Size engine
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"^(-?\d*\.?\d+)([a-zA-Z%]*)$")


def parse_size(value: Any) -> Optional[Tuple[float, str]]:
    """Parse '16px' -> (16.0, 'px'), 16 -> (16.0, 'px'), '1.5rem' -> (1.5, 'rem')."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value), "px")
    s = str(value).strip()
    if not s:
        return None
    m = _SIZE_RE.match(s)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).lower()
    if unit and unit not in SIZE_UNITS:
        return None
    return (num, unit)


def format_size(value: float, unit: str = "px") -> str:
    if unit == "":
        return f"{value:g}"
    return f"{value:g}{unit}"


def convert_size(
    value: float,
    from_unit: str,
    to_unit: str,
    base: float = 16.0,
    viewport: Tuple[float, float] = (1200.0, 800.0),
) -> float:
    """Convert a size between units.

    - `base` = reference size in px for em/rem/ex/ch/%.
    - `viewport` = (width_px, height_px) for vw/vh/vmin/vmax.
    """
    fu = from_unit.lower()
    tu = to_unit.lower()

    def to_px(n: float, u: str) -> float:
        if u in _PX_FACTORS:
            return n * _PX_FACTORS[u]
        if u in {"em", "rem", "%"}:
            return n / 100.0 * base if u == "%" else n * base
        if u == "ex":
            return n * base * 0.5  # approximation
        if u == "ch":
            return n * base * 0.5  # approximation
        vw, vh = viewport
        if u == "vw":
            return n * vw / 100.0
        if u == "vh":
            return n * vh / 100.0
        if u == "vmin":
            return n * min(vw, vh) / 100.0
        if u == "vmax":
            return n * max(vw, vh) / 100.0
        raise FontSettingsError(f"Cannot convert from {u!r}")

    px = to_px(value, fu)
    if tu in _PX_FACTORS:
        return px / _PX_FACTORS[tu]
    if tu in {"em", "rem"}:
        return px / base
    if tu == "%":
        return px / base * 100.0
    if tu == "ex" or tu == "ch":
        return px / (base * 0.5)
    vw, vh = viewport
    if tu == "vw":
        return px * 100.0 / vw
    if tu == "vh":
        return px * 100.0 / vh
    if tu == "vmin":
        return px * 100.0 / min(vw, vh)
    if tu == "vmax":
        return px * 100.0 / max(vw, vh)
    raise FontSettingsError(f"Cannot convert to {tu!r}")


# ---------------------------------------------------------------------------
# Font stack
# ---------------------------------------------------------------------------

def _quote_family(name: str) -> str:
    n = name.strip()
    if not n:
        return ""
    low = n.lower()
    if low in GENERIC_FAMILIES:
        return n
    if n[0] in "\"'" and n[-1] == n[0]:
        return n  # already quoted
    # Quote if it contains whitespace or a comma-like char
    if re.search(r"[\s,;]", n):
        esc = n.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{esc}"'
    return n


def font_stack(family: Union[str, Iterable[str]], generic: str = "sans-serif") -> str:
    """Build a CSS font-family list.

    - Accepts a comma-separated string or an iterable of family names.
    - Quotes family names that require it.
    - Appends `generic` if the last entry isn't already a generic family.
    """
    if isinstance(family, str):
        parts: list[str] = []
        for chunk in family.split(","):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
    else:
        parts = list(family)
    quoted = [_quote_family(p) for p in parts if p]
    if not quoted:
        return generic
    last = quoted[-1].strip().lower()
    if last not in GENERIC_FAMILIES and generic:
        quoted.append(generic)
    return ", ".join(quoted)


# ---------------------------------------------------------------------------
# FontSettings
# ---------------------------------------------------------------------------

@dataclass
class FontSettings:
    """A single typography configuration.

    Fields are lenient by default — pass anything reasonable and call
    `.validate()` to enforce strict CSS value checks.
    """

    family: Optional[Union[str, Iterable[str]]] = None
    size: Optional[Union[str, float, int]] = None
    weight: Optional[Union[str, int]] = None
    style: Optional[str] = None
    stretch: Optional[str] = None
    variant: Optional[str] = None

    line_height: Optional[Union[str, float, int]] = None
    letter_spacing: Optional[Union[str, float]] = None
    word_spacing: Optional[Union[str, float]] = None

    color: Optional[str] = None
    background_color: Optional[str] = None

    text_align: Optional[str] = None
    vertical_align: Optional[str] = None
    direction: Optional[str] = None
    text_indent: Optional[Union[str, float]] = None
    white_space: Optional[str] = None

    text_decoration: Optional[str] = None
    text_transform: Optional[str] = None
    text_shadow: Optional[str] = None

    font_feature_settings: Optional[str] = None
    font_variant_ligatures: Optional[str] = None
    opacity: Optional[Union[float, str]] = None

    extra: Dict[str, str] = field(default_factory=dict)

    # -- Serialization --------------------------------------------------
    def _serialize(self, name: str, value: Any) -> Optional[str]:
        if value is None:
            return None
        if name == "family":
            return font_stack(value) if not isinstance(value, str) or "," not in value and any(
                v in (value or "").lower() for v in GENERIC_FAMILIES
            ) is False and value != value.strip() and False else (
                value if isinstance(value, str) and "," in value else font_stack(value)
            )
        if name in ("size", "letter_spacing", "word_spacing", "text_indent"):
            if isinstance(value, (int, float)):
                return format_size(float(value), "px" if name == "size" else "")
            return str(value)
        if name == "weight":
            if isinstance(value, int):
                return str(value)
            low = str(value).strip().lower()
            if low in WEIGHT_KEYWORDS:
                return str(WEIGHT_KEYWORDS[low])
            return low
        if name == "line_height":
            if isinstance(value, (int, float)):
                return f"{float(value):g}"
            return str(value)
        if name == "opacity":
            if isinstance(value, (int, float)):
                return f"{float(value):g}"
            return str(value)
        return str(value)

    def to_props(self) -> Dict[str, str]:
        """Return a CSS-property-name → value dict (kebab-case keys)."""
        props: Dict[str, str] = {}
        for f in _FIELD_ORDER:
            val = getattr(self, f)
            ser = self._serialize(f, val)
            if ser is not None and ser != "":
                props[_CSS_NAME[f]] = ser
        for k, v in (self.extra or {}).items():
            if v is not None and v != "":
                props[k] = str(v)
        return props

    # Alias
    to_dict = to_props

    def to_json(self, indent: Optional[int] = 2, sort_keys: bool = False) -> str:
        return json.dumps(self.to_props(), indent=indent, sort_keys=sort_keys, ensure_ascii=False)

    def to_css(self, selector: Optional[str] = None, indent: int = 2) -> str:
        props = self.to_props()
        decls = "\n".join(f"{' ' * indent}{k}: {v};" for k, v in props.items())
        if selector is None:
            return decls
        return f"{selector} {{\n{decls}\n}}"

    def to_inline(self) -> str:
        return "; ".join(f"{k}: {v}" for k, v in self.to_props().items())

    def to_style_attr(self) -> str:
        return f'style="{_html.escape(self.to_inline(), quote=True)}"'

    def to_hash(self, length: int = 8) -> str:
        canonical = json.dumps(self.to_props(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]

    def suggest_class(self, prefix: str = "font") -> str:
        return f"{prefix}-{self.to_hash()}"

    # -- Composition ----------------------------------------------------
    def copy(self, **overrides: Any) -> "FontSettings":
        extra = dict(self.extra)
        if "extra" in overrides and isinstance(overrides["extra"], dict):
            extra = overrides.pop("extra")
        clone = replace(self, extra=extra, **overrides)
        return clone

    def merge(self, *others: "FontSettings") -> "FontSettings":
        """Return a new FontSettings; later instances override earlier ones for non-None fields."""
        merged: Dict[str, Any] = {f: getattr(self, f) for f in _FIELD_ORDER}
        extra: Dict[str, str] = dict(self.extra)
        for other in others:
            if not isinstance(other, FontSettings):
                raise FontSettingsError("merge() requires FontSettings instances")
            for f in _FIELD_ORDER:
                v = getattr(other, f)
                if v is not None:
                    merged[f] = v
            for k, v in other.extra.items():
                if v is not None:
                    extra[k] = v
        return FontSettings(**merged, extra=extra)

    # -- Validation -----------------------------------------------------
    def validate(self) -> "FontSettings":
        if self.style is not None and self.style not in STYLE_VALUES:
            raise FontSettingsError(f"Invalid font-style: {self.style!r}")
        if self.stretch is not None and self.stretch not in STRETCH_VALUES:
            raise FontSettingsError(f"Invalid font-stretch: {self.stretch!r}")
        if self.text_align is not None and self.text_align not in ALIGN_VALUES:
            raise FontSettingsError(f"Invalid text-align: {self.text_align!r}")
        if self.vertical_align is not None and self.vertical_align not in VERTICAL_ALIGN_VALUES:
            # allow lengths / percentages
            if parse_size(self.vertical_align) is None:
                raise FontSettingsError(f"Invalid vertical-align: {self.vertical_align!r}")
        if self.direction is not None and self.direction not in DIRECTION_VALUES:
            raise FontSettingsError(f"Invalid direction: {self.direction!r}")
        if self.text_transform is not None and self.text_transform not in TRANSFORM_VALUES:
            raise FontSettingsError(f"Invalid text-transform: {self.text_transform!r}")
        if self.white_space is not None and self.white_space not in WHITE_SPACE_VALUES:
            raise FontSettingsError(f"Invalid white-space: {self.white_space!r}")

        if self.size is not None:
            p = parse_size(self.size)
            if p is None or p[0] < 0:
                raise FontSettingsError(f"Invalid font-size: {self.size!r}")

        if self.weight is not None:
            w = self.weight
            if isinstance(w, int):
                if not (1 <= w <= 1000):
                    raise FontSettingsError(f"Invalid font-weight: {w!r}")
            else:
                low = str(w).strip().lower()
                if low not in WEIGHT_KEYWORDS and low not in WEIGHT_RELATIVE:
                    if not re.fullmatch(r"\d{1,4}", low) or not (1 <= int(low) <= 1000):
                        raise FontSettingsError(f"Invalid font-weight: {w!r}")

        for name in ("color", "background_color"):
            v = getattr(self, name)
            if v is not None and not is_valid_color(v):
                raise FontSettingsError(f"Invalid {_CSS_NAME[name]}: {v!r}")

        if self.text_decoration is not None:
            for tok in str(self.text_decoration).split():
                if tok in DECORATION_LINE_VALUES or tok in DECORATION_STYLE_VALUES:
                    continue
                if is_valid_color(tok) or parse_size(tok) is not None:
                    continue
                raise FontSettingsError(f"Invalid text-decoration token: {tok!r}")

        return self

    # -- Construction from other representations ------------------------
    @classmethod
    def from_props(cls, props: Dict[str, Any]) -> "FontSettings":
        """Rebuild from a kebab-case CSS property map."""
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, str] = {}
        for k, v in (props or {}).items():
            if v is None:
                continue
            key = k.strip()
            field_name = _CSS_TO_FIELD.get(key)
            if field_name:
                kwargs[field_name] = v
            else:
                extra[key] = str(v)
        return cls(**kwargs, extra=extra)

    from_dict = from_props

    @classmethod
    def from_json(cls, data: Union[str, bytes]) -> "FontSettings":
        return cls.from_props(json.loads(data))

    @classmethod
    def from_css(cls, css: str) -> "FontSettings":
        """Parse from an inline style string or a full CSS rule.

        Accepts:
            "font-size: 16px; color: #333"
            "h2 { font-size: 24px; color: red; }"
        """
        text = css.strip()
        m = re.search(r"\{([^{}]*)\}", text)
        body = m.group(1) if m else text
        props: Dict[str, str] = {}
        # Split declarations naively but respecting quotes/parentheses.
        buf, depth, quote = "", 0, ""
        for ch in body:
            if quote:
                buf += ch
                if ch == quote:
                    quote = ""
                continue
            if ch in "\"'":
                quote = ch
                buf += ch
                continue
            if ch == "(":
                depth += 1
                buf += ch
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                buf += ch
                continue
            if ch == ";" and depth == 0:
                _add_decl(buf, props)
                buf = ""
                continue
            buf += ch
        if buf.strip():
            _add_decl(buf, props)
        return cls.from_props(props)


def _add_decl(chunk: str, out: Dict[str, str]) -> None:
    chunk = chunk.strip()
    if not chunk or ":" not in chunk:
        return
    name, _, value = chunk.partition(":")
    name = name.strip()
    value = value.strip()
    # Strip !important — the flag has no representation in FontSettings
    value = re.sub(r"\s*!important\s*$", "", value, flags=re.IGNORECASE)
    if name and value:
        out[name] = value


# ---------------------------------------------------------------------------
# Presets & theming
# ---------------------------------------------------------------------------

PRESETS: Dict[str, FontSettings] = {
    "body": FontSettings(
        family="system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        size="1rem",
        weight=400,
        line_height=1.5,
        color="#1f2937",
        text_align="left",
        direction="ltr",
    ),
    "h1": FontSettings(
        family="inherit", size="2.25rem", weight=700, line_height=1.2,
        letter_spacing="-0.02em", color="inherit",
    ),
    "h2": FontSettings(
        family="inherit", size="1.875rem", weight=700, line_height=1.25,
        letter_spacing="-0.015em", color="inherit",
    ),
    "h3": FontSettings(
        family="inherit", size="1.5rem", weight=600, line_height=1.3, color="inherit",
    ),
    "h4": FontSettings(
        family="inherit", size="1.25rem", weight=600, line_height=1.35, color="inherit",
    ),
    "h5": FontSettings(
        family="inherit", size="1.125rem", weight=600, line_height=1.4, color="inherit",
    ),
    "h6": FontSettings(
        family="inherit", size="1rem", weight=600, line_height=1.4,
        text_transform="uppercase", letter_spacing="0.05em", color="inherit",
    ),
    "link": FontSettings(
        color="#2563eb", text_decoration="underline",
    ),
    "code": FontSettings(
        family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        size="0.9em", weight=400, color="#0f172a",
        background_color="#f1f5f9",
    ),
    "quote": FontSettings(
        family="Georgia, 'Times New Roman', serif", size="1.125rem",
        style="italic", line_height=1.6, color="#374151",
    ),
    "caption": FontSettings(
        family="inherit", size="0.875rem", weight=400, line_height=1.4,
        color="#6b7280",
    ),
    "button": FontSettings(
        family="inherit", size="1rem", weight=500, text_align="center",
        white_space="nowrap",
    ),
}


def theme(name: str = "light") -> Dict[str, FontSettings]:
    """Return a dict of preset → FontSettings adjusted for the named theme."""
    base = dict(PRESETS)
    name = name.lower()

    if name == "light":
        return base

    if name == "dark":
        out = {}
        for k, fs in base.items():
            out[k] = fs.merge(
                FontSettings(color="#e5e7eb")
                if k not in ("link", "code")
                else FontSettings()
            )
        out["body"] = out["body"].merge(FontSettings(color="#e5e7eb"))
        out["link"] = out["link"].merge(FontSettings(color="#60a5fa"))
        out["code"] = out["code"].merge(
            FontSettings(color="#e5e7eb", background_color="#1f2937")
        )
        out["caption"] = out["caption"].merge(FontSettings(color="#9ca3af"))
        out["quote"] = out["quote"].merge(FontSettings(color="#d1d5db"))
        return out

    if name == "print":
        out = {}
        for k, fs in base.items():
            out[k] = fs.merge(
                FontSettings(
                    family="Georgia, 'Times New Roman', serif",
                    color="#000000",
                    background_color=None,
                )
            ).copy(background_color=None)
        out["code"] = FontSettings(
            family="'Courier New', Courier, monospace",
            size="0.9em", color="#000000", background_color=None,
        )
        return out

    if name == "high-contrast":
        out = {}
        for k, fs in base.items():
            out[k] = fs.merge(
                FontSettings(
                    color="#000000",
                    background_color="#ffffff",
                    weight=700 if k.startswith("h") or k == "body" else fs.weight,
                )
            )
        out["link"] = out["link"].merge(FontSettings(color="#0000ee", text_decoration="underline"))
        return out

    raise FontSettingsError(f"Unknown theme: {name!r}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    body = PRESETS["body"].merge(
        FontSettings(size="1.125rem", line_height=1.65, color="#0f172a")
    ).validate()

    print(body.to_css(".body-text"))
    print("----")
    print(body.to_inline())
    print("----")
    print(body.to_json(indent=None))
    print("----")

    # Contrast check
    ratio = contrast_ratio("#333333", "#ffffff")
    print(f"Contrast (#333 on #fff): {ratio:.2f}")
    safe = ensure_contrast("#aaaaaa", "#ffffff", min_ratio=4.5)
    print(f"Ensured fg on white: {safe} (ratio={contrast_ratio(safe, '#fff'):.2f})")

    # Parse from CSS
    fs = FontSettings.from_css("h2 { font-size: 24px; color: #333; font-weight: bold; }")
    print("----")
    print(fs.to_props())
    print(fs.suggest_class())

    # Theme
    dark_link = theme("dark")["link"]
    print("----")
    print("Dark link:", dark_link.to_props())