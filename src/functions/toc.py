"""
toc.py — Universal Table of Contents handler (Python).

Parse HTML or Markdown, build a hierarchical TOC, generate slugs,
and render as HTML, Markdown, JSON, or a nested list.

Public API:
    TocOptions, TocItem, Toc, TocBuilder, default_slugify

Example:
    from toc import TocBuilder, TocOptions

    builder = TocBuilder(TocOptions(min_level=2, max_level=4, numbered=True))
    toc = builder.from_html(html_string)
    print(toc.render_html())
    print(toc.to_json())

    # Inject IDs back into the source HTML (best-effort):
    new_html = builder.inject_ids(html_string, toc)
"""

from __future__ import annotations

import html as _html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable, Iterable, Iterator, Optional

__all__ = [
    "TocOptions",
    "TocItem",
    "Toc",
    "TocBuilder",
    "default_slugify",
]

SlugifyFn = Callable[[str], str]


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_WS_RE = re.compile(r"[\s_]+", re.UNICODE)


def default_slugify(text: str, separator: str = "-") -> str:
    """Generate a URL-safe slug from text.

    - NFKD normalization with combining-mark stripping
    - Lowercased
    - Punctuation removed (Unicode letters retained)
    - Whitespace/underscores collapsed to `separator`
    - Repeated separators collapsed to one
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(separator, text)
    if separator:
        text = re.sub(re.escape(separator) + "+", separator, text)
    return text.strip(separator)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class TocOptions:
    """Configuration for TocBuilder / Toc."""

    min_level: int = 1                # Lowest heading level included
    max_level: int = 6                # Highest heading level included
    slugify: Optional[SlugifyFn] = None
    slug_separator: str = "-"         # Used for slug and dedupe suffix
    unique_slugs: bool = True         # Append -1, -2, ... on collision
    use_existing_ids: bool = True     # Reuse `id="..."` when present
    numbered: bool = False            # Emit "1.2.3" style numbers
    ordered_list: bool = False        # Use <ol> instead of <ul>
    list_class: str = "toc"
    item_class: str = "toc-item"
    link_class: str = "toc-link"
    active_class: str = "toc-active"  # Used by scroll-spy in TS sibling
    max_depth: Optional[int] = None   # Truncate deeper items (None = unlimited)
    scrollspy: bool = True            # Emit data-toc-id/level attributes


# ---------------------------------------------------------------------------
# Item
# ---------------------------------------------------------------------------

@dataclass
class TocItem:
    id: str
    text: str
    level: int
    children: list["TocItem"] = field(default_factory=list)
    number: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "level": self.level,
            "number": self.number,
            "children": [c.to_dict() for c in self.children],
        }

    def __repr__(self) -> str:
        return (
            f"TocItem(id={self.id!r}, text={self.text!r}, "
            f"level={self.level}, children={len(self.children)})"
        )


# ---------------------------------------------------------------------------
# Toc — holds the built tree and renders it
# ---------------------------------------------------------------------------

class Toc:
    """The built table of contents."""

    def __init__(self, items: list[TocItem], options: TocOptions):
        self.items = items
        self.options = options

    # -- Iteration --------------------------------------------------------
    def flat(self) -> Iterator[TocItem]:
        """Depth-first, document-order iteration over every item."""
        for item in self.items:
            yield from self._walk(item)

    def _walk(self, item: TocItem) -> Iterator[TocItem]:
        yield item
        for c in item.children:
            yield from self._walk(c)

    def find(self, item_id: str) -> Optional[TocItem]:
        for item in self.flat():
            if item.id == item_id:
                return item
        return None

    def __len__(self) -> int:
        return sum(1 for _ in self.flat())

    def __iter__(self) -> Iterator[TocItem]:
        return self.flat()

    # -- Serialization ----------------------------------------------------
    def to_dict(self) -> list[dict[str, Any]]:
        return [i.to_dict() for i in self.items]

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # -- Rendering --------------------------------------------------------
    def render_html(self) -> str:
        o = self.options
        if not self.items:
            return f'<nav class="{_esc(o.list_class)}" aria-label="Table of contents"></nav>'
        tag = "ol" if o.ordered_list else "ul"
        parts = [
            f'<nav class="{_esc(o.list_class)}" aria-label="Table of contents">',
            f"<{tag}>",
        ]
        for item in self.items:
            parts.append(self._render_item(item, tag))
        parts.append(f"</{tag}>")
        parts.append("</nav>")
        return "".join(parts)

    def _render_item(self, item: TocItem, tag: str) -> str:
        o = self.options
        attrs = (
            f' data-toc-id="{_esc(item.id)}" data-toc-level="{item.level}"'
            if o.scrollspy
            else ""
        )
        label = (
            f"{_esc(item.number)} {_esc(item.text)}"
            if item.number
            else _esc(item.text)
        )
        link = (
            f'<a class="{_esc(o.link_class)}" href="#{_esc(item.id)}"{attrs}>'
            f"{label}</a>"
        )
        li = (
            f'<li class="{_esc(o.item_class)}" data-level="{item.level}">{link}'
        )
        if item.children:
            li += f"<{tag}>"
            for c in item.children:
                li += self._render_item(c, tag)
            li += f"</{tag}>"
        return li + "</li>"

    def render_markdown(self) -> str:
        lines: list[str] = []
        for item in self.items:
            lines.extend(self._render_md_item(item, 0))
        return "\n".join(lines)

    def _render_md_item(self, item: TocItem, depth: int) -> list[str]:
        bullet = "1." if self.options.numbered else "-"
        prefix = "  " * depth + f"{bullet} "
        text = f"{item.number} {item.text}" if item.number else item.text
        out = [f"{prefix}[{text}](#{item.id})"]
        for c in item.children:
            out.extend(self._render_md_item(c, depth + 1))
        return out

    def render_links(self) -> str:
        """Flat list of links (handy for compact sidebars)."""
        return "\n".join(
            f'<a class="{_esc(self.options.link_class)}" href="#{_esc(i.id)}">'
            f"{_esc(i.text)}</a>"
            for i in self.flat()
        )


def _esc(s: str) -> str:
    return _html.escape(s, quote=True)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

_HEADING_TAG_RE = re.compile(r"^h([1-6])$", re.IGNORECASE)


class _HeadingExtractor(HTMLParser):
    """Extract heading level, text, and existing id from HTML."""

    def __init__(self, options: TocOptions):
        super().__init__(convert_charrefs=True)
        self.options = options
        self.headings: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        m = _HEADING_TAG_RE.match(tag)
        if not m:
            return
        level = int(m.group(1))
        if not (self.options.min_level <= level <= self.options.max_level):
            self._stack.append({"skip": True, "level": level})
            return
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        self._stack.append(
            {"skip": False, "level": level, "attrs": attr_dict, "text_parts": []}
        )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        # Rare self-closing heading; treat as empty heading.
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        m = _HEADING_TAG_RE.match(tag)
        if not m or not self._stack:
            return
        level = int(m.group(1))
        if self._stack[-1].get("level") != level:
            return  # ignore stray close tags
        top = self._stack.pop()
        if top.get("skip"):
            return
        text = re.sub(r"\s+", " ", "".join(top["text_parts"]).strip())
        if not text:
            return
        self.headings.append(
            {
                "level": level,
                "text": text,
                "id": (top["attrs"].get("id") or "").strip(),
            }
        )

    def handle_data(self, data: str) -> None:
        if self._stack and not self._stack[-1].get("skip"):
            self._stack[-1]["text_parts"].append(data)


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_FENCE_RE = re.compile(r"^(```|~~~)")


def _extract_markdown_headings(md: str, options: TocOptions) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    in_fence = False
    for line in md.splitlines():
        if _MD_FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _MD_HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        if not (options.min_level <= level <= options.max_level):
            continue
        text = m.group(2).strip()
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links -> text
        text = re.sub(r"[*_`~]+", "", text)                    # inline emphasis
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        headings.append({"level": level, "text": text, "id": ""})
    return headings


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TocBuilder:
    """Parses sources and builds a Toc."""

    def __init__(self, options: Optional[TocOptions] = None):
        self.options = options or TocOptions()
        if not (1 <= self.options.min_level <= self.options.max_level <= 6):
            raise ValueError("Requires 1 <= min_level <= max_level <= 6")

    # -- Public --------------------------------------------------------
    def from_html(self, html: str) -> Toc:
        parser = _HeadingExtractor(self.options)
        parser.feed(html)
        parser.close()
        return self._build(parser.headings)

    def from_markdown(self, md: str) -> Toc:
        return self._build(_extract_markdown_headings(md, self.options))

    def from_headings(self, headings: Iterable[dict[str, Any]]) -> Toc:
        return self._build(list(headings))

    def from_auto(self, content: str) -> Toc:
        """Auto-detect: HTML if any <hN> tag present, else Markdown."""
        if re.search(r"<h[1-6]\b", content, re.IGNORECASE):
            return self.from_html(content)
        return self.from_markdown(content)

    # -- ID injection --------------------------------------------------
    def inject_ids(self, html: str, toc: Toc) -> str:
        """Best-effort insertion of `id` attributes into heading tags.

        - Matches `<hN ...>` in document order against toc.flat().
        - Skips tags that already have an `id="..."` attribute.
        - Requires that `toc` was built from the same HTML (same options).
        - Attribute values containing `>` are not supported.
        """
        items = list(toc.flat())
        idx = 0

        def repl(m: re.Match) -> str:
            nonlocal idx
            if idx >= items.length if False else idx >= len(items):
                return m.group(0)
            level = int(m.group(1))
            if items[idx].level != level:
                return m.group(0)  # this heading wasn't in the TOC
            item = items[idx]
            idx += 1
            if re.search(r"\bid\s*=", m.group(2), re.IGNORECASE):
                return m.group(0)
            return f'<h{level}{m.group(2)} id="{_esc(item.id)}">'

        return re.sub(r"<h([1-6])\b([^>]*)>", repl, html, flags=re.IGNORECASE)

    # -- Internal ------------------------------------------------------
    def _build(self, raw: list[dict[str, Any]]) -> Toc:
        o = self.options
        used: dict[str, int] = {}
        flat: list[TocItem] = []

        for h in raw:
            try:
                level = int(h["level"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (o.min_level <= level <= o.max_level):
                continue
            text = (h.get("text") or "").strip()
            if not text:
                continue

            existing = ((h.get("id") or "").strip() if o.use_existing_ids else "")
            slug = existing
            if not slug:
                slugify = o.slugify or (
                    lambda s: default_slugify(s, o.slug_separator)
                )
                slug = slugify(text) or "section"

            if o.unique_slugs:
                if slug in used:
                    used[slug] += 1
                    slug = f"{slug}{o.slug_separator}{used[slug]}"
                else:
                    used[slug] = 0

            flat.append(TocItem(id=slug, text=text, level=level))

        if o.numbered:
            self._number(flat)

        tree = self._treeify(flat)
        if o.max_depth:
            self._truncate(tree, 1)
        return Toc(tree, o)

    @staticmethod
    def _number(items: list[TocItem]) -> None:
        counters = [0] * 7
        for item in items:
            lvl = item.level
            counters[lvl] += 1
            for i in range(lvl + 1, 7):
                counters[i] = 0
            item.number = ".".join(
                str(counters[i]) for i in range(1, lvl + 1) if counters[i] > 0
            )

    @staticmethod
    def _treeify(items: list[TocItem]) -> list[TocItem]:
        roots: list[TocItem] = []
        stack: list[TocItem] = []
        for item in items:
            while stack and stack[-1].level >= item.level:
                stack.pop()
            if stack:
                stack[-1].children.append(item)
            else:
                roots.append(item)
            stack.append(item)
        return roots

    def _truncate(self, items: list[TocItem], depth: int) -> None:
        max_depth = self.options.max_depth or 10**9
        for item in items:
            if depth >= max_depth:
                item.children = []
            else:
                self._truncate(item.children, depth + 1)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    <h1>Guide</h1>
    <p>Intro.</p>
    <h2 id="install">Installation</h2>
    <h3>Linux</h3>
    <h3>macOS</h3>
    <h2>Usage</h2>
    <h3>Install</h3>
    """
    builder = TocBuilder(TocOptions(numbered=True, min_level=1, max_level=3))
    toc = builder.from_html(sample)
    print(toc.render_markdown())
    print("----")
    print(toc.render_html())
    print("----")
    print(toc.to_json(indent=None))
    print("----")
    print(builder.inject_ids(sample, toc))