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

from .utils.functions_helpers import _fingerprint
from .utils.functions_error import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Table of Content")
printer = PrettyPrinter()


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
    if not isinstance(separator, str) or not separator:
        raise TocOptionsError(
            "slug_separator must be a non-empty string",
            details={"separator": repr(separator)},
        )
    if not isinstance(text, str):
        logger.debug(
            "slugify received non-string input; coercing",
            extra={"input_type": type(text).__name__},
        )
        text = "" if text is None else str(text)

    try:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = text.lower()
        text = _PUNCT_RE.sub("", text)
        text = _WS_RE.sub(separator, text)
        text = re.sub(re.escape(separator) + "+", separator, text)
        return text.strip(separator)
    except Exception as exc:
        logger.error("Slugification failed", extra={"error": str(exc)})
        raise TocError(
            f"Slugification failed: {exc}",
            details={"raw": text[:200]},
        ) from exc


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class TocOptions:
    """Configuration for TocBuilder / Toc.

    Validated on construction; raises :class:`TocOptionsError` on bad input.
    """

    min_level: int = 1
    max_level: int = 6
    slugify: Optional[SlugifyFn] = None
    slug_separator: str = "-"
    unique_slugs: bool = True
    use_existing_ids: bool = True
    numbered: bool = False
    ordered_list: bool = False
    list_class: str = "toc"
    item_class: str = "toc-item"
    link_class: str = "toc-link"
    active_class: str = "toc-active"
    max_depth: Optional[int] = None
    scrollspy: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "TocOptions":
        if not isinstance(self.min_level, int) or isinstance(self.min_level, bool):
            raise TocOptionsError(
                "min_level must be an integer",
                details={"min_level": repr(self.min_level)},
            )
        if not isinstance(self.max_level, int) or isinstance(self.max_level, bool):
            raise TocOptionsError(
                "max_level must be an integer",
                details={"max_level": repr(self.max_level)},
            )
        if not (1 <= self.min_level <= self.max_level <= 6):
            raise TocOptionsError(
                f"Requires 1 <= min_level <= max_level <= 6, got "
                f"min_level={self.min_level}, max_level={self.max_level}",
                details={
                    "min_level": self.min_level,
                    "max_level": self.max_level,
                },
            )
        if not isinstance(self.slug_separator, str) or not self.slug_separator:
            raise TocOptionsError(
                "slug_separator must be a non-empty string",
                details={"slug_separator": repr(self.slug_separator)},
            )
        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or self.max_depth < 1
        ):
            raise TocOptionsError(
                "max_depth must be None or a positive integer",
                details={"max_depth": repr(self.max_depth)},
            )
        if self.slugify is not None and not callable(self.slugify):
            raise TocOptionsError(
                "slugify must be callable",
                details={"slugify": repr(self.slugify)},
            )
        for name in ("list_class", "item_class", "link_class", "active_class"):
            if not isinstance(getattr(self, name), str):
                raise TocOptionsError(f"{name} must be a string")
        return self

    def fingerprint(self) -> str:
        """Stable hash of the rendered TOC shape — useful for caching."""
        return _fingerprint({
            "min_level": self.min_level,
            "max_level": self.max_level,
            "slug_separator": self.slug_separator,
            "unique_slugs": self.unique_slugs,
            "use_existing_ids": self.use_existing_ids,
            "numbered": self.numbered,
            "ordered_list": self.ordered_list,
            "max_depth": self.max_depth,
        })


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
# Toc
# ---------------------------------------------------------------------------

class Toc:
    """The built table of contents."""

    def __init__(self, items: list[TocItem], options: TocOptions):
        self.items = items
        self.options = options

    def flat(self) -> Iterator[TocItem]:
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

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, Optional[str]]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        m = _HEADING_TAG_RE.match(tag)
        if not m or not self._stack:
            return
        level = int(m.group(1))
        if self._stack[-1].get("level") != level:
            return
        top = self._stack.pop()
        if top.get("skip"):
            return
        text = re.sub(r"\s+", " ", "".join(top["text_parts"]).strip())
        if not text:
            logger.debug("Heading with empty text skipped", extra={"level": level})
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
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"[*_`~]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            logger.debug("Markdown heading with empty text skipped",
                         extra={"level": level})
            continue
        headings.append({"level": level, "text": text, "id": ""})
    return headings


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TocBuilder:
    """Parses sources and builds a Toc."""

    def __init__(self, options: Optional[TocOptions] = None):
        self.options = options if options is not None else TocOptions()
        logger.debug(
            "TocBuilder initialized",
            extra={
                "min_level": self.options.min_level,
                "max_level": self.options.max_level,
                "numbered": self.options.numbered,
                "max_depth": self.options.max_depth,
            },
        )

    # -- Public --------------------------------------------------------
    def from_html(self, html: str) -> Toc:
        if not isinstance(html, str):
            raise TocParseError("html", f"expected str, got {type(html).__name__}")
        logger.debug("Parsing TOC from HTML", extra={"length": len(html)})
        try:
            parser = _HeadingExtractor(self.options)
            parser.feed(html)
            parser.close()
        except Exception as exc:
            logger.error("HTML parse failed", extra={"error": str(exc)})
            raise TocParseError("html", str(exc)) from exc
        return self._build(parser.headings)

    def from_markdown(self, md: str) -> Toc:
        if not isinstance(md, str):
            raise TocParseError("markdown",
                                f"expected str, got {type(md).__name__}")
        logger.debug("Parsing TOC from Markdown", extra={"length": len(md)})
        try:
            headings = _extract_markdown_headings(md, self.options)
        except Exception as exc:
            logger.error("Markdown parse failed", extra={"error": str(exc)})
            raise TocParseError("markdown", str(exc)) from exc
        return self._build(headings)

    def from_headings(self, headings: Iterable[dict[str, Any]]) -> Toc:
        if not isinstance(headings, Iterable) or isinstance(headings, (str, bytes)):
            raise TocParseError(
                "headings",
                "expected an iterable of {level, text, id?} dicts",
            )
        try:
            as_list = list(headings)
        except Exception as exc:
            raise TocParseError("headings", str(exc)) from exc
        return self._build(as_list)

    def from_auto(self, content: str) -> Toc:
        if re.search(r"<h[1-6]\b", content, re.IGNORECASE):
            return self.from_html(content)
        return self.from_markdown(content)

    # -- ID injection --------------------------------------------------
    def inject_ids(self, html: str, toc: Toc) -> str:
        """Best-effort insertion of ``id`` attributes into heading tags.

        Matches each ``<hN>…</hN>`` against ``toc.flat()`` in document order
        using **both level and text** so headings that were filtered out
        during TOC construction (empty text, out of range) do not desync
        subsequent matches. Headings that already carry an ``id`` are left
        alone and their slot in ``toc.flat()`` is preserved.
        """
        if not isinstance(html, str):
            raise TocInjectError("html must be a string")
        if not isinstance(toc, Toc):
            raise TocInjectError("toc must be a Toc instance")

        items = list(toc.flat())
        idx = 0

        def repl(m: re.Match) -> str:
            nonlocal idx
            if idx >= len(items):
                return m.group(0)
            level = int(m.group(1))
            attrs = m.group(2)
            inner = m.group(3)
            plain = _html.unescape(
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", inner)).strip()
            )
            item = items[idx]
            if item.level != level or item.text != plain:
                return m.group(0)
            idx += 1
            if re.search(r"\bid\s*=", attrs, re.IGNORECASE):
                return m.group(0)
            return f'<h{level}{attrs} id="{_esc(item.id)}">{inner}</h{level}>'

        try:
            result = re.sub(
                r"<h([1-6])\b([^>]*)>([\s\S]*?)</h\1\s*>",
                repl,
                html,
                flags=re.IGNORECASE,
            )
        except re.error as exc:
            logger.error("ID injection regex failed", extra={"error": str(exc)})
            raise TocInjectError(
                f"Regex error: {exc}", html_hint=html[:80]
            ) from exc

        logger.info(
            "Injected IDs into HTML",
            extra={
                "injected": idx,
                "total_items": len(items),
                "html_length": len(html),
            },
        )
        if idx != len(items):
            logger.warning(
                "ID injection matched only a subset of TOC items",
                extra={"injected": idx, "expected": len(items)},
            )
        return result

    # -- Internal ------------------------------------------------------
    def _build(self, raw: list[dict[str, Any]]) -> Toc:
        o = self.options
        used: dict[str, int] = {}
        flat: list[TocItem] = []

        for h in raw:
            try:
                level = int(h["level"])
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping heading with invalid level",
                             extra={"heading": repr(h)[:120]})
                continue
            if not (o.min_level <= level <= o.max_level):
                logger.debug(
                    "Skipping heading outside level range",
                    extra={"level": level,
                           "min_level": o.min_level,
                           "max_level": o.max_level},
                )
                continue
            text = (h.get("text") or "").strip()
            if not text:
                logger.debug("Skipping heading with empty text",
                             extra={"level": level})
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
                    original = slug
                    slug = f"{slug}{o.slug_separator}{used[slug]}"
                    logger.debug(
                        "Duplicate slug disambiguated",
                        extra={"original": original, "resolved": slug},
                    )
                else:
                    used[slug] = 0

            flat.append(TocItem(id=slug, text=text, level=level))

        if o.numbered:
            self._number(flat)

        tree = self._treeify(flat)
        if o.max_depth:
            self._truncate(tree, 1)

        logger.info(
            "TOC built",
            extra={
                "items": len(flat),
                "roots": len(tree),
                "max_level": max((i.level for i in flat), default=0),
                "options_fingerprint": o.fingerprint()[:12],
            },
        )
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


__all__ = [
    "TocOptions",
    "TocItem",
    "Toc",
    "TocBuilder",
    "default_slugify",
]


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