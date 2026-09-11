"""Stage 1: Parse PDF into structured markdown + extract figures."""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

log = logging.getLogger(__name__)

# Docling only renders element images when the pipeline is told to; the default
# pipeline produces markdown with no bitmaps at all.
IMAGE_SCALE = 2.0

_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_FIGNUM_RE = re.compile(r"\b(?:figure|fig\.?|table|tab\.?)\s*([0-9]+|[ivxlc]+)\b", re.IGNORECASE)


def _build_converter() -> DocumentConverter:
    """Converter configured to render picture and table images."""
    options = PdfPipelineOptions()
    options.images_scale = IMAGE_SCALE
    options.generate_picture_images = True
    options.generate_table_images = True
    # Page images are heavy and we don't use them; keep them off where supported.
    for attr, value in (("generate_page_images", False), ("do_ocr", False)):
        if hasattr(options, attr):
            setattr(options, attr, value)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _to_b64_png(pil_image) -> str | None:
    """Encode a PIL image as base64 PNG, or None if it cannot be encoded."""
    if pil_image is None:
        return None
    try:
        buf = io.BytesIO()
        if pil_image.mode not in ("RGB", "RGBA", "L"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        log.warning("could not encode figure image: %s", exc)
        return None


def _element_image(element, doc):
    """Get a PIL image for a docling element across API variations."""
    # Preferred: DocItem.get_image(doc) resolves the stored ImageRef.
    getter = getattr(element, "get_image", None)
    if callable(getter):
        try:
            image = getter(doc)
            if image is not None:
                return image
        except Exception as exc:
            log.debug("get_image failed: %s", exc)

    # Fallback: element.image is an ImageRef wrapping a PIL image.
    ref = getattr(element, "image", None)
    if ref is None:
        return None
    pil = getattr(ref, "pil_image", None)
    if pil is not None:
        return pil
    # Very old versions stored the PIL image directly.
    return ref if hasattr(ref, "save") else None


def _element_caption(element, doc) -> str:
    """Caption text for a picture/table element, across API variations."""
    getter = getattr(element, "caption_text", None)
    if callable(getter):
        try:
            text = getter(doc)
            if text:
                return str(text).strip()
        except Exception as exc:
            log.debug("caption_text failed: %s", exc)

    caption = getattr(element, "caption", None)
    if isinstance(caption, str):
        return caption.strip()
    return ""


def _element_page(element) -> int | None:
    """Page number for an element, read from its provenance record."""
    prov = getattr(element, "prov", None) or []
    if prov:
        page = getattr(prov[0], "page_no", None)
        if page is not None:
            return page
    return getattr(element, "page_no", None)


def _figure_number(caption: str, kind: str) -> str | None:
    """
    Pull the human-facing figure/table number out of a caption.

    Used to match a figure to the claim that cites it, instead of pairing every
    claim with whichever figure happens to come first.
    """
    match = _FIGNUM_RE.search(caption or "")
    if match:
        return f"{kind}:{match.group(1).lower()}"
    return None


def extract_title(doc, pdf_path: str, markdown: str) -> str:
    """Best available paper title: docling title item → first heading → filename."""
    # Docling labels a title item when it can identify one.
    for attr in ("texts", "body"):
        items = getattr(doc, attr, None) or []
        if not isinstance(items, list):
            continue
        for item in items:
            label = str(getattr(item, "label", "")).lower()
            text = (getattr(item, "text", "") or "").strip()
            if "title" in label and len(text) > 8:
                return text

    # Otherwise the first real markdown heading. Papers often open with a
    # licence or attribution paragraph, so scan past it rather than giving up.
    boilerplate = (
        "provided proper attribution", "permission to reproduce", "creative commons",
        "license", "licence", "copyright", "all rights reserved", "preprint",
        "under review", "arxiv:", "doi:", "abstract",
    )
    for line in markdown.splitlines()[:60]:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        heading = stripped.lstrip("#").strip()
        if len(heading) > 8 and not any(b in heading.lower() for b in boilerplate):
            return heading

    name = getattr(doc, "name", None)
    return name or Path(pdf_path).stem


def extract_year(markdown: str, title: str = "", doc=None) -> int | None:
    """
    Determine the paper's publication year.

    Order: Docling metadata → explicit date lines → Semantic Scholar title
    lookup → loose scrape of the header. Returns None if everything fails.
    """
    # 1. Docling metadata, when the PDF carried usable creation info.
    origin = getattr(doc, "origin", None) if doc is not None else None
    for attr in ("creation_date", "created", "date"):
        value = getattr(origin, attr, None) if origin is not None else None
        if value is not None:
            year = getattr(value, "year", None)
            if isinstance(year, int) and 1980 <= year <= 2049:
                return year
            match = _YEAR_RE.search(str(value))
            if match:
                return int(match.group(1))

    # 2. Lines that state the paper's own date. These outrank anything inferred.
    head = markdown[:4000]
    for pattern in (
        r"(?:published|submitted|accepted|revised)[^\n]{0,40}?\b(19[89]\d|20[0-4]\d)\b",
        r"\barxiv:[^\n]{0,60}?\b(19[89]\d|20[0-4]\d)\b",
        r"(?:©|\(c\)|copyright)\s*(19[89]\d|20[0-4]\d)\b",
    ):
        match = re.search(pattern, head, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # 3. Ask Semantic Scholar about the title. This beats scraping a year out of
    #    the body, where the first "(YYYY)" is usually a citation rather than the
    #    paper's own date.
    if title:
        try:
            from paper_dissector.tools.semantic_scholar import lookup_paper_by_title

            record = lookup_paper_by_title(title)
            year = (record or {}).get("year")
            if isinstance(year, int) and 1980 <= year <= 2049:
                return year
        except Exception as exc:
            log.debug("Semantic Scholar year lookup failed: %s", exc)

    # 4. Last resort: the most recent plausible year in the header block.
    years = [int(y) for y in _YEAR_RE.findall(head)]
    if years:
        log.info("falling back to a year scraped from the paper text; may be a citation")
        return max(years)

    log.info("could not determine publication year; staleness checks will be skipped")
    return None


def extract_authors(doc, markdown: str) -> list[str]:
    """Author names, if docling labelled them. Empty list when unavailable."""
    authors: list[str] = []
    for item in getattr(doc, "texts", None) or []:
        label = str(getattr(item, "label", "")).lower()
        if "author" in label:
            text = (getattr(item, "text", "") or "").strip()
            if text:
                parts = re.split(r",| and ", text)
                authors.extend(p.strip() for p in parts if 2 < len(p.strip()) < 60)
    # De-duplicate while preserving order.
    seen = set()
    return [a for a in authors if not (a in seen or seen.add(a))][:20]


def parse_pdf(pdf_path: str) -> dict:
    """
    Parse a scientific PDF into structured markdown and extract figures.

    Returns:
        {
            "markdown": str,          # full structured markdown
            "title": str,
            "authors": list[str],
            "year": int | None,
            "figures": [              # extracted figure images
                {
                    "figure_id": "figure_1",
                    "kind": "figure",        # or "table"
                    "figure_number": "figure:1" | None,
                    "image_b64": "...",      # base64 PNG
                    "caption": "...",
                    "page": 3,
                }
            ]
        }
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    converter = _build_converter()
    result = converter.convert(pdf_path)
    doc = result.document

    markdown = doc.export_to_markdown()

    title = extract_title(doc, pdf_path, markdown)
    authors = extract_authors(doc, markdown)
    year = extract_year(markdown, title, doc)

    figures = _extract_figures(doc)
    log.info(
        "parsed %s: %d chars of markdown, %d figures, title=%r, year=%s",
        Path(pdf_path).name, len(markdown), len(figures), title[:60], year,
    )

    return {
        "markdown": markdown,
        "title": title,
        "authors": authors,
        "year": year,
        "figures": figures,
    }


def _extract_figures(doc) -> list[dict]:
    """Collect rendered picture and table images with their captions."""
    try:
        from docling_core.types.doc import PictureItem, TableItem
    except ImportError:  # pragma: no cover - very old docling-core
        PictureItem = TableItem = ()  # type: ignore[assignment]

    figures: list[dict] = []
    counters = {"figure": 0, "table": 0}

    try:
        items = list(doc.iterate_items())
    except Exception as exc:
        log.warning("could not iterate document items: %s", exc)
        return figures

    for entry in items:
        # iterate_items() yields (item, level); older versions yielded bare items.
        element = entry[0] if isinstance(entry, tuple) else entry

        if PictureItem and isinstance(element, PictureItem):
            kind = "figure"
        elif TableItem and isinstance(element, TableItem):
            kind = "table"
        else:
            continue

        pil_image = _element_image(element, doc)
        b64 = _to_b64_png(pil_image)
        if not b64:
            continue

        counters[kind] += 1
        caption = _element_caption(element, doc)
        figures.append({
            "figure_id": f"{kind}_{counters[kind]}",
            "kind": kind,
            # Filled in below: caption numbers first, then positional fallbacks.
            "figure_number": _figure_number(caption, kind),
            "image_b64": b64,
            "caption": caption,
            "page": _element_page(element),
        })

    # A caption number is authoritative. Only after those are reserved do we give
    # uncaptioned elements a positional number, skipping any already taken — a
    # stray uncaptioned graphic must not shadow the real "Figure 2".
    taken = {f["figure_number"] for f in figures if f["figure_number"]}
    positions = {"figure": 0, "table": 0}
    for fig in figures:
        if fig["figure_number"]:
            continue
        kind = fig["kind"]
        while True:
            positions[kind] += 1
            candidate = f"{kind}:{positions[kind]}"
            if candidate not in taken:
                break
        fig["figure_number"] = candidate
        taken.add(candidate)

    if not figures:
        log.info("no figure images extracted — visual verification will be skipped")
    return figures
