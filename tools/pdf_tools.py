"""PDF analysis tools for LLM function calling.

Provides tools that the LLM can invoke during report generation to
inspect the paper with spatial precision (page + y-position).
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

# Normalized Y scale: 0 = page top, 1000 = page bottom
Y_SCALE = 1000


# ---------------------------------------------------------------------------
# PDF Tool Context — holds an open document for reuse across tool calls
# ---------------------------------------------------------------------------

class PdfToolContext:
    """Holds an open fitz.Document for reuse across tool calls within one generation."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc: fitz.Document = fitz.open(pdf_path)

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_y(page: fitz.Page, y_abs: float) -> int:
    """Convert absolute y coordinate to 0-1000 normalized scale."""
    page_height = page.rect.height
    if page_height <= 0:
        return 0
    return int(round((y_abs / page_height) * Y_SCALE))


def _block_text(block: dict) -> str:
    """Extract full text from a dict-mode text block."""
    parts = []
    for line in block.get("lines", []):
        for span in line["spans"]:
            parts.append(span["text"])
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Tool 1: get_paper_structure
# ---------------------------------------------------------------------------

def get_paper_structure(ctx: PdfToolContext) -> dict:
    """Extract section headings with page numbers and y-positions."""
    doc = ctx.doc
    sections: list[dict] = []

    # Strategy 1: PDF TOC (outline/bookmarks)
    toc = doc.get_toc(simple=True)  # [level, title, page]
    if len(toc) >= 3:
        for level, title, page_num in toc:
            y_norm = 0
            if 1 <= page_num <= doc.page_count:
                # Search for the heading text on the page to get y-position
                page = doc[page_num - 1]
                rects = page.search_for(title[:60], quads=False)
                if rects:
                    y_norm = _normalize_y(page, rects[0].y0)
            sections.append({
                "level": level,
                "title": title.strip(),
                "page": page_num,
                "y": y_norm,
            })
        return {"sections": sections}

    # Strategy 2: Font-size heuristic
    body_sizes: list[float] = []
    heading_candidates: list[dict] = []

    scan_pages = min(doc.page_count, 25)
    for page_idx in range(scan_pages):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            text = _block_text(block)
            if not text or len(text) < 2:
                continue
            lines = block.get("lines", [])
            sizes = [s["size"] for l in lines for s in l["spans"]]
            if not sizes:
                continue
            avg_size = round(sum(sizes) / len(sizes), 1)
            body_sizes.append(avg_size)
            has_bold = any(
                "bold" in s.get("font", "").lower()
                for l in lines for s in l["spans"]
            )
            heading_candidates.append({
                "size": avg_size,
                "text": text,
                "page": page_idx + 1,
                "y_abs": block["bbox"][1],
                "bold": has_bold,
            })

    if not body_sizes:
        return {"sections": []}

    body_size = Counter(body_sizes).most_common(1)[0][0]

    HEADING_RE = re.compile(
        r"^(\d+\.?\s|[IVXLC]+\.?\s|[A-Z]\.?\s|Abstract|Introduction|Conclusion|"
        r"Related Work|Experiments|Results|Discussion|Method|References|Appendix)",
        re.IGNORECASE,
    )

    for cand in heading_candidates:
        text = cand["text"]
        if len(text) > 120 or len(text) < 3:
            continue
        is_larger = cand["size"] > body_size + 0.5
        is_bold_larger = cand["bold"] and cand["size"] >= body_size
        if not (is_larger or is_bold_larger):
            continue
        if not HEADING_RE.match(text):
            continue

        size_diff = cand["size"] - body_size
        level = 1 if size_diff > 3 else (2 if size_diff > 1 else 3)
        page = doc[cand["page"] - 1]

        sections.append({
            "level": level,
            "title": text[:100],
            "page": cand["page"],
            "y": _normalize_y(page, cand["y_abs"]),
        })

    # Deduplicate
    seen: set[tuple] = set()
    unique = []
    for s in sections:
        key = (s["page"], s["title"][:50])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return {"sections": unique}


# ---------------------------------------------------------------------------
# Tool 2: read_page_detail
# ---------------------------------------------------------------------------

def read_page_detail(ctx: PdfToolContext, page_num: int) -> dict:
    """Get detailed text blocks of a specific page with y-positions."""
    doc = ctx.doc
    if page_num < 1 or page_num > doc.page_count:
        return {"error": f"Page {page_num} out of range (1-{doc.page_count})"}

    page = doc[page_num - 1]
    blocks_data = page.get_text("dict")["blocks"]
    result_blocks = []

    for block in blocks_data:
        if block["type"] != 0:
            continue
        text = _block_text(block)
        if not text.strip():
            continue

        bbox = block["bbox"]
        lines = block.get("lines", [])
        sizes = [s["size"] for l in lines for s in l["spans"]]
        avg_size = round(sum(sizes) / len(sizes), 1) if sizes else 0

        result_blocks.append({
            "text": text[:500],
            "y_start": _normalize_y(page, bbox[1]),
            "y_end": _normalize_y(page, bbox[3]),
            "font_size": avg_size,
        })

    return {"page": page_num, "total_pages": doc.page_count, "blocks": result_blocks}


# ---------------------------------------------------------------------------
# Tool 3: search_text
# ---------------------------------------------------------------------------

def search_text(ctx: PdfToolContext, query: str, max_results: int = 10) -> dict:
    """Search for text across all pages, return matches with positions and context."""
    doc = ctx.doc
    matches: list[dict] = []
    query_stripped = query.strip()
    if not query_stripped:
        return {"query": query, "matches": []}

    for page_idx in range(doc.page_count):
        if len(matches) >= max_results:
            break
        page = doc[page_idx]
        rects = page.search_for(query_stripped, quads=False)
        if not rects:
            continue

        page_text = page.get_text("text")
        query_lower = query_stripped.lower()
        text_lower = page_text.lower()

        pos = 0
        for rect in rects:
            if len(matches) >= max_results:
                break
            idx = text_lower.find(query_lower, pos)
            if idx == -1:
                idx = pos  # fallback

            ctx_start = max(0, idx - 60)
            ctx_end = min(len(page_text), idx + len(query_stripped) + 60)
            context = page_text[ctx_start:ctx_end].replace("\n", " ").strip()
            if ctx_start > 0:
                context = "..." + context
            if ctx_end < len(page_text):
                context = context + "..."

            matches.append({
                "page": page_idx + 1,
                "y": _normalize_y(page, rect.y0),
                "context": context,
                "exact_match": page_text[idx : idx + len(query_stripped)]
                if idx < len(page_text)
                else query_stripped,
            })
            pos = idx + len(query_stripped)

    return {"query": query_stripped, "matches": matches[:max_results]}


# ---------------------------------------------------------------------------
# Tool 4: get_figure_context
# ---------------------------------------------------------------------------

def get_figure_context(ctx: PdfToolContext, figure_caption: str) -> dict:
    """Get text surrounding a figure identified by its caption."""
    doc = ctx.doc
    caption_lower = figure_caption.strip().lower()
    if not caption_lower:
        return {"caption_found": False, "query": figure_caption}

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []

        for block in blocks:
            if block["type"] != 0:
                continue
            text = _block_text(block)
            if text:
                text_blocks.append({"text": text, "bbox": block["bbox"]})

        for j, tb in enumerate(text_blocks):
            if caption_lower in tb["text"].lower():
                text_before = text_blocks[j - 1]["text"][:300] if j > 0 else ""
                text_after = (
                    text_blocks[j + 1]["text"][:300]
                    if j < len(text_blocks) - 1
                    else ""
                )
                return {
                    "caption_found": True,
                    "page": page_idx + 1,
                    "y": _normalize_y(page, tb["bbox"][1]),
                    "caption_text": tb["text"][:200],
                    "text_before": text_before,
                    "text_after": text_after,
                }

    return {"caption_found": False, "query": figure_caption}


# ---------------------------------------------------------------------------
# Tool 5: locate_quote
# ---------------------------------------------------------------------------

def locate_quote(ctx: PdfToolContext, quote: str, page_hint: int = 0) -> dict:
    """Find exact position of a verbatim quote in the PDF."""
    doc = ctx.doc
    quote_stripped = quote.strip()
    if not quote_stripped:
        return {"found": False, "quote": quote}

    def _search_page(page_idx: int) -> dict | None:
        page = doc[page_idx]
        # Exact search
        rects = page.search_for(quote_stripped, quads=False)
        if rects:
            return {
                "found": True,
                "page": page_idx + 1,
                "y": _normalize_y(page, rects[0].y0),
                "matched_text": quote_stripped,
            }
        # Flexible whitespace match
        words = quote_stripped.split()
        if len(words) >= 2:
            pattern = r"\s+".join(re.escape(w) for w in words)
            page_text = page.get_text("text")
            m = re.search(pattern, page_text, re.IGNORECASE)
            if m:
                snippet = " ".join(words[:3])
                rects2 = page.search_for(snippet, quads=False)
                y = _normalize_y(page, rects2[0].y0) if rects2 else 500
                return {
                    "found": True,
                    "page": page_idx + 1,
                    "y": y,
                    "matched_text": m.group()[:200],
                }
        return None

    # Search page_hint first if provided
    pages = list(range(doc.page_count))
    if 1 <= page_hint <= doc.page_count:
        hint_idx = page_hint - 1
        pages = [hint_idx] + [i for i in pages if i != hint_idx]

    for page_idx in pages:
        result = _search_page(page_idx)
        if result:
            return result

    return {"found": False, "quote": quote_stripped[:100]}


# ---------------------------------------------------------------------------
# Tool Schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_paper_structure",
            "description": (
                "Extract the section headings and structure of the paper with "
                "page numbers and y-positions. Call this first to understand the "
                "paper's organization."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page_detail",
            "description": (
                "Get the detailed text blocks of a specific page with y-positions. "
                "Useful to examine a particular section or page in detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_num": {
                        "type": "integer",
                        "description": "1-based page number to read",
                    },
                },
                "required": ["page_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "Search for specific text across all pages of the paper. Returns "
                "matches with page numbers, y-positions, and surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for (case-insensitive)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_figure_context",
            "description": (
                "Get the text surrounding a figure or table to understand its context. "
                "Pass the figure label like 'Figure 3' or part of its caption."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "figure_caption": {
                        "type": "string",
                        "description": "Figure label or caption text to search for",
                    },
                },
                "required": ["figure_caption"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_quote",
            "description": (
                "Find the exact page and y-position of a verbatim quote in the paper. "
                "Use this to get precise citation coordinates for [[p.N:Y \"quote\"]] format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quote": {
                        "type": "string",
                        "description": "The verbatim text to locate in the paper",
                    },
                    "page_hint": {
                        "type": "integer",
                        "description": "Optional page number where you expect the quote (1-based)",
                    },
                },
                "required": ["quote"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_figure",
            "description": (
                "Generate a high-quality explanatory diagram by writing HTML/CSS/SVG "
                "code. The code is rendered in a headless browser and saved as a PNG "
                "image. Write the BODY content only — the outer <html>/<body> tags, "
                "fonts, and padding are provided automatically. After calling this "
                "tool, insert the returned image path in your report using "
                "![description](path) syntax."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "HTML/CSS/SVG code for the diagram body content. "
                            "Use <style> for CSS, <div> for layout, <svg> for "
                            "vector graphics. Do NOT include <html> or <body> tags."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Brief description used as filename and alt text "
                            "(e.g. 'model_architecture', 'training_pipeline')"
                        ),
                    },
                },
                "required": ["code", "description"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(ctx: PdfToolContext, tool_name: str, arguments: dict) -> Any:
    """Execute a PDF tool by name. Returns JSON-serializable result."""
    _TOOL_MAP = {
        "get_paper_structure": lambda args: get_paper_structure(ctx),
        "read_page_detail": lambda args: read_page_detail(ctx, args.get("page_num", 1)),
        "search_text": lambda args: search_text(ctx, args.get("query", "")),
        "get_figure_context": lambda args: get_figure_context(
            ctx, args.get("figure_caption", "")
        ),
        "locate_quote": lambda args: locate_quote(
            ctx, args.get("quote", ""), args.get("page_hint", 0)
        ),
    }

    handler = _TOOL_MAP.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return handler(arguments)
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {e}"}
