"""HTML-based analysis tools for LLM function calling.

Provides the same tool signatures as pdf_tools.py but operates on plain text
(virtual pages split from HTML content). This allows the LLM to use identical
tool schemas for both PDF and HTML sources.
"""

import re
from typing import Any

# Normalized Y scale: 0 = page top, 1000 = page bottom
Y_SCALE = 1000


# ---------------------------------------------------------------------------
# HTML Tool Context — holds parsed virtual pages
# ---------------------------------------------------------------------------

class HtmlToolContext:
    """Holds parsed virtual pages for reuse across tool calls."""

    def __init__(self, full_text: str):
        self.full_text = full_text
        self.pages: dict[int, str] = {}
        self._parse_pages()

    def _parse_pages(self):
        """Parse `--- Page N ---` markers into page dict."""
        parts = re.split(r"--- Page (\d+) ---\n?", self.full_text)
        # parts = ['', '1', 'page1 text', '2', 'page2 text', ...]
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            page_text = parts[i + 1] if i + 1 < len(parts) else ""
            self.pages[page_num] = page_text

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tool 1: get_paper_structure
# ---------------------------------------------------------------------------

# Patterns that look like section headings
_HEADING_RE = re.compile(
    r"^(\d+\.?\s|[IVXLC]+\.?\s|[A-Z]\.?\s|Abstract|Introduction|Conclusion|"
    r"Related Work|Experiments|Results|Discussion|Method|References|Appendix|"
    r"Background|Evaluation|Implementation|Overview|Acknowledgment)",
    re.IGNORECASE,
)


def get_paper_structure(ctx: HtmlToolContext) -> dict:
    """Extract section headings with page numbers and y-positions."""
    sections = []

    for page_num, page_text in sorted(ctx.pages.items()):
        lines = page_text.split("\n")
        page_len = len(page_text) or 1

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) > 120 or len(line_stripped) < 3:
                continue

            # Check if it looks like a heading
            if not _HEADING_RE.match(line_stripped):
                continue

            # Headings are typically short and may be title case
            if len(line_stripped) > 80:
                continue

            # Compute y-position by character offset
            char_offset = page_text.find(line_stripped)
            y_norm = int((char_offset / page_len) * Y_SCALE) if char_offset >= 0 else 0

            # Determine heading level
            level = 2  # default
            if re.match(r"^\d+\.\s", line_stripped):
                level = 2
            elif re.match(r"^\d+\.\d+", line_stripped):
                level = 3
            elif line_stripped.lower() in ("abstract", "references", "appendix"):
                level = 1

            sections.append({
                "level": level,
                "title": line_stripped[:100],
                "page": page_num,
                "y": y_norm,
            })

    return {"sections": sections}


# ---------------------------------------------------------------------------
# Tool 2: read_page_detail
# ---------------------------------------------------------------------------

def read_page_detail(ctx: HtmlToolContext, page_num: int) -> dict:
    """Get detailed text blocks of a specific virtual page with y-positions."""
    if page_num not in ctx.pages:
        return {"error": f"Page {page_num} out of range (1-{ctx.page_count})"}

    page_text = ctx.pages[page_num]
    # Split on double newlines to get "blocks"
    raw_blocks = re.split(r"\n{2,}", page_text)
    page_len = len(page_text) or 1

    result_blocks = []
    for block_text in raw_blocks:
        block_text = block_text.strip()
        if not block_text:
            continue

        # Compute y-position by character offset
        char_offset = page_text.find(block_text)
        y_start = int((char_offset / page_len) * Y_SCALE) if char_offset >= 0 else 0
        y_end = int(((char_offset + len(block_text)) / page_len) * Y_SCALE) if char_offset >= 0 else Y_SCALE

        result_blocks.append({
            "text": block_text[:500],
            "y_start": y_start,
            "y_end": min(y_end, Y_SCALE),
            "font_size": 12.0,  # placeholder since HTML doesn't have font info
        })

    return {"page": page_num, "total_pages": ctx.page_count, "blocks": result_blocks}


# ---------------------------------------------------------------------------
# Tool 3: search_text
# ---------------------------------------------------------------------------

def search_text(ctx: HtmlToolContext, query: str, max_results: int = 10) -> dict:
    """Search for text across all virtual pages with positions and context."""
    query_stripped = query.strip()
    if not query_stripped:
        return {"query": query, "matches": []}

    matches = []
    query_lower = query_stripped.lower()

    for page_num, page_text in sorted(ctx.pages.items()):
        if len(matches) >= max_results:
            break

        text_lower = page_text.lower()
        page_len = len(page_text) or 1
        pos = 0

        while pos < len(text_lower) and len(matches) < max_results:
            idx = text_lower.find(query_lower, pos)
            if idx == -1:
                break

            # Compute y-position
            y_norm = int((idx / page_len) * Y_SCALE)

            # Extract context
            ctx_start = max(0, idx - 60)
            ctx_end = min(len(page_text), idx + len(query_stripped) + 60)
            context = page_text[ctx_start:ctx_end].replace("\n", " ").strip()
            if ctx_start > 0:
                context = "..." + context
            if ctx_end < len(page_text):
                context = context + "..."

            matches.append({
                "page": page_num,
                "y": y_norm,
                "context": context,
                "exact_match": page_text[idx: idx + len(query_stripped)],
            })

            pos = idx + len(query_stripped)

    return {"query": query_stripped, "matches": matches[:max_results]}


# ---------------------------------------------------------------------------
# Tool 4: get_figure_context
# ---------------------------------------------------------------------------

def get_figure_context(ctx: HtmlToolContext, figure_caption: str) -> dict:
    """Get text surrounding a figure identified by its caption."""
    caption_lower = figure_caption.strip().lower()
    if not caption_lower:
        return {"caption_found": False, "query": figure_caption}

    for page_num, page_text in sorted(ctx.pages.items()):
        text_lower = page_text.lower()
        idx = text_lower.find(caption_lower)
        if idx == -1:
            # Try partial match
            words = caption_lower.split()
            if len(words) >= 3:
                partial = " ".join(words[:3])
                idx = text_lower.find(partial)
            if idx == -1:
                continue

        page_len = len(page_text) or 1
        y_norm = int((idx / page_len) * Y_SCALE)

        # Get surrounding text
        ctx_start = max(0, idx - 300)
        ctx_end = min(len(page_text), idx + len(caption_lower) + 300)

        return {
            "caption_found": True,
            "page": page_num,
            "y": y_norm,
            "caption_text": page_text[idx: idx + 200],
            "text_before": page_text[ctx_start:idx][-300:],
            "text_after": page_text[idx + len(caption_lower): ctx_end][:300],
        }

    return {"caption_found": False, "query": figure_caption}


# ---------------------------------------------------------------------------
# Tool 5: locate_quote
# ---------------------------------------------------------------------------

def locate_quote(ctx: HtmlToolContext, quote: str, page_hint: int = 0) -> dict:
    """Find exact position of a verbatim quote in the text."""
    quote_stripped = quote.strip()
    if not quote_stripped:
        return {"found": False, "quote": quote}

    def _search_page(page_num: int) -> dict | None:
        page_text = ctx.pages.get(page_num, "")
        if not page_text:
            return None

        page_len = len(page_text) or 1

        # Exact match
        idx = page_text.lower().find(quote_stripped.lower())
        if idx >= 0:
            y_norm = int((idx / page_len) * Y_SCALE)
            return {
                "found": True,
                "page": page_num,
                "y": y_norm,
                "matched_text": page_text[idx: idx + len(quote_stripped)],
            }

        # Flexible whitespace match
        words = quote_stripped.split()
        if len(words) >= 2:
            pattern = r"\s+".join(re.escape(w) for w in words)
            m = re.search(pattern, page_text, re.IGNORECASE)
            if m:
                y_norm = int((m.start() / page_len) * Y_SCALE)
                return {
                    "found": True,
                    "page": page_num,
                    "y": y_norm,
                    "matched_text": m.group()[:200],
                }

        return None

    # Search page_hint first
    page_nums = sorted(ctx.pages.keys())
    if page_hint in ctx.pages:
        page_nums = [page_hint] + [p for p in page_nums if p != page_hint]

    for page_num in page_nums:
        result = _search_page(page_num)
        if result:
            return result

    return {"found": False, "quote": quote_stripped[:100]}


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(ctx: HtmlToolContext, tool_name: str, arguments: dict) -> Any:
    """Execute an HTML tool by name. Returns JSON-serializable result."""
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
