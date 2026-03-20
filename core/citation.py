"""Citation normalization, postprocessing, and y-position enhancement."""

import re

from tools.pdf_tools import PdfToolContext, locate_quote as pdf_locate_quote
from tools.html_tools import HtmlToolContext, locate_quote as html_locate_quote


# ---------------------------------------------------------------------------
# Normalize curly/typographic quotes in citations to ASCII quotes
# ---------------------------------------------------------------------------

def normalize_citation_quotes(text: str) -> str:
    """Normalize citation format inside [[p.N ...]] brackets:
    1. Remove stray whitespace after 'p.' so [[p. 2 ...]] → [[p.2 ...]]
    2. Convert curly/typographic quotes to ASCII quotes.
    3. Collapse multiple quoted strings to just the first
       e.g. [[p.2 "q1" "q2"]] → [[p.2 "q1"]]
    """
    def fix(m):
        s = m.group(0)
        # 1. Normalize spacing: [[p. 2 → [[p.2
        s = re.sub(r'\[\[p\.\s+', '[[p.', s)
        # 2. Curly quotes → ASCII
        s = (s.replace('\u201C', '"').replace('\u201D', '"')
              .replace('\u2018', "'").replace('\u2019', "'"))
        # 3. Collapse extra quoted strings after the first
        s = re.sub(r'("(?:[^"]*)")(\s+"[^"]*")+(\]\])', r'\1\3', s)
        return s
    return re.sub(r'\[\[p\.\s*[^\]]*?\]\]', fix, text)


# ---------------------------------------------------------------------------
# Fallback: inject citations if LLM didn't produce them
# ---------------------------------------------------------------------------

def postprocess_citations(report: str, full_text: str) -> str:
    """If the LLM failed to produce [[p.N]] citations at all, inject them."""
    if re.search(r"\[\[p\.\s*\d+", report):
        return report  # citations already present

    page_markers = list(re.finditer(r"--- Page (\d+) ---", full_text))
    if not page_markers:
        return report

    def _find_page(snippet: str) -> tuple[int | None, str]:
        words = snippet.split()
        for length in (8, 5, 3):
            if len(words) >= length:
                phrase = " ".join(words[:length])
                pos = full_text.find(phrase)
                if pos != -1:
                    page = 1
                    for m in page_markers:
                        if m.start() <= pos:
                            page = int(m.group(1))
                        else:
                            break
                    return page, phrase
        return None, ""

    lines = report.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("#")
            or stripped.startswith("!")
            or stripped.startswith("![")
            or not stripped
            or len(stripped) < 30
        ):
            result.append(line)
            continue

        page, quote = _find_page(stripped)
        if page is not None:
            if quote:
                line = line + f' [[p.{page} "{quote}"]]'
            else:
                line = line + f" [[p.{page}]]"
        result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Post-processing: enhance citations with y-positions
# ---------------------------------------------------------------------------

def enhance_citations_with_positions(report: str, pdf_path: str) -> str:
    """Add y-positions to citations that lack them (PDF mode)."""
    # Match [[p.N "quote"]] without :Y
    pattern = re.compile(r'\[\[p\.(\d+)\s+"([^"]+)"\]\]')

    if not pattern.search(report):
        return report  # nothing to enhance

    try:
        with PdfToolContext(pdf_path) as ctx:

            def replacer(match):
                page = int(match.group(1))
                quote = match.group(2)
                result = pdf_locate_quote(ctx, quote, page_hint=page)
                if result.get("found") and "y" in result:
                    return f'[[p.{result["page"]}:{result["y"]} "{quote}"]]'
                return match.group(0)

            return pattern.sub(replacer, report)
    except Exception:
        return report  # if PDF can't be opened, leave as-is


def enhance_citations_html(report: str, full_text: str) -> str:
    """Add y-positions to citations that lack them (HTML mode)."""
    pattern = re.compile(r'\[\[p\.(\d+)\s+"([^"]+)"\]\]')

    if not pattern.search(report):
        return report

    try:
        ctx = HtmlToolContext(full_text)

        def replacer(match):
            page = int(match.group(1))
            quote = match.group(2)
            result = html_locate_quote(ctx, quote, page_hint=page)
            if result.get("found") and "y" in result:
                return f'[[p.{result["page"]}:{result["y"]} "{quote}"]]'
            return match.group(0)

        return pattern.sub(replacer, report)
    except Exception:
        return report
