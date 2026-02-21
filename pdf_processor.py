import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import io


@dataclass
class ExtractedFigure:
    fig_index: int
    filename: str
    page_num: int
    width: int
    height: int
    caption: str = ""


@dataclass
class ProcessedPaper:
    title: str
    authors: str
    abstract: str
    full_text: str
    num_pages: int
    figures: list[ExtractedFigure]


# Minimum thresholds for filtering decorative images
MIN_IMAGE_DIM = 100  # pixels
MIN_IMAGE_BYTES = 5000  # 5KB

# Text truncation limits
MAX_TEXT_CHARS = 60000
KEEP_HEAD = 45000
KEEP_TAIL = 15000

# Rendering
RENDER_DPI = 200
RENDER_SCALE = RENDER_DPI / 72

# Caption must START a text block/line (not inline reference)
# Match "Figure 1 |", "Figure 1:", "Figure 1.", "Table 1 |", "Fig. 1:" etc.
CAPTION_RE = re.compile(
    r"^((?:Figure|Fig\.|Table)\s*\d+)\s*[|:.](.+)",
    re.IGNORECASE | re.DOTALL,
)

PAD_ABOVE = 10
PAD_BELOW = 8


# ---------------------------------------------------------------------------
# Title / Authors / Abstract helpers
# ---------------------------------------------------------------------------

def _extract_title(doc: fitz.Document) -> str:
    meta_title = (doc.metadata.get("title") or "").strip()
    if meta_title and len(meta_title) > 5 and "untitled" not in meta_title.lower():
        return meta_title
    if doc.page_count > 0:
        blocks = doc[0].get_text("dict")["blocks"]
        lines_by_size: dict[float, list[str]] = {}
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block.get("lines", []):
                for span in line["spans"]:
                    text = span["text"].strip()
                    size = round(span["size"], 1)
                    if (
                        text
                        and len(text) > 2
                        and not text.lower().startswith(("arxiv", "preprint", "http"))
                        and not re.match(r"^\d+$", text)
                    ):
                        lines_by_size.setdefault(size, []).append(text)
        if lines_by_size:
            max_size = max(lines_by_size.keys())
            title_parts = lines_by_size[max_size]
            title = " ".join(title_parts)
            if len(title) > 300:
                title = title[:300]
            return title
    return "Untitled Paper"


def _extract_authors(doc: fitz.Document) -> str:
    meta_author = (doc.metadata.get("author") or "").strip()
    return meta_author if meta_author else ""


def _extract_abstract(full_text: str) -> str:
    match = re.search(
        r"(?i)\babstract\b[:\s]*\n?(.*?)(?=\n\s*(?:1[\.\s]|introduction|keywords?\b|I\.\s))",
        full_text,
        re.DOTALL,
    )
    if match:
        abstract = match.group(1).strip()
        if len(abstract) > 3000:
            abstract = abstract[:3000] + "..."
        return abstract
    return ""


def _truncate_text(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return (
        text[:KEEP_HEAD]
        + "\n\n[... middle content truncated for length ...]\n\n"
        + text[-KEEP_TAIL:]
    )


# ---------------------------------------------------------------------------
# Figure / Table region detection
# ---------------------------------------------------------------------------

def _get_block_text(block: dict) -> str:
    """Get full text of a text block."""
    parts = []
    for line in block.get("lines", []):
        for span in line["spans"]:
            parts.append(span["text"])
    return "".join(parts).strip()


def _find_captions_on_page(page: fitz.Page) -> list[dict]:
    """Find real Figure/Table captions (not inline references) on a page.

    A real caption is identified by a text block that STARTS with
    "Figure N |" / "Table N |" etc., i.e. the caption is the primary
    content of that block, not a passing reference in body text.
    """
    blocks = page.get_text("dict")["blocks"]
    results = []

    for block in blocks:
        if block["type"] != 0:
            continue
        text = _get_block_text(block)
        m = CAPTION_RE.match(text)
        if not m:
            continue

        label = m.group(1).strip()              # e.g. "Figure 1"
        desc = m.group(2).strip()[:200]         # first 200 chars of caption
        bbox = fitz.Rect(block["bbox"])

        results.append({
            "label": label,
            "caption": f"{label}: {desc}",
            "rect": bbox,                       # full caption block rect
        })

    # Sort top to bottom
    results.sort(key=lambda c: c["rect"].y0)
    return results


def _find_top_boundary(
    page: fitz.Page,
    caption_y0: float,
    prev_caption_y1: float | None,
) -> float:
    """Determine the top edge of a figure region.

    Walk upward from the caption looking for the nearest visual break:
    - a section heading (larger font, short text like "3. Evaluations")
    - bottom of a previous caption block
    - a body-text paragraph that's clearly separated by a gap
    - page top margin
    """
    blocks = page.get_text("dict")["blocks"]
    page_top = page.rect.y0

    # Collect all text blocks above the caption, sorted bottom-up
    above_blocks = []
    for b in blocks:
        if b["type"] != 0:
            continue
        bbox = fitz.Rect(b["bbox"])
        if bbox.y1 <= caption_y0:
            above_blocks.append((bbox, b))
    above_blocks.sort(key=lambda x: x[0].y1, reverse=True)

    # If there's a previous caption on this page, don't go above it
    hard_top = prev_caption_y1 if prev_caption_y1 is not None else page_top

    # Walk upward looking for the start of non-figure content
    for bbox, b in above_blocks:
        if bbox.y1 <= hard_top:
            break

        text = _get_block_text(b)
        lines = b.get("lines", [])

        # Check if it's a section heading (short text, larger font)
        if lines:
            avg_size = sum(
                s["size"] for l in lines for s in l["spans"]
            ) / max(sum(len(l["spans"]) for l in lines), 1)

            # Section heading: short, larger font, starts with a number
            if (
                len(text) < 80
                and avg_size > 12
                and re.match(r"^\d+[\.\s]", text)
            ):
                return bbox.y0 - PAD_ABOVE

        # Check for body text paragraph: long, multi-line, full-width prose
        page_w = page.rect.width
        if (
            len(lines) >= 3
            and bbox.width > page_w * 0.6
            and len(text) > 150
        ):
            # This looks like a body paragraph, the figure starts below it
            return bbox.y1

    return max(hard_top, page_top)


def _crop_region(page: fitz.Page, region: fitz.Rect, save_path: Path) -> tuple[int, int]:
    """Render and crop a region from a page. Returns (width, height)."""
    mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
    pix = page.get_pixmap(matrix=mat, clip=region, alpha=False)
    pix.save(str(save_path))
    return pix.width, pix.height


def _extract_figure_regions(
    doc: fitz.Document, out_path: Path
) -> list[ExtractedFigure]:
    """Detect and crop individual Figure/Table regions from the PDF."""
    figures = []
    fig_idx = 0

    for page_num in range(doc.page_count):
        page = doc[page_num]
        captions = _find_captions_on_page(page)
        if not captions:
            continue

        prev_caption_y1 = None

        for cap in captions:
            cap_rect = cap["rect"]

            top = _find_top_boundary(page, cap_rect.y0, prev_caption_y1)
            top = max(top - PAD_ABOVE, page.rect.y0)
            bottom = min(cap_rect.y1 + PAD_BELOW, page.rect.y1)

            # Skip degenerate regions
            if bottom - top < 40:
                continue

            fig_idx += 1
            fig_filename = f"fig_{fig_idx}.png"
            fig_path = out_path / fig_filename

            region = fitz.Rect(page.rect.x0, top, page.rect.x1, bottom)
            w, h = _crop_region(page, region, fig_path)

            figures.append(
                ExtractedFigure(
                    fig_index=fig_idx,
                    filename=fig_filename,
                    page_num=page_num,
                    width=w,
                    height=h,
                    caption=cap["caption"],
                )
            )

            prev_caption_y1 = cap_rect.y1

    return figures


def _extract_embedded_images(
    doc: fitz.Document, out_path: Path
) -> list[ExtractedFigure]:
    """Extract embedded raster images from the PDF."""
    figures = []
    seen_xrefs = set()
    fig_idx = 0

    for page_num in range(doc.page_count):
        page = doc[page_num]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            img_bytes = base_image["image"]
            w, h = base_image["width"], base_image["height"]

            if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                continue
            if len(img_bytes) < MIN_IMAGE_BYTES:
                continue

            fig_idx += 1
            fig_filename = f"fig_{fig_idx}.png"
            fig_path = out_path / fig_filename

            try:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode in ("CMYK", "P", "LA", "PA"):
                    img = img.convert("RGB")
                img.save(str(fig_path), "PNG")
            except Exception:
                fig_path.write_bytes(img_bytes)

            figures.append(
                ExtractedFigure(
                    fig_index=fig_idx,
                    filename=fig_filename,
                    page_num=page_num,
                    width=w,
                    height=h,
                    caption=f"Embedded image from page {page_num + 1}",
                )
            )

    return figures


def process_pdf(pdf_bytes: bytes, paper_id: str, output_dir: str) -> ProcessedPaper:
    """Extract text, metadata, and figures from a PDF. Synchronous."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    title = _extract_title(doc)
    authors = _extract_authors(doc)

    pages_text = []
    for i, page in enumerate(doc):
        page_text = page.get_text("text")
        pages_text.append(f"--- Page {i + 1} ---\n{page_text}")
    full_text = "\n".join(pages_text)

    abstract = _extract_abstract(full_text)
    full_text = _truncate_text(full_text)

    # 1. Try caption-based region cropping (best quality)
    # 2. Fall back to embedded raster images
    figures = _extract_figure_regions(doc, out_path)
    if not figures:
        figures = _extract_embedded_images(doc, out_path)

    num_pages = doc.page_count
    doc.close()

    return ProcessedPaper(
        title=title,
        authors=authors,
        abstract=abstract,
        full_text=full_text,
        num_pages=num_pages,
        figures=figures,
    )


def figures_to_dicts(figures: list[ExtractedFigure]) -> list[dict]:
    return [asdict(f) for f in figures]
