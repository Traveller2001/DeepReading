"""Use vision LLM to scan PDF pages and identify figures/tables with bounding boxes."""

import base64
import io
import json
import logging
import re
import time
from pathlib import Path

import fitz  # PyMuPDF

from llm_client import generate_sync
from config import (
    SCAN_DPI, CROP_DPI, BBOX_PAD,
    VISION_MODEL, VISION_TEMPERATURE, VISION_MAX_TOKENS,
    VISION_TIMEOUT, VISION_BATCH_SIZE as BATCH_SIZE,
    VISION_RETRY_DELAY as RETRY_DELAY,
)

logger = logging.getLogger(__name__)

CROP_SCALE = CROP_DPI / 72

VISION_PROMPT = """\
You are analyzing pages from an academic PDF. For each page image, identify ALL figures, \
tables, charts, diagrams, algorithms, and other visual elements. Do NOT include pure text \
paragraphs, page headers, footers, or page numbers.

For each visual element, return:
- "page": the page number (provided below each image)
- "type": "figure" | "table" | "chart" | "diagram" | "algorithm"
- "label": the label if visible (e.g. "Figure 3", "Table 2"), or ""
- "caption": the full caption text if visible below/above the element, or ""
- "description": 1-2 sentences describing what the visual shows \
(architecture, data trends, comparison, workflow, etc.)
- "bbox": {"top": float, "left": float, "bottom": float, "right": float}
  These are fractions of page dimensions (0.0 = top/left edge, 1.0 = bottom/right edge).
  The bbox should TIGHTLY enclose the visual element AND its caption.

Return ONLY a valid JSON array. If no visual elements exist on any page, return [].
"""


def _render_page_base64(page: fitz.Page, dpi: int = SCAN_DPI) -> str:
    """Render a PDF page as PNG and return base64-encoded string."""
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode()


def _build_batch_messages(
    doc: fitz.Document, page_indices: list[int]
) -> list[dict]:
    """Build the messages array for a batch of pages."""
    content_parts = []

    for page_idx in page_indices:
        page = doc[page_idx]
        b64 = _render_page_base64(page)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
        content_parts.append({
            "type": "text",
            "text": f"Page {page_idx + 1}",
        })

    return [
        {"role": "system", "content": VISION_PROMPT},
        {"role": "user", "content": content_parts},
    ]


def _parse_vision_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        logger.warning("Failed to parse vision response as JSON: %s", text[:200])
    return []


def _crop_figure(
    doc: fitz.Document,
    page_idx: int,
    bbox: dict,
    save_path: Path,
) -> tuple[int, int]:
    """Crop a figure region from a page using bbox fractions. Returns (width, height)."""
    page = doc[page_idx]
    page_rect = page.rect

    clip = fitz.Rect(
        page_rect.x0 + bbox["left"] * page_rect.width - BBOX_PAD,
        page_rect.y0 + bbox["top"] * page_rect.height - BBOX_PAD,
        page_rect.x0 + bbox["right"] * page_rect.width + BBOX_PAD,
        page_rect.y0 + bbox["bottom"] * page_rect.height + BBOX_PAD,
    )
    # Clamp to page bounds
    clip &= page_rect

    mat = fitz.Matrix(CROP_SCALE, CROP_SCALE)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    pix.save(str(save_path))
    return pix.width, pix.height


def extract_figures_with_vision(
    doc: fitz.Document, out_path: Path
) -> list[dict]:
    """Render PDF pages, send to vision LLM, get figure regions + descriptions, crop & save.

    Returns list of dicts with keys:
        fig_index, filename, page_num, width, height, caption, description
    Returns empty list if all API calls fail (caller should fall back).
    """
    total_pages = doc.page_count

    # Build page batches
    batches = []
    for start in range(0, total_pages, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_pages)
        batches.append(list(range(start, end)))

    all_detections = []
    any_success = False

    for batch_idx, page_indices in enumerate(batches):
        logger.info(
            "Vision batch %d/%d: pages %s",
            batch_idx + 1, len(batches),
            [p + 1 for p in page_indices],
        )
        messages = _build_batch_messages(doc, page_indices)

        for attempt in range(2):  # max 2 attempts per batch
            try:
                reply = generate_sync(
                    messages=messages,
                    model=VISION_MODEL,
                    temperature=VISION_TEMPERATURE,
                    max_tokens=VISION_MAX_TOKENS,
                    timeout=VISION_TIMEOUT,
                )
                detections = _parse_vision_response(reply)
                all_detections.extend(detections)
                any_success = True
                logger.info(
                    "Vision batch %d: found %d elements", batch_idx + 1, len(detections)
                )
                break  # success, move to next batch
            except Exception as e:
                logger.warning(
                    "Vision batch %d attempt %d failed: %s",
                    batch_idx + 1, attempt + 1, e,
                )
                if attempt == 0:
                    time.sleep(RETRY_DELAY)
                # On second failure, skip this batch

    if not any_success:
        logger.warning("All vision batches failed, returning empty list")
        return []

    if not all_detections:
        logger.info("Vision found no figures/tables in the document")
        return []

    # Crop and save each detected figure
    figures = []
    fig_idx = 0

    for det in all_detections:
        bbox = det.get("bbox")
        if not bbox:
            continue

        # Validate bbox has required keys and values in range
        try:
            top = float(bbox.get("top", 0))
            left = float(bbox.get("left", 0))
            bottom = float(bbox.get("bottom", 1))
            right = float(bbox.get("right", 1))
        except (TypeError, ValueError):
            continue

        # Sanity check: bbox must have positive area
        if bottom <= top or right <= left:
            continue
        # Clamp to [0, 1]
        top = max(0.0, min(1.0, top))
        left = max(0.0, min(1.0, left))
        bottom = max(0.0, min(1.0, bottom))
        right = max(0.0, min(1.0, right))

        # Page number: vision returns 1-based, we need 0-based index
        page_num = int(det.get("page", 1)) - 1
        if page_num < 0 or page_num >= doc.page_count:
            continue

        fig_idx += 1
        fig_filename = f"fig_{fig_idx}.png"
        fig_path = out_path / fig_filename

        try:
            w, h = _crop_figure(
                doc, page_num,
                {"top": top, "left": left, "bottom": bottom, "right": right},
                fig_path,
            )
        except Exception as e:
            logger.warning("Failed to crop figure %d: %s", fig_idx, e)
            fig_idx -= 1
            continue

        # Build caption from label + caption text (avoid duplication)
        label = det.get("label", "").strip()
        caption_text = det.get("caption", "").strip()
        if caption_text and label and caption_text.lower().startswith(label.lower()):
            # Caption already contains the label, use as-is
            caption = caption_text
        elif label and caption_text:
            caption = f"{label}: {caption_text}"
        elif caption_text:
            caption = caption_text
        elif label:
            caption = label
        else:
            caption = f"{det.get('type', 'Figure')} on page {page_num + 1}"

        figures.append({
            "fig_index": fig_idx,
            "filename": fig_filename,
            "page_num": page_num,
            "width": w,
            "height": h,
            "caption": caption[:300],  # truncate very long captions
            "description": (det.get("description") or "")[:500],
        })

    logger.info("Vision extraction complete: %d figures saved", len(figures))
    return figures
