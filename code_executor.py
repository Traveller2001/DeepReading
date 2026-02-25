"""Render HTML/SVG diagrams to PNG using a headless browser (Playwright)."""

import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "figures"

# HTML template that wraps user content with proper fonts and a tight layout.
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    display: inline-block;
    background: white;
    font-family: system-ui, -apple-system, "Noto Sans CJK SC", "Noto Sans CJK", sans-serif;
    color: #1e293b;
    padding: 32px;
  }
</style>
</head>
<body>
%s
</body>
</html>
"""


def execute_html_figure(code: str, paper_id: str, fig_name: str) -> dict:
    """Render user-provided HTML/SVG code to a PNG image.

    Args:
        code: HTML/CSS/SVG code for the diagram.  Should NOT include
              ``<html>`` or ``<body>`` — those are provided by the template.
        paper_id: Paper identifier (used to namespace output files).
        fig_name: Short description used as the filename (sanitized).

    Returns:
        dict with ``success`` and ``path`` (on success) or ``error``.
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", fig_name)[:60] or "figure"

    out_dir = DATA_DIR / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gen_{safe_name}.png"

    full_html = _HTML_TEMPLATE % code

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(device_scale_factor=2)  # 2x for retina
            page.set_content(full_html)
            page.wait_for_timeout(500)

            # Screenshot the body content (auto-crops to content size)
            element = page.query_selector("body")
            if element:
                element.screenshot(path=str(out_path))
            else:
                page.screenshot(path=str(out_path))
            browser.close()

    except Exception as e:
        return {"success": False, "error": f"Render failed: {e}"}

    if not out_path.exists():
        return {"success": False, "error": "Image file was not created"}

    web_path = f"/data/figures/{paper_id}/gen_{safe_name}.png"
    return {"success": True, "path": web_path}
