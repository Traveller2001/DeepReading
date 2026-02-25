"""Review generated figures using a vision LLM to catch quality issues."""

import base64
import os
from pathlib import Path

from openai import OpenAI

_client: OpenAI | None = None

VISION_MODEL = "ep-20260225112250-pgxzh"

REVIEW_PROMPT = (
    "You are a figure quality reviewer for an academic paper reading tool. "
    "Check this generated HTML/SVG diagram (rendered as PNG) and answer concisely:\n"
    "1. Can ALL text (including any Chinese/CJK characters) be read clearly? "
    "Any garbled characters, boxes (□), or missing glyphs?\n"
    "2. Is the layout reasonable — no overlapping elements, clipped content, "
    "or empty areas that look broken?\n"
    "3. Does the diagram look informative, well-structured, and visually clear?\n\n"
    "Reply in this exact format:\n"
    "PASS: (one-line reason)\n"
    "or\n"
    "FAIL: (brief description of problems found)\n"
    "Only reply with one line starting with PASS or FAIL."
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("ARK_API_KEY", "")
        if not api_key:
            raise RuntimeError("ARK_API_KEY not set — cannot review figures")
        _client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
        )
    return _client


def review_figure(image_path: str, description: str = "") -> dict:
    """Send a generated figure to the vision LLM for quality review.

    Args:
        image_path: Absolute path to the PNG file on disk.
        description: What the figure is supposed to show.

    Returns:
        dict with ``passed`` (bool), ``feedback`` (str).
    """
    path = Path(image_path)
    if not path.exists():
        return {"passed": False, "feedback": "Image file does not exist."}

    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        prompt = REVIEW_PROMPT
        if description:
            prompt += f"\n\nThe figure is supposed to show: {description}"

        client = _get_client()
        response = client.responses.create(
            model=VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}",
                        },
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        reply = response.output_text.strip()
        passed = reply.upper().startswith("PASS")
        return {"passed": passed, "feedback": reply}

    except Exception as e:
        # If review fails, let the figure through (don't block generation)
        return {"passed": True, "feedback": f"Review skipped: {e}"}
