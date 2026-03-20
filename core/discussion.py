"""Reader-Writer discussion rounds and report polishing."""

import asyncio
import json
import logging
import re

from llm_client import generate_stream
from core.database import update_report, update_discussion
from core.prompts import (
    READER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    POLISH_SYSTEM_PROMPT,
    LANG_INSTRUCTIONS,
)
from core.citation import (
    normalize_citation_quotes,
    postprocess_citations,
    enhance_citations_with_positions,
    enhance_citations_html,
)
from config import (
    LLM_MODEL as MODEL,
    LLM_TEMPERATURE as TEMPERATURE,
    MAX_REPORT_TOKENS,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

DISCUSSION_ROUNDS = 3
MAX_PAPER_TEXT_LEN = 30000  # truncate paper text for writer/polish context


async def _stream_simple_completion(messages, max_tokens=4096):
    """Async generator for streaming a simple (no tool-calling) LLM completion."""
    async for sc in generate_stream(
        messages,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    ):
        if sc.content:
            yield sc.content


async def generate_discussion_stream(paper: dict, figures: list[dict], report: str, lang: str = "en"):
    """Async generator yielding JSON events for the discussion + polish flow.

    SSE Protocol:
      {"type": "discussion_start"}
      {"type": "discussion_round", "round": N, "total": 3}
      {"type": "reader_chunk", "round": N, "content": "..."}
      {"type": "writer_chunk", "round": N, "content": "..."}
      {"type": "discussion_end"}
      {"type": "polish_start"}
      {"type": "polish_chunk", "content": "..."}
      {"type": "polish_end", "report": "full polished markdown"}
    """
    lang_inst = LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS["en"])

    # Truncate paper text for context
    paper_text = paper.get("full_text", "")
    if len(paper_text) > MAX_PAPER_TEXT_LEN:
        paper_text = paper_text[:MAX_PAPER_TEXT_LEN] + "\n\n... (truncated)"

    # Build figure list summary
    fig_summary = ""
    if figures:
        fig_lines = []
        for fig in figures:
            caption = fig.get("caption", "")
            fig_lines.append(f"- {caption} (page {fig['page_num'] + 1})")
        fig_summary = "\n## Figures in the paper:\n" + "\n".join(fig_lines)

    discussion_messages = []  # collect all messages for saving

    yield {"type": "discussion_start"}

    discussion_context = ""  # accumulates reader/writer exchanges

    for round_num in range(1, DISCUSSION_ROUNDS + 1):
        yield {"type": "discussion_round", "round": round_num, "total": DISCUSSION_ROUNDS}

        # --- Reader asks a question ---
        reader_messages = [
            {"role": "system", "content": READER_SYSTEM_PROMPT + f"\n\nLanguage requirement: {lang_inst}"},
            {"role": "user", "content": (
                f"Here is the report to review:\n\n{report}\n\n"
                + (f"Previous discussion:\n{discussion_context}\n\n" if discussion_context else "")
                + f"This is round {round_num} of {DISCUSSION_ROUNDS}. Ask ONE new question about an aspect not yet covered."
            )},
        ]

        reader_text = []
        async for chunk in _stream_simple_completion(reader_messages):
            reader_text.append(chunk)
            yield {"type": "reader_chunk", "round": round_num, "content": chunk}

        reader_full = "".join(reader_text)
        discussion_context += f"\n### Question (Round {round_num}):\n{reader_full}\n"
        discussion_messages.append({"role": "reader", "round": round_num, "content": reader_full})

        # --- Writer answers the question ---
        writer_messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT + f"\n\nLanguage requirement: {lang_inst}"},
            {"role": "user", "content": (
                f"## Original paper text:\n{paper_text}\n"
                f"{fig_summary}\n\n"
                f"## Your report:\n{report}\n\n"
                f"## Reader's question (Round {round_num}):\n{reader_full}\n\n"
                "Answer the question with evidence from the paper."
            )},
        ]

        writer_text = []
        async for chunk in _stream_simple_completion(writer_messages):
            writer_text.append(chunk)
            yield {"type": "writer_chunk", "round": round_num, "content": chunk}

        writer_full = "".join(writer_text)
        discussion_context += f"\n### Writer (Round {round_num}):\n{writer_full}\n"
        discussion_messages.append({"role": "writer", "round": round_num, "content": writer_full})

    yield {"type": "discussion_end"}

    # Save discussion
    await update_discussion(paper["id"], json.dumps(discussion_messages, ensure_ascii=False), "completed")

    # --- Polish the report ---
    yield {"type": "polish_start"}

    polish_messages = [
        {"role": "system", "content": POLISH_SYSTEM_PROMPT + f"\n\nLanguage requirement: {lang_inst}"},
        {"role": "user", "content": (
            f"## Original paper text:\n{paper_text}\n"
            f"{fig_summary}\n\n"
            f"## Current report:\n{report}\n\n"
            f"## Discussion transcript:\n{discussion_context}\n\n"
            "Revise the report based on the discussion. Output the full revised report."
        )},
    ]

    polished_chunks = []
    async for chunk in _stream_simple_completion(polish_messages, max_tokens=MAX_REPORT_TOKENS):
        polished_chunks.append(chunk)
        yield {"type": "polish_chunk", "content": chunk}

    polished_report = normalize_citation_quotes("".join(polished_chunks))

    # Post-process polished report (non-blocking)
    source_type = paper.get("source_type", "pdf")
    if source_type == "html":
        polished_report = await asyncio.to_thread(
            enhance_citations_html, polished_report, paper.get("full_text", "")
        )
    else:
        pdf_path = str(DATA_DIR / "uploads" / f"{paper['id']}.pdf")
        polished_report = await asyncio.to_thread(
            enhance_citations_with_positions, polished_report, pdf_path
        )
    if not re.search(r"\[\[p\.\s*\d+", polished_report):
        polished_report = postprocess_citations(polished_report, paper.get("full_text", ""))

    await update_report(paper["id"], polished_report)

    yield {"type": "polish_end", "report": polished_report}
