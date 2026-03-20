"""Report generation with tool-calling loop and continuation logic."""

import asyncio
import json
import logging
import re

from llm_client import generate_stream
from core.database import update_report
from core.prompts import SYSTEM_PROMPT, LANG_INSTRUCTIONS, build_user_prompt
from core.citation import (
    normalize_citation_quotes,
    postprocess_citations,
    enhance_citations_with_positions,
    enhance_citations_html,
)
from core.discussion import generate_discussion_stream, MAX_PAPER_TEXT_LEN
from tools.pdf_tools import (
    PdfToolContext,
    TOOL_SCHEMAS,
    execute_tool as pdf_execute_tool,
)
from tools.html_tools import (
    HtmlToolContext,
    execute_tool as html_execute_tool,
)
from tools.code_executor import execute_html_figure
from tools.figure_reviewer import review_figure
from config import (
    REPORT_MODEL,
    LLM_TEMPERATURE as TEMPERATURE,
    MAX_REPORT_TOKENS, MAX_TOOL_ROUNDS,
    MAX_TOOL_RESULT_LEN, MAX_CONTINUATIONS,
    BASE_DIR, DATA_DIR,
    LLM_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Required sections to consider a report complete
_REQUIRED_SECTIONS = ["## experiment", "## conclusion"]

# Per-chunk stall timeout: if no chunk arrives within this many seconds,
# abort the stream (protects against API hangs or malformed responses).
_STREAM_STALL_TIMEOUT = LLM_TIMEOUT


def _report_is_incomplete(report_text: str) -> bool:
    """Check if report is missing required sections."""
    lower = report_text.lower()
    # Must have TLDR to be considered a real report
    if "## tldr" not in lower:
        return False  # not a report at all, don't try to continue
    return any(s not in lower for s in _REQUIRED_SECTIONS)


# ---------------------------------------------------------------------------
# Status message helpers (embedded in stream as HTML comments)
# ---------------------------------------------------------------------------

_STATUS_PREFIX = "<!--STATUS:"
_STATUS_SUFFIX = "-->"


def _make_status(msg: str) -> str:
    return f"{_STATUS_PREFIX}{msg}{_STATUS_SUFFIX}"


def _tool_status_message(tool_name: str, arguments: dict) -> str:
    status_map = {
        "get_paper_structure": "Analyzing paper structure...",
        "read_page_detail": f"Reading page {arguments.get('page_num', '?')} in detail...",
        "search_text": f"Searching for \"{str(arguments.get('query', '?'))[:40]}\"...",
        "get_figure_context": "Examining figure context...",
        "locate_quote": "Locating citation position...",
        "generate_figure": f"Generating diagram: {str(arguments.get('description', ''))[:40]}...",
    }
    return status_map.get(tool_name, f"Running {tool_name}...")


# ---------------------------------------------------------------------------
# Main generation: streaming with tool-calling loop
# ---------------------------------------------------------------------------

async def generate_report_stream(paper: dict, figures: list[dict], lang: str = "en"):
    """Async generator yielding Markdown text chunks from the LLM.

    Supports multi-turn tool calling: when the LLM invokes a tool, the tool
    is executed locally, the result is fed back, and generation continues.
    Status messages are embedded as <!--STATUS:...--> HTML comments.
    """
    user_prompt = build_user_prompt(paper, figures)
    lang_inst = LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS["en"])
    source_type = paper.get("source_type", "pdf")

    system = SYSTEM_PROMPT + f"\n\nLanguage requirement: {lang_inst}"
    if source_type == "html":
        system += "\n\nNote: This paper was loaded from an HTML page. Page numbers refer to virtual content sections, not physical PDF pages."

    pdf_path = str(DATA_DIR / "uploads" / f"{paper['id']}.pdf")

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    full_report: list[str] = []

    # Create the appropriate tool context
    if source_type == "html":
        tool_ctx = HtmlToolContext(paper.get("full_text", ""))
        exec_tool = html_execute_tool
    else:
        tool_ctx = PdfToolContext(pdf_path)
        exec_tool = pdf_execute_tool

    try:
        for _round in range(MAX_TOOL_ROUNDS):
            text_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            finish_reason = None

            try:
                stream = generate_stream(
                    messages,
                    model=REPORT_MODEL,
                    tools=TOOL_SCHEMAS,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_REPORT_TOKENS,
                )
                ait = stream.__aiter__()
                while True:
                    try:
                        sc = await asyncio.wait_for(ait.__anext__(), timeout=_STREAM_STALL_TIMEOUT)
                    except StopAsyncIteration:
                        break

                    if sc.finish_reason:
                        finish_reason = sc.finish_reason

                    if sc.reasoning:
                        reasoning_chunks.append(sc.reasoning)

                    if sc.content:
                        text_chunks.append(sc.content)
                        full_report.append(sc.content)
                        yield sc.content

                    if sc.tool_calls:
                        for tcd in sc.tool_calls:
                            idx = tcd.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tcd.id:
                                tool_calls_acc[idx]["id"] = tcd.id
                            if tcd.name:
                                tool_calls_acc[idx]["name"] = tcd.name
                            if tcd.arguments:
                                tool_calls_acc[idx]["arguments"] += tcd.arguments
            except asyncio.TimeoutError:
                logger.error("Round %d: stream stalled for %ds, aborting round", _round, _STREAM_STALL_TIMEOUT)
            except Exception as e:
                logger.error("Round %d: streaming error: %s", _round, e)

            # If model finished without requesting tools, we're done
            if finish_reason != "tool_calls" or not tool_calls_acc:
                report_so_far = "".join(text_chunks)
                logger.info(
                    "Report generation round %d ended: finish_reason=%s, "
                    "text_len=%d chars, tool_calls=%d, last_50_chars=%r",
                    _round, finish_reason, len(report_so_far),
                    len(tool_calls_acc), report_so_far[-50:] if report_so_far else "",
                )
                # Handle truncation: either explicit length limit or incomplete content
                needs_continue = (
                    (finish_reason == "length" and text_chunks)
                    or (text_chunks and _report_is_incomplete("".join(full_report)))
                )
                if needs_continue:
                    logger.info("Report incomplete (finish_reason=%s), auto-continuing...", finish_reason)
                    report_so_far = "".join(full_report)
                    # Use lightweight context for continuation: no full paper text,
                    # so the model has room to generate the remaining sections.
                    paper_excerpt = paper.get("full_text", "")
                    if len(paper_excerpt) > MAX_PAPER_TEXT_LEN:
                        paper_excerpt = paper_excerpt[:MAX_PAPER_TEXT_LEN] + "\n\n... (truncated)"
                    cont_messages: list[dict] = [
                        {"role": "system", "content": (
                            "You are writing an academic paper reading report. "
                            "Maintain the same style, language, and [[p.N \"quote\"]] citation format.\n\n"
                            f"Language requirement: {lang_inst}"
                        )},
                        {"role": "user", "content": (
                            f"Paper text (for reference):\n{paper_excerpt}\n\n"
                            "Write a detailed reading report for this paper."
                        )},
                        {"role": "assistant", "content": report_so_far},
                        {"role": "user", "content": (
                            "Your report was cut off. Continue writing from EXACTLY where you stopped. "
                            "Do NOT repeat any content already written. "
                            "Complete all remaining sections including ## Experiments and ## Conclusion."
                        )},
                    ]
                    yield _make_status("Continuing report generation...")
                    for _cont in range(MAX_CONTINUATIONS):
                        try:
                            cont_chunks: list[str] = []
                            cont_finish = None
                            async for sc in generate_stream(
                                cont_messages,
                                model=REPORT_MODEL,
                                temperature=TEMPERATURE,
                                max_tokens=16384,
                            ):
                                if sc.finish_reason:
                                    cont_finish = sc.finish_reason
                                if sc.content:
                                    cont_chunks.append(sc.content)
                                    full_report.append(sc.content)
                                    yield sc.content
                        except Exception as e:
                            logger.error("Continuation %d API call failed: %s", _cont + 1, e)
                            break
                        logger.info(
                            "Continuation %d ended: finish_reason=%s, chunk_len=%d",
                            _cont + 1, cont_finish, len("".join(cont_chunks)),
                        )
                        if not cont_chunks:
                            logger.warning("Continuation %d produced no content, giving up", _cont + 1)
                            break
                        if not _report_is_incomplete("".join(full_report)):
                            break
                        # Rebuild with accumulated report in assistant role
                        accumulated = "".join(full_report)
                        cont_messages = [
                            cont_messages[0],  # system prompt
                            cont_messages[1],  # user: paper text
                            {"role": "assistant", "content": accumulated},
                            {"role": "user", "content": (
                                "Your report was cut off. Continue from EXACTLY where you stopped. "
                                "Do NOT repeat any content already written."
                            )},
                        ]
                break

            # Tool-calling round: remove any intermediate text from report
            if text_chunks:
                del full_report[-len(text_chunks):]

            # Build assistant message with tool_calls for message history
            assistant_tool_calls = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                tc_id = tc["id"] or f"call_{_round}_{idx}"
                assistant_tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                })

            assistant_msg: dict = {"role": "assistant", "tool_calls": assistant_tool_calls}
            if text_chunks:
                assistant_msg["content"] = "".join(text_chunks)
            else:
                assistant_msg["content"] = ""
            if reasoning_chunks:
                assistant_msg["reasoning_content"] = "".join(reasoning_chunks)
            messages.append(assistant_msg)

            # Execute each tool call and feed results back
            for tc_msg in assistant_tool_calls:
                tool_name = tc_msg["function"]["name"]
                try:
                    arguments = json.loads(tc_msg["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                status_text = _tool_status_message(tool_name, arguments)
                yield _make_status(status_text)

                if tool_name == "generate_figure":
                    result = execute_html_figure(
                        code=arguments.get("code", ""),
                        paper_id=paper["id"],
                        fig_name=arguments.get("description", "figure"),
                    )
                    if result.get("success") and result.get("path"):
                        abs_path = str(
                            BASE_DIR / result["path"].lstrip("/")
                        )
                        yield _make_status("Reviewing figure quality...")
                        review = review_figure(
                            abs_path,
                            description=arguments.get("description", ""),
                        )
                        result["review"] = review["feedback"]
                        if not review["passed"]:
                            result["success"] = False
                            result["error"] = (
                                f"Figure review FAILED: {review['feedback']}. "
                                "Please fix the issues and call generate_figure again."
                            )
                else:
                    result = exec_tool(tool_ctx, tool_name, arguments)
                result_str = json.dumps(result, ensure_ascii=False)

                if len(result_str) > MAX_TOOL_RESULT_LEN:
                    result_str = result_str[:MAX_TOOL_RESULT_LEN] + "... (truncated)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_msg["id"],
                    "content": result_str,
                })
    finally:
        if hasattr(tool_ctx, "close"):
            tool_ctx.close()

    # Fallback: if all tool-call rounds were exhausted without producing report text,
    # make one final call WITHOUT tools to force the model to write the report.
    if not "".join(full_report).strip():
        yield _make_status("Writing report...")
        messages.append({
            "role": "user",
            "content": (
                "You have thoroughly analyzed the paper. "
                "Now write the complete report. "
                "Start directly with ## TLDR — no preamble."
            ),
        })
        try:
            async for sc in generate_stream(
                messages,
                model=REPORT_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_REPORT_TOKENS,
            ):
                if sc.content:
                    full_report.append(sc.content)
                    yield sc.content
        except Exception:
            pass

    # Post-processing
    report_text = normalize_citation_quotes("".join(full_report))
    if report_text.strip():
        yield _make_status("Enhancing citations...")

        if source_type == "html":
            enhanced = await asyncio.to_thread(
                enhance_citations_html, report_text, paper.get("full_text", "")
            )
        else:
            enhanced = await asyncio.to_thread(
                enhance_citations_with_positions, report_text, pdf_path
            )

        if not re.search(r"\[\[p\.\s*\d+", enhanced):
            enhanced = postprocess_citations(enhanced, paper.get("full_text", ""))

        if enhanced != report_text:
            yield f"\n\n<!--FULL_REPLACE-->{enhanced}<!--/FULL_REPLACE-->"
            report_text = enhanced

        await update_report(paper["id"], report_text)

    # === Discussion phase (automatic, same stream) ===
    if report_text.strip():
        yield _make_status("Starting review discussion...")
        try:
            async for event in generate_discussion_stream(paper, figures, report_text, lang):
                yield event
        except Exception as e:
            yield {"type": "discussion_error", "error": str(e)}
