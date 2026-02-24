import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from database import update_report
from pdf_tools import (
    PdfToolContext,
    TOOL_SCHEMAS,
    execute_tool,
    locate_quote,
)

load_dotenv()

client = AsyncOpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("LLM_API_KEY", ""),
)

MODEL = "deepseek-chat"
TEMPERATURE = 0.7
MAX_TOOL_ROUNDS = 15  # safety limit to prevent infinite tool-call loops
MAX_TOOL_RESULT_LEN = 8000  # truncate large tool results

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
    }
    return status_map.get(tool_name, f"Running {tool_name}...")


# ---------------------------------------------------------------------------
# System Prompt (with tool-calling instructions)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert academic paper reader. You have access to PDF analysis tools that help you examine the paper precisely.

AVAILABLE TOOLS (use selectively when helpful):
- get_paper_structure(): Understand the paper's section organization
- read_page_detail(page_num): Examine a specific page in detail
- search_text(query): Find specific text with page and position info
- get_figure_context(figure_caption): Understand context around a figure
- locate_quote(quote, page_hint): Find exact position of a verbatim quote

You may call tools before writing to better understand the paper. Use them selectively — you do not need to call them for every citation. The system will automatically enhance citation positions after generation.

CRITICAL: Do NOT output any thinking, planning, or explanatory text. Start your report DIRECTLY with "## TLDR". No preamble.

CITATION FORMAT:
After every key claim, cite the source with [[p.N "verbatim quote"]] where:
- N = page number (from "--- Page N ---" markers in the text)
- verbatim quote = 3-10 word phrase copied EXACTLY from the paper
- If you have a y-position from locate_quote tool results, use [[p.N:Y "quote"]] (Y = 0-1000)
Every paragraph MUST contain at least one citation. This is mandatory.

Example: "The model uses 256 experts. [[p.3 "mixture-of-experts with 256 routed experts"]]"

REPORT STRUCTURE (use these exact headings):

## TLDR
2-3 sentence summary with citation.

## Motivation
What problem? Why important? (1-2 paragraphs with citations)

## Method
Detailed approach with sub-headings. Insert relevant figures using exact ![...](...) syntax.

## Experiments
Setup and key results. Insert relevant figures/tables.

## Conclusion
Key takeaways and limitations. (1 paragraph with citations)

FIGURE RULES:
- Include images in Method and Experiments sections using the EXACT ![...](...) syntax from the figure list.
- Place each image on its own line after the relevant paragraph.
- Only include figures that genuinely support the text.
- Do NOT modify image paths or invent figures.

WRITING RULES:
- Write in clear, concise academic English (or the specified language).
- Use bullet points and sub-headings where helpful.
- REMEMBER: Every paragraph MUST contain at least one [[p.N "verbatim quote"]] citation."""

LANG_INSTRUCTIONS = {
    "en": "Write the entire report in English.",
    "zh": "用中文撰写整篇报告。所有标题、正文、总结都必须使用中文，但保留专有名词和术语的英文原文。",
}


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(paper: dict, figures: list[dict]) -> str:
    parts = [f"# Paper: {paper['title']}"]

    if paper.get("authors"):
        parts.append(f"**Authors:** {paper['authors']}")

    if figures:
        parts.append("\n## Available Figures (cropped from the PDF)")
        parts.append(
            "Below are the figures and tables extracted from the paper. "
            "Choose the most relevant ones and insert them into your report "
            "using the exact Markdown syntax shown."
        )
        for fig in figures:
            caption = fig.get("caption", "")
            parts.append(
                f"- {caption} (page {fig['page_num'] + 1})\n"
                f"  Syntax: ![{caption}](/data/figures/{paper['id']}/{fig['filename']})"
            )
    else:
        parts.append("\n(No figures were extracted from this paper.)")

    parts.append(f"\n## Full Paper Text\n{paper['full_text']}")
    parts.append(
        "\n\nREMINDER: You MUST include [[p.N \"verbatim quote\"]] citations in every paragraph. "
        "Look at the --- Page N --- markers above to determine which page each fact comes from. "
        "Copy a distinctive 3-10 word phrase exactly from the paper as the verbatim quote. "
        "Do NOT output any thinking text — start directly with ## TLDR."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Fallback: inject citations if LLM didn't produce them
# ---------------------------------------------------------------------------

def _postprocess_citations(report: str, full_text: str) -> str:
    """If the LLM failed to produce [[p.N]] citations at all, inject them."""
    if re.search(r"\[\[p\.\d+", report):
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

def _enhance_citations_with_positions(report: str, pdf_path: str) -> str:
    """Add y-positions to citations that lack them."""
    # Match [[p.N "quote"]] without :Y
    pattern = re.compile(r'\[\[p\.(\d+)\s+"([^"]+)"\]\]')

    if not pattern.search(report):
        return report  # nothing to enhance

    try:
        with PdfToolContext(pdf_path) as ctx:

            def replacer(match):
                page = int(match.group(1))
                quote = match.group(2)
                result = locate_quote(ctx, quote, page_hint=page)
                if result.get("found") and "y" in result:
                    return f'[[p.{result["page"]}:{result["y"]} "{quote}"]]'
                return match.group(0)

            return pattern.sub(replacer, report)
    except Exception:
        return report  # if PDF can't be opened, leave as-is


# ---------------------------------------------------------------------------
# Main generation: streaming with tool-calling loop
# ---------------------------------------------------------------------------

async def generate_report_stream(paper: dict, figures: list[dict], lang: str = "en"):
    """Async generator yielding Markdown text chunks from the LLM.

    Supports multi-turn tool calling: when the LLM invokes a tool, the tool
    is executed locally, the result is fed back, and generation continues.
    Status messages are embedded as <!--STATUS:...--> HTML comments.
    """
    user_prompt = _build_user_prompt(paper, figures)
    lang_inst = LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS["en"])
    system = SYSTEM_PROMPT + f"\n\nLanguage requirement: {lang_inst}"

    pdf_path = str(
        Path(__file__).parent / "data" / "uploads" / f"{paper['id']}.pdf"
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    full_report: list[str] = []

    with PdfToolContext(pdf_path) as ctx:
        for _round in range(MAX_TOOL_ROUNDS):
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=TEMPERATURE,
                stream=True,
            )

            text_chunks: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            finish_reason = None

            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                # Stream text content immediately
                if delta.content:
                    text_chunks.append(delta.content)
                    full_report.append(delta.content)
                    yield delta.content

                # Accumulate tool calls from streamed deltas
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments

            # If model finished without requesting tools, we're done
            if finish_reason != "tool_calls" or not tool_calls_acc:
                break

            # Build assistant message with tool_calls for message history
            assistant_tool_calls = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                assistant_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                })

            assistant_msg: dict = {"role": "assistant", "tool_calls": assistant_tool_calls}
            if text_chunks:
                assistant_msg["content"] = "".join(text_chunks)
            messages.append(assistant_msg)

            # Execute each tool call and feed results back
            for tc_msg in assistant_tool_calls:
                tool_name = tc_msg["function"]["name"]
                try:
                    arguments = json.loads(tc_msg["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                # Send status to frontend
                status_text = _tool_status_message(tool_name, arguments)
                yield _make_status(status_text)

                # Execute tool
                result = execute_tool(ctx, tool_name, arguments)
                result_str = json.dumps(result, ensure_ascii=False)

                # Truncate large results
                if len(result_str) > MAX_TOOL_RESULT_LEN:
                    result_str = result_str[:MAX_TOOL_RESULT_LEN] + "... (truncated)"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_msg["id"],
                    "content": result_str,
                })

    # Post-processing
    report_text = "".join(full_report)
    if report_text.strip():
        # 1. Enhance citations with y-positions (for those without :Y)
        enhanced = _enhance_citations_with_positions(report_text, pdf_path)

        # 2. Fallback: inject citations if LLM didn't produce any
        if not re.search(r"\[\[p\.\d+", enhanced):
            enhanced = _postprocess_citations(enhanced, paper.get("full_text", ""))

        if enhanced != report_text:
            yield f"\n\n<!--FULL_REPLACE-->{enhanced}<!--/FULL_REPLACE-->"
            report_text = enhanced

        await update_report(paper["id"], report_text)
