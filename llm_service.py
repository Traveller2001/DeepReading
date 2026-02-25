import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from database import update_report, update_discussion
from pdf_tools import (
    PdfToolContext,
    TOOL_SCHEMAS,
    execute_tool,
    locate_quote,
)
from code_executor import execute_html_figure
from figure_reviewer import review_figure

load_dotenv()

client = AsyncOpenAI(
    base_url=os.environ.get("LLM_API_BASE", "https://api.deepseek.com"),
    api_key=os.environ.get("LLM_API_KEY", ""),
)

MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
REPORT_MODEL = os.environ.get("REPORT_MODEL", MODEL)
TEMPERATURE = 0.7
MAX_TOOL_ROUNDS = 25  # safety limit to prevent infinite tool-call loops
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
        "generate_figure": f"Generating diagram: {str(arguments.get('description', ''))[:40]}...",
    }
    return status_map.get(tool_name, f"Running {tool_name}...")


# ---------------------------------------------------------------------------
# System Prompt (with tool-calling instructions)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert academic paper reader. You have access to PDF analysis tools that let you examine the paper with spatial precision.

YOU MUST FOLLOW THIS MULTI-PHASE WORKFLOW — do not skip any phase:

## Phase 1 — Understand Structure (MANDATORY)
Call get_paper_structure() FIRST to learn the paper's section layout and page numbers.

## Phase 2 — Deep Read (MANDATORY)
Before writing anything, deeply investigate the paper:
- Call read_page_detail() on key pages: the abstract page, core method pages, experiment/results pages. Read at least 3-5 pages.
- Call search_text() to locate specific claims, equations, key terminology, or numerical results you want to cite precisely.
- Call get_figure_context() to understand the context and significance of important figures before inserting them.
- You MUST call at least 5 tools total before writing any report content.

## Phase 3 — Write the Report
Only after thorough investigation, write the full report with precise citations.
Actively consider which concepts would benefit from a mermaid diagram to aid understanding — for example:
- The overall model architecture or system pipeline
- Multi-stage training or inference workflows
- The relationship between key components (e.g., attention, routing, experts)
- Data flow or processing steps
Include at least one mermaid diagram in the Method section to visually explain the core approach.

AVAILABLE TOOLS:
- get_paper_structure(): Get section headings with page numbers and y-positions
- read_page_detail(page_num): Examine a specific page's text blocks in detail
- search_text(query): Find specific text across all pages with positions and context
- get_figure_context(figure_caption): Understand text surrounding a figure/table
- locate_quote(quote, page_hint): Find exact position of a verbatim quote
- generate_figure(code, description): Create a diagram using HTML/CSS/SVG code. The code is rendered in a browser and saved as PNG. Write BODY content only (no <html>/<body> tags). After calling, insert the returned path with ![description](path).

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

DIAGRAM RULES (generate_figure tool):
- You SHOULD call generate_figure() to create explanatory diagrams that aid understanding.
- Include at least 1 diagram in the report (preferably in the Method section).
- Write HTML/CSS/SVG code for the body content. The system auto-wraps it in an HTML page with fonts and padding.
- Design guidelines:
  - Use flexbox/grid for layout. Use rounded boxes with colored backgrounds and borders.
  - Use arrows (→, ←, ↓, ↑) or SVG lines/arrows to show data flow.
  - Use a consistent color palette: blues (#e0f2fe/#0369a1), yellows (#fef3c7/#92400e), greens (#d1fae5/#065f46), purples (#e0e7ff/#4338ca).
  - Keep text concise inside boxes. Font size 14-16px for labels, 20-24px for titles.
  - Use <style> tag for CSS, inline styles for fine-tuning.
- Example of a good diagram:
  <style>
    .title{text-align:center;font-size:20px;font-weight:700;margin-bottom:20px;color:#1e293b}
    .flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:center}
    .box{border-radius:10px;padding:12px 20px;font-weight:600;font-size:14px;border:2px solid;text-align:center}
    .arrow{font-size:22px;color:#64748b}
  </style>
  <div class="title">Model Pipeline</div>
  <div class="flow">
    <div class="box" style="background:#e0f2fe;border-color:#7dd3fc;color:#0369a1">Input</div>
    <div class="arrow">→</div>
    <div class="box" style="background:#fef3c7;border-color:#fcd34d;color:#92400e">Encoder</div>
    <div class="arrow">→</div>
    <div class="box" style="background:#e0e7ff;border-color:#a5b4fc;color:#4338ca">MoE Layer</div>
    <div class="arrow">→</div>
    <div class="box" style="background:#d1fae5;border-color:#6ee7b7;color:#065f46">Output</div>
  </div>
- For complex architectures, use nested divs, multiple rows, and SVG arrows.
- Use Chinese labels when the report language is Chinese.

WRITING RULES:
- Write in clear, concise academic English (or the specified language).
- PERSPECTIVE: You are a reader summarizing someone else's paper. Use THIRD-PERSON perspective throughout: "the authors propose...", "this paper presents...", "the model achieves...". NEVER use first-person ("we", "our") as if you are the paper's author.
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
                model=REPORT_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=TEMPERATURE,
                stream=True,
            )

            text_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            finish_reason = None

            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                # Capture reasoning_content (deepseek-reasoner)
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_chunks.append(rc)

                # Stream text content to client in real time
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

            # Tool-calling round: remove any intermediate text from report
            if text_chunks:
                del full_report[-len(text_chunks):]

            # Build assistant message with tool_calls for message history
            assistant_tool_calls = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                # Some APIs return empty tool_call_id; generate one if missing
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
            # Only include reasoning_content if present (DeepSeek-specific)
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

                # Send status to frontend
                status_text = _tool_status_message(tool_name, arguments)
                yield _make_status(status_text)

                # Execute tool
                if tool_name == "generate_figure":
                    result = execute_html_figure(
                        code=arguments.get("code", ""),
                        paper_id=paper["id"],
                        fig_name=arguments.get("description", "figure"),
                    )
                    # Vision review: check generated figure quality
                    if result.get("success") and result.get("path"):
                        abs_path = str(
                            Path(__file__).parent / result["path"].lstrip("/")
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
        yield _make_status("Enhancing citations...")

        # Run blocking PDF operations in a thread to avoid blocking the event loop
        enhanced = await asyncio.to_thread(
            _enhance_citations_with_positions, report_text, pdf_path
        )

        # 2. Fallback: inject citations if LLM didn't produce any
        if not re.search(r"\[\[p\.\d+", enhanced):
            enhanced = _postprocess_citations(enhanced, paper.get("full_text", ""))

        if enhanced != report_text:
            yield f"\n\n<!--FULL_REPLACE-->{enhanced}<!--/FULL_REPLACE-->"
            report_text = enhanced

        await update_report(paper["id"], report_text)

    # === Discussion phase (automatic, same stream) ===
    if report_text.strip():
        # Keep-alive: yield status so SSE connection stays open during the transition
        yield _make_status("Starting review discussion...")
        try:
            async for event in generate_discussion_stream(paper, figures, report_text, lang):
                yield event
        except Exception as e:
            yield {"type": "discussion_error", "error": str(e)}


# ---------------------------------------------------------------------------
# Discussion / Review Feature
# ---------------------------------------------------------------------------

DISCUSSION_ROUNDS = 3
MAX_PAPER_TEXT_LEN = 30000  # truncate paper text for writer/polish context

READER_SYSTEM_PROMPT = """You are a critical academic reader — an outsider researcher encountering this paper's report for the first time. Your job is to ask ONE pointed, specific question that will improve the report's clarity and completeness.

Focus on ONE of the following (pick the most important gap in this round):
- A concept or term that is used without adequate explanation
- A gap in the methodology or experimental setup description
- A missing comparison, baseline, or context that would help understanding
- A claim that lacks supporting evidence or seems vague
- Jargon or abbreviations that aren't defined

Ask exactly ONE question. Be specific — reference exact sections or sentences from the report. Do NOT ask generic questions. Do NOT be polite or add filler text — get straight to the question."""

WRITER_SYSTEM_PROMPT = """You are the author of this academic paper report. A reader has asked a question about your report. Answer the question thoroughly and precisely, drawing from the original paper text provided.

Rules:
- Answer the question directly with evidence from the paper
- Include [[p.N "verbatim quote"]] citations to reference the original paper, where N is the page number and "verbatim quote" is a 3-10 word phrase copied exactly from the paper
- If the answer involves formulas or equations, write them using LaTeX notation wrapped in $...$ (inline) or $$...$$ (block)
- If the paper doesn't contain enough information to answer, say so honestly
- Keep the answer concise but complete — typically 2-4 sentences
- Do NOT add pleasantries or filler — answer directly"""

POLISH_SYSTEM_PROMPT = """You are revising an academic paper reading report based on a discussion between a reader and the report's author. The discussion revealed areas where the report could be clearer, more complete, or better structured.

Your task:
- Revise the report to incorporate the insights from the discussion
- Address the questions that were raised by adding missing context, clarifying jargon, filling gaps
- Keep the EXACT same report structure (## TLDR, ## Motivation, ## Method, ## Experiments, ## Conclusion)
- Preserve ALL existing [[p.N "quote"]] citations — do NOT remove or modify them
- Preserve ALL existing image references ![...](...) — do NOT remove or modify them
- Preserve ALL existing mermaid diagrams — do NOT remove or modify them
- You may add new sentences or paragraphs, but do NOT delete existing content unless replacing it with something better
- PERSPECTIVE: Use THIRD-PERSON perspective throughout: "the authors propose...", "this paper presents...". NEVER use first-person ("we", "our") as if you are the paper's author. If the original report incorrectly uses first-person, fix it to third-person.
- Write in the same language and style as the original report
- Output ONLY the revised report starting with ## TLDR — no preamble or explanation"""


async def _stream_simple_completion(messages):
    """Async generator for streaming a simple (no tool-calling) LLM completion."""
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=4096,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


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
    async for chunk in _stream_simple_completion(polish_messages):
        polished_chunks.append(chunk)
        yield {"type": "polish_chunk", "content": chunk}

    polished_report = "".join(polished_chunks)

    # Post-process polished report (non-blocking)
    pdf_path = str(Path(__file__).parent / "data" / "uploads" / f"{paper['id']}.pdf")
    polished_report = await asyncio.to_thread(
        _enhance_citations_with_positions, polished_report, pdf_path
    )
    if not re.search(r"\[\[p\.\d+", polished_report):
        polished_report = _postprocess_citations(polished_report, paper.get("full_text", ""))

    await update_report(paper["id"], polished_report)

    yield {"type": "polish_end", "report": polished_report}
