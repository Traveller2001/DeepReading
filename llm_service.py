import os
from openai import AsyncOpenAI
from database import update_report

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("LLM_API_KEY", ""),
)

MODEL = "arcee-ai/trinity-large-preview:free"
TEMPERATURE = 0.7

SYSTEM_PROMPT = """You are an expert academic paper reader. Given the full text of a research paper and a list of extracted figures/tables (cropped from the original PDF), generate a structured reading report in Markdown format.

Your report MUST follow this exact structure with these exact headings:

## TLDR
A 2-3 sentence summary of the paper's key contribution.

## Motivation
What problem does this paper address? Why is it important? (1-2 paragraphs)

## Method
Describe the proposed method/approach in detail. Use sub-headings where appropriate.
You MUST insert the relevant figure images at appropriate positions within the text to illustrate the method. Place each image right after the paragraph that discusses it.

## Experiments
Summarize the experimental setup and key results. Highlight important findings.
You MUST insert the relevant figure/table images at appropriate positions to support the analysis. Place each image right after the paragraph that discusses it.

## Conclusion
Key takeaways, contributions, and limitations. (1 paragraph)

Important rules:
- You MUST include images in the Method and Experiments sections by copying the EXACT ![...](...) syntax provided in the figure list below.
- Place each image on its own line, right after the relevant text paragraph.
- Choose the most relevant figures for each section. You do NOT need to use every figure — only include figures that genuinely support the text.
- Do NOT modify the image paths. Use them exactly as provided.
- Do NOT invent figures that are not in the provided list.
- Write in clear, concise academic English.
- Use bullet points and sub-headings within sections where helpful."""


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
    return "\n".join(parts)


async def generate_report_stream(paper: dict, figures: list[dict]):
    """Async generator yielding Markdown text chunks from the LLM."""
    user_prompt = _build_user_prompt(paper, figures)

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        stream=True,
    )

    full_report = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_report.append(text)
            yield text

    # Save completed report to database
    report_text = "".join(full_report)
    if report_text.strip():
        await update_report(paper["id"], report_text)
