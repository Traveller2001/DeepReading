# DeepReading

AI-powered academic paper reading assistant. Upload a PDF paper, get a structured reading report with precise, clickable citations that jump to the exact location in the original PDF.

## Features

- **PDF Upload & Analysis** — Upload a PDF, automatically extract text, metadata, figures, and tables
- **LLM-Powered Report Generation** — Generates structured reports (TLDR, Motivation, Method, Experiments, Conclusion) via streaming SSE
- **LLM Tool Calling** — The LLM can call PDF analysis tools during generation to better understand the paper structure, search for specific content, and verify facts
- **Precise Citations** — Every claim in the report is backed by a `[[p.N:Y "verbatim quote"]]` citation with page number and normalized y-position
- **Click-to-Jump** — Click any citation badge to scroll the PDF viewer to the exact position and highlight the quoted text
- **Figure Extraction** — Automatically detects and crops figures/tables from the PDF using caption-based region detection
- **Multi-Language** — Generate reports in English or Chinese
- **Split View** — Read the report on the left while referencing the PDF on the right

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Frontend    │◄───►│  FastAPI      │◄───►│  LLM API         │
│  (SPA)       │ SSE │  (app.py)     │     │  (tool calling)  │
└─────────────┘     └──────┬───────┘     └────────┬─────────┘
                           │                       │
                    ┌──────┴───────┐        ┌──────┴─────────┐
                    │  SQLite DB   │        │  pdf_tools.py  │
                    │  (aiosqlite) │        │  (PyMuPDF)     │
                    └──────────────┘        └────────────────┘
```

### Key Files

| File | Description |
|------|-------------|
| `app.py` | FastAPI server, API endpoints, SSE streaming |
| `llm_service.py` | LLM integration, tool-calling loop, report generation |
| `pdf_tools.py` | PDF analysis tools for LLM function calling |
| `pdf_processor.py` | PDF text/figure extraction on upload |
| `database.py` | Async SQLite storage |
| `static/index.html` | Single-page frontend application |

## PDF Tools (LLM Function Calling)

During report generation, the LLM can invoke these tools to analyze the paper:

| Tool | Description |
|------|-------------|
| `get_paper_structure()` | Extract section headings with page numbers and y-positions |
| `read_page_detail(page_num)` | Get text blocks of a specific page with spatial layout |
| `search_text(query)` | Search for text across all pages, returns matches with positions |
| `get_figure_context(figure_caption)` | Get text surrounding a figure to understand its context |
| `locate_quote(quote, page_hint)` | Find the exact position of a verbatim quote |

Tools return y-positions on a normalized 0–1000 scale (0 = top, 1000 = bottom). The frontend uses these coordinates to scroll to the precise location when a citation is clicked.

## Citation System

Reports use the format `[[p.N:Y "verbatim quote"]]`:
- `N` — 1-based page number
- `Y` — normalized y-position (0–1000), added automatically by post-processing
- `"verbatim quote"` — exact phrase from the paper

In the frontend, citations render as clickable `[p.N]` badges. Clicking one:
1. Opens the PDF panel (if closed)
2. Scrolls to the exact y-position on page N
3. Highlights the quoted text with a yellow overlay (fades after 5s)
4. Falls back to y-position indicator or page-level highlight if text matching fails

## Quick Start

### Prerequisites

- Python 3.11+
- A DeepSeek API key (or any OpenAI-compatible API)

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd DeepReading

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "LLM_API_KEY=your-api-key-here" > .env
```

### Run

```bash
python app.py
```

The server starts at `http://localhost:8000`.

### Usage

1. Open `http://localhost:8000` in your browser
2. Upload a PDF paper (drag-drop or click the upload zone)
3. Click **Generate Report** (choose English or Chinese)
4. Watch the report stream in real-time with the thinking panel showing tool call progress
5. Click any `[p.N]` citation to jump to the source in the PDF viewer

## Configuration

Environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for the LLM provider | (required) |

To use a different LLM provider, edit `llm_service.py`:

```python
client = AsyncOpenAI(
    base_url="https://api.deepseek.com",  # Change to your provider
    api_key=os.environ.get("LLM_API_KEY", ""),
)
MODEL = "deepseek-chat"  # Change to your model
```

## Data Storage

All data is stored locally:

```
data/
├── uploads/      # Original PDF files ({paper_id}.pdf)
├── figures/      # Extracted figure images ({paper_id}/fig_N.png)
└── papers.db     # SQLite database (metadata, full text, reports)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/papers/upload` | Upload a PDF |
| `GET` | `/api/papers` | List papers (optional `?q=search`) |
| `GET` | `/api/papers/{id}` | Get paper detail + figures |
| `DELETE` | `/api/papers/{id}` | Delete paper and all associated data |
| `GET` | `/api/papers/{id}/generate` | Generate report (SSE stream, `?lang=en\|zh`) |
| `GET` | `/api/papers/{id}/report` | Download report as `.md` |

## Tech Stack

- **Backend**: FastAPI, uvicorn, aiosqlite
- **PDF Processing**: PyMuPDF (fitz), Pillow
- **LLM**: OpenAI-compatible API (with function calling)
- **Frontend**: Vanilla HTML/CSS/JS, marked.js, pdf.js
