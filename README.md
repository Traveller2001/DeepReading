# DeepReading

AI-powered academic paper reading assistant. Upload a PDF, get a structured reading report with precise, clickable citations that jump to the exact location in the original PDF.

## Features

- **Structured Report** — Generates TLDR, Motivation, Method, Experiments, Conclusion
- **Precise Citations** — Click any citation to jump to the exact position in the PDF and highlight the source text
- **LLM Tool Calling** — LLM calls PDF analysis tools during generation to understand paper structure and verify facts
- **Figure Extraction** — Automatically detects and crops figures/tables from the PDF
- **Multi-Language** — English or Chinese reports
- **Split View** — Report on the left, PDF on the right

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure API key
echo "LLM_API_KEY=your-api-key-here" > .env

# Run
python app.py
```

Open `http://localhost:8000`, upload a PDF, click **Generate Report**.

## Configuration

Set `LLM_API_KEY` in `.env`. To switch LLM provider, edit `base_url` and `MODEL` in `llm_service.py`.

## Tech Stack

FastAPI / PyMuPDF / OpenAI-compatible LLM API / pdf.js
