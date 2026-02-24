import asyncio
import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from database import (
    init_db, insert_paper, insert_figures, get_paper, get_figures,
    list_papers, delete_paper,
)
from pdf_processor import process_pdf, figures_to_dicts
from llm_service import generate_report_stream

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FIGURES_DIR = DATA_DIR / "figures"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="DeepReading")

# Ensure directories exist before mounting StaticFiles
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup():
    await init_db()


# --- Static file serving ---
app.mount("/data/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
app.mount("/data/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# --- Paper CRUD ---

@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    paper_id = uuid4().hex
    pdf_bytes = await file.read()

    # Save original PDF
    pdf_path = UPLOADS_DIR / f"{paper_id}.pdf"
    pdf_path.write_bytes(pdf_bytes)

    # Extract text and figures (CPU-bound, run in thread)
    fig_dir = FIGURES_DIR / paper_id
    try:
        result = await asyncio.to_thread(
            process_pdf, pdf_bytes, paper_id, str(fig_dir)
        )
    except Exception as e:
        pdf_path.unlink(missing_ok=True)
        shutil.rmtree(fig_dir, ignore_errors=True)
        raise HTTPException(500, f"Failed to process PDF: {e}")

    # Store in database
    await insert_paper({
        "id": paper_id,
        "title": result.title,
        "authors": result.authors,
        "abstract": result.abstract,
        "full_text": result.full_text,
        "num_pages": result.num_pages,
        "num_figures": len(result.figures),
        "filename": file.filename,
    })
    if result.figures:
        await insert_figures(paper_id, figures_to_dicts(result.figures))

    return {
        "paper_id": paper_id,
        "title": result.title,
        "num_figures": len(result.figures),
        "num_pages": result.num_pages,
    }


@app.get("/api/papers")
async def api_list_papers(q: str = Query("", description="Search query")):
    papers = await list_papers(q)
    return papers


@app.get("/api/papers/{paper_id}")
async def api_get_paper(paper_id: str):
    paper = await get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    figures = await get_figures(paper_id)
    return {**paper, "figures": figures}


@app.delete("/api/papers/{paper_id}")
async def api_delete_paper(paper_id: str):
    ok = await delete_paper(paper_id)
    if not ok:
        raise HTTPException(404, "Paper not found")
    # Clean up files
    shutil.rmtree(FIGURES_DIR / paper_id, ignore_errors=True)
    (UPLOADS_DIR / f"{paper_id}.pdf").unlink(missing_ok=True)
    return {"ok": True}


# --- Report generation (SSE streaming) ---

@app.get("/api/papers/{paper_id}/generate")
async def api_generate_report(paper_id: str, lang: str = Query("en", description="Report language: en or zh")):
    paper = await get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    figures = await get_figures(paper_id)

    async def event_stream():
        try:
            async for chunk in generate_report_stream(paper, figures, lang=lang):
                escaped = json.dumps(chunk)
                yield f"data: {escaped}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/papers/{paper_id}/report")
async def api_download_report(paper_id: str):
    paper = await get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    if not paper.get("report"):
        raise HTTPException(404, "Report not yet generated")

    return StreamingResponse(
        iter([paper["report"]]),
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename={paper_id}_report.md"
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
