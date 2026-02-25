import aiosqlite
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "papers.db"


def _db():
    """Return an aiosqlite connection context manager."""
    return aiosqlite.connect(str(DB_PATH))


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with _db() as conn:
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT '',
                authors     TEXT NOT NULL DEFAULT '',
                abstract    TEXT NOT NULL DEFAULT '',
                full_text   TEXT NOT NULL DEFAULT '',
                num_pages   INTEGER NOT NULL DEFAULT 0,
                num_figures INTEGER NOT NULL DEFAULT 0,
                filename    TEXT NOT NULL DEFAULT '',
                report      TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS figures (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id    TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                fig_index   INTEGER NOT NULL,
                filename    TEXT NOT NULL,
                page_num    INTEGER NOT NULL,
                width       INTEGER NOT NULL,
                height      INTEGER NOT NULL,
                caption     TEXT NOT NULL DEFAULT ''
            );
        """)
        # Migration: add discussion columns if missing
        cursor = await conn.execute("PRAGMA table_info(papers)")
        existing_cols = {row[1] for row in await cursor.fetchall()}
        if "discussion" not in existing_cols:
            await conn.execute("ALTER TABLE papers ADD COLUMN discussion TEXT")
        if "discussion_status" not in existing_cols:
            await conn.execute("ALTER TABLE papers ADD COLUMN discussion_status TEXT")
        await conn.commit()


async def insert_paper(paper: dict):
    async with _db() as conn:
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute(
            """INSERT INTO papers (id, title, authors, abstract, full_text,
               num_pages, num_figures, filename)
               VALUES (:id, :title, :authors, :abstract, :full_text,
               :num_pages, :num_figures, :filename)""",
            paper,
        )
        await conn.commit()


async def insert_figures(paper_id: str, figures: list[dict]):
    async with _db() as conn:
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.executemany(
            """INSERT INTO figures (paper_id, fig_index, filename, page_num, width, height, caption)
               VALUES (:paper_id, :fig_index, :filename, :page_num, :width, :height, :caption)""",
            [{"paper_id": paper_id, **f} for f in figures],
        )
        await conn.commit()


async def get_paper(paper_id: str) -> dict | None:
    async with _db() as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_figures(paper_id: str) -> list[dict]:
    async with _db() as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            "SELECT * FROM figures WHERE paper_id = ? ORDER BY fig_index",
            (paper_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def list_papers(query: str = "") -> list[dict]:
    async with _db() as conn:
        conn.row_factory = aiosqlite.Row
        if query:
            cursor = await conn.execute(
                """SELECT id, title, authors, filename, num_pages, num_figures,
                   report IS NOT NULL as has_report, discussion_status, created_at
                   FROM papers
                   WHERE title LIKE ? OR authors LIKE ? OR filename LIKE ?
                   ORDER BY created_at DESC""",
                (f"%{query}%", f"%{query}%", f"%{query}%"),
            )
        else:
            cursor = await conn.execute(
                """SELECT id, title, authors, filename, num_pages, num_figures,
                   report IS NOT NULL as has_report, discussion_status, created_at
                   FROM papers ORDER BY created_at DESC"""
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def delete_paper(paper_id: str) -> bool:
    async with _db() as conn:
        await conn.execute("PRAGMA foreign_keys = ON")
        cursor = await conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        await conn.commit()
        return cursor.rowcount > 0


async def update_report(paper_id: str, report: str):
    async with _db() as conn:
        await conn.execute(
            "UPDATE papers SET report = ?, updated_at = datetime('now') WHERE id = ?",
            (report, paper_id),
        )
        await conn.commit()


async def update_discussion(paper_id: str, discussion_json: str, status: str):
    async with _db() as conn:
        await conn.execute(
            "UPDATE papers SET discussion = ?, discussion_status = ?, updated_at = datetime('now') WHERE id = ?",
            (discussion_json, status, paper_id),
        )
        await conn.commit()
