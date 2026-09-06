"""
Database module - SQLite operations for complaint history
College NLP PBL Project - Academic Year 2026-27
"""

import os
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = os.path.join("database", "database.db")


def get_connection():
    """Opens a thread-safe SQLite connection with row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def db_connection():
    """Context manager for auto-committing/rollback SQLite connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Creates tables if they don't exist."""
    with db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_text  TEXT    NOT NULL,
                clean_text      TEXT,
                predicted_category TEXT NOT NULL,
                department      TEXT,
                model_score     REAL,
                status          TEXT    DEFAULT 'Pending',
                created_at      TEXT    DEFAULT (datetime('now', 'localtime'))
            )
        """)
    print("Database initialized at:", DB_PATH)


def insert_complaint(
    complaint_text: str,
    clean_text: str,
    predicted_category: str,
    department: str,
    model_score: float
) -> int:
    """Inserts a new classified complaint record and returns its ID."""
    with db_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO complaints
                (complaint_text, clean_text, predicted_category, department, model_score, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'Pending', datetime('now', 'localtime'))
            """,
            (complaint_text, clean_text, predicted_category, department, model_score)
        )
        return cur.lastrowid


def get_all_complaints(
    search: str = None,
    category: str = None,
    status: str = None,
    limit: int = 200
) -> list:
    """Fetches complaint history with optional filters."""
    query  = "SELECT * FROM complaints WHERE 1=1"
    params = []

    if search:
        query += " AND complaint_text LIKE ?"
        params.append(f"%{search}%")
    if category and category != "All":
        query += " AND predicted_category = ?"
        params.append(category)
    if status and status != "All":
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with db_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def update_status(complaint_id: int, new_status: str) -> bool:
    """Updates the status (Pending / Resolved) of a complaint by ID."""
    if new_status not in ("Pending", "Resolved"):
        return False
    with db_connection() as conn:
        cur = conn.execute(
            "UPDATE complaints SET status = ? WHERE id = ?",
            (new_status, complaint_id)
        )
    return cur.rowcount > 0


def get_dashboard_stats() -> dict:
    """Returns aggregated statistics used by the Dashboard page."""
    with db_connection() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Resolved'").fetchone()[0]
        pending  = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Pending'").fetchone()[0]

        # Category breakdown
        rows = conn.execute(
            "SELECT predicted_category, COUNT(*) as cnt FROM complaints GROUP BY predicted_category"
        ).fetchall()
        by_category = {r["predicted_category"]: r["cnt"] for r in rows}

        # Average confidence
        avg_score = conn.execute(
            "SELECT AVG(model_score) FROM complaints WHERE model_score IS NOT NULL"
        ).fetchone()[0]

    return {
        "total":       total,
        "resolved":    resolved,
        "pending":     pending,
        "by_category": by_category,
        "avg_score":   round(float(avg_score), 2) if avg_score else 0.0
    }


def get_complaint_by_id(complaint_id: int) -> dict:
    """Fetches a single complaint record by its primary key."""
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM complaints WHERE id = ?", (complaint_id,)
        ).fetchone()
    return dict(row) if row else None
