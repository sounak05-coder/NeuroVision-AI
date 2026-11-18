
import io
import re
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_DIR / "results"

DATABASE_PATH = (
    RESULTS_DIR / "segmentation_history.db"
)

SESSIONS_DIR = (
    RESULTS_DIR / "segmentation_sessions"
)


def initialize_database():
    """
    Create the SQLite database and segmentation history table.
    Safe to call repeatedly.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=30
    )

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS segmentation_sessions (
                session_id TEXT PRIMARY KEY,
                report_id TEXT,
                created_at TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                threshold REAL NOT NULL,
                tumour_pixels INTEGER NOT NULL,
                total_pixels INTEGER NOT NULL,
                tumour_percentage REAL NOT NULL,
                original_path TEXT NOT NULL,
                mask_path TEXT NOT NULL,
                overlay_path TEXT NOT NULL,
                report_path TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_segmentation_created_at
            ON segmentation_sessions(created_at)
        """)

        connection.commit()

    finally:
        connection.close()

    return DATABASE_PATH


def _safe_filename(filename):
    """Remove unsafe characters from an uploaded filename."""

    name = Path(str(filename)).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)

    return name or "uploaded_mri"


def _save_png(array, destination, mask=False):
    """Save an image array as PNG."""

    array = np.asarray(array)

    if mask:
        array = (array > 0).astype(np.uint8) * 255
        image = Image.fromarray(array)
    else:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        image = Image.fromarray(array)

    image.save(destination, format="PNG")


def save_segmentation_session(
    filename,
    original_image_bytes,
    display_mri,
    predicted_mask,
    overlay,
    threshold,
    tumour_pixels,
    tumour_percentage,
    pdf_bytes,
    report_id
):
    """
    Save one completed segmentation session.

    Stores image files and PDF in Google Drive project storage,
    then records their paths and statistics in SQLite.
    """

    initialize_database()

    session_id = str(uuid.uuid4())
    created_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=False)

    safe_name = _safe_filename(filename)

    original_path = session_dir / safe_name
    mask_path = session_dir / "predicted_mask.png"
    overlay_path = session_dir / "tumour_overlay.png"
    report_path = session_dir / "segmentation_report.pdf"

    total_pixels = int(np.asarray(predicted_mask).size)

    try:
        # Preserve the exact uploaded MRI file.
        original_path.write_bytes(original_image_bytes)

        # Save the generated segmentation images.
        _save_png(
            predicted_mask,
            mask_path,
            mask=True
        )

        _save_png(
            overlay,
            overlay_path
        )

        # Save the generated PDF report.
        report_path.write_bytes(pdf_bytes)

        # Store metadata and paths in SQLite.
        connection = sqlite3.connect(
            str(DATABASE_PATH),
            timeout=30
        )

        try:
            connection.execute("""
                INSERT INTO segmentation_sessions (
                    session_id,
                    report_id,
                    created_at,
                    original_filename,
                    threshold,
                    tumour_pixels,
                    total_pixels,
                    tumour_percentage,
                    original_path,
                    mask_path,
                    overlay_path,
                    report_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                report_id,
                created_at,
                safe_name,
                float(threshold),
                int(tumour_pixels),
                total_pixels,
                float(tumour_percentage),
                str(original_path),
                str(mask_path),
                str(overlay_path),
                str(report_path)
            ))

            connection.commit()

        finally:
            connection.close()

    except Exception:
        # Avoid leaving incomplete session folders.
        import shutil
        shutil.rmtree(session_dir, ignore_errors=True)
        raise

    return {
        "session_id": session_id,
        "report_id": report_id,
        "created_at": created_at,
        "session_directory": str(session_dir),
        "original_path": str(original_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "report_path": str(report_path)
    }


def get_segmentation_history(limit=100):
    """Return the most recent saved segmentation sessions."""

    initialize_database()

    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute("""
            SELECT *
            FROM segmentation_sessions
            ORDER BY created_at DESC
            LIMIT ?
        """, (max(1, min(int(limit), 1000)),)).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()
