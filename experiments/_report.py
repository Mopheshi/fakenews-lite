"""Shared helpers for writing experiment results.

Every experiment emits the same two things: a JSON record for downstream tooling
and a Markdown table that can be pasted straight into the paper. Keeping that in
one place means the numbers in the paper and the numbers in the artifacts cannot
drift apart.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fakenews.config import RESULTS_DIR, ensure_directories

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    """Render rows as a GitHub-flavoured Markdown table."""
    if not rows:
        return "_(no rows)_"

    columns = list(columns or rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(
    name: str,
    *,
    payload: Mapping[str, Any],
    tables: Iterable[tuple[str, str]] = (),
    directory: Path | None = None,
) -> tuple[Path, Path]:
    """Write ``name.json`` and ``name.md``.

    Args:
        payload: Machine-readable record.
        tables: ``(heading, markdown)`` pairs written into the Markdown report.

    Returns:
        ``(json_path, markdown_path)``.
    """
    ensure_directories()
    target = directory or RESULTS_DIR

    json_path = target / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    sections = [f"# {name.replace('_', ' ').title()}", ""]
    for heading, table in tables:
        sections.extend([f"## {heading}", "", table, ""])

    markdown_path = target / f"{name}.md"
    markdown_path.write_text("\n".join(sections), encoding="utf-8")

    logger.info("Wrote %s and %s", json_path.name, markdown_path.name)
    return json_path, markdown_path


def load_result(name: str, *, directory: Path | None = None) -> dict:
    """Load a previously written result JSON."""
    path = (directory or RESULTS_DIR) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}; run the experiment that produces it first.")
    return json.loads(path.read_text(encoding="utf-8"))
