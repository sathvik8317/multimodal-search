"""CSV table ingestion: parses a CSV file into a single markdown-table Row."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mmsearch import config
from mmsearch.clients.protocols import EmbedInput, EmbeddingClient
from mmsearch.schema import Modality, Row, TextSource, make_id


def _rows_to_markdown(columns: list[str], data_rows: list[list[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for data_row in data_rows:
        lines.append("| " + " | ".join(data_row) + " |")
    return "\n".join(lines)


def _select_embedded_rows(
    columns: list[str], data_rows: list[list[str]], max_rows: int, max_chars: int
) -> list[list[str]]:
    """Cap by row count AND by a conservative character budget, whichever
    binds first. A wide table (many columns) can blow an embedding
    provider's token limit well before hitting max_rows -- MAX_TABLE_ROWS
    was tuned against Cohere's much larger context; OpenAI's
    text-embedding-3-small hard-rejects past 8192 tokens (confirmed live:
    3 of 4 real corpus CSVs -- 12-18 columns -- exceeded it at the
    200-row cap while a 9-column CSV did not). At least one data row is
    always kept even if it alone would exceed the budget.
    """
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    budget = max_chars - len(header) - len(separator) - 2  # 2 newlines
    selected: list[list[str]] = []
    used = 0
    for data_row in data_rows[:max_rows]:
        line = "| " + " | ".join(data_row) + " |"
        line_len = len(line) + 1  # + newline
        if selected and used + line_len > budget:
            break
        selected.append(data_row)
        used += line_len
    return selected


def ingest_table(
    table_path: Path,
    corpus_root: Path,
    embedding_client: EmbeddingClient,
    max_rows: int = config.MAX_TABLE_ROWS,
    max_chars: int = config.MAX_TABLE_EMBED_CHARS,
) -> Row:
    with open(table_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    columns = rows[0] if rows else []
    data_rows = rows[1:]
    total_rows = len(data_rows)
    embedded_rows = _select_embedded_rows(columns, data_rows, max_rows, max_chars)
    truncated = len(embedded_rows) < total_rows

    content_text = _rows_to_markdown(columns, embedded_rows)

    vectors = embedding_client.embed_documents([EmbedInput(text=content_text)])
    vector_openai = vectors[0]

    relpath = table_path.relative_to(corpus_root).as_posix()

    metadata = json.dumps(
        {
            "n_rows": len(embedded_rows),
            "n_cols": len(columns),
            "columns": columns,
            "truncated": truncated,
            "total_rows": total_rows,
        }
    )

    return Row(
        id=make_id(Modality.TABLE, relpath),
        modality=Modality.TABLE,
        content_text=content_text,
        text_source=TextSource.TABLE_MARKDOWN,
        vector_openai=vector_openai,
        source_path=relpath,
        thumbnail_ref="",
        metadata=metadata,
    )
