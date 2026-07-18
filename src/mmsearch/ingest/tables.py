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


def ingest_table(
    table_path: Path,
    corpus_root: Path,
    embedding_client: EmbeddingClient,
    max_rows: int = config.MAX_TABLE_ROWS,
) -> Row:
    with open(table_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    columns = rows[0] if rows else []
    data_rows = rows[1:]
    total_rows = len(data_rows)
    truncated = total_rows > max_rows
    embedded_rows = data_rows[:max_rows] if truncated else data_rows

    content_text = _rows_to_markdown(columns, embedded_rows)

    vectors = embedding_client.embed_documents([EmbedInput(text=content_text)])
    vector = vectors[0]

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
        vector=vector,
        source_path=relpath,
        thumbnail_ref="",
        metadata=metadata,
    )
