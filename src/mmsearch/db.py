from __future__ import annotations

from pathlib import Path

import lancedb
from lancedb.index import FTS
from lancedb.table import Table

from mmsearch import config
from mmsearch.schema import ARROW_SCHEMA, Row, rows_to_table


def open_table(
    uri: str | Path = config.LANCEDB_URI,
    table_name: str = config.TABLE_NAME,
    storage_options: dict[str, str] | None = None,
    create_if_missing: bool = True,
) -> Table:
    db = lancedb.connect(str(uri), storage_options=storage_options)
    if table_name in db.list_tables().tables:
        return db.open_table(table_name)
    if not create_if_missing:
        raise RuntimeError(
            f"LanceDB table {table_name!r} not found at {uri} and create_if_missing=False "
            "-- refusing to silently serve an empty table (likely a misconfigured uri/credentials)"
        )
    return db.create_table(table_name, schema=ARROW_SCHEMA)


def ensure_fts_index(table: Table, column: str = "content_text") -> None:
    table.create_index(column, config=FTS(), replace=True)


def upsert(table: Table, rows: list[Row]) -> None:
    if not rows:
        return
    data = rows_to_table(rows)
    (
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(data)
    )
