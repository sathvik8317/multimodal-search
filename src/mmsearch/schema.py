from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pyarrow as pa

from mmsearch import config


class Modality(str, Enum):
    PDF_PAGE = "pdf_page"
    DIAGRAM = "diagram"
    TABLE = "table"
    CODE = "code"


class TextSource(str, Enum):
    PDF_TEXT_LAYER = "pdf_text_layer"
    VLM_CAPTION = "vlm_caption"
    TABLE_MARKDOWN = "table_markdown"
    CODE_SOURCE = "code_source"


_ID_PREFIX = {
    Modality.PDF_PAGE: "pdf",
    Modality.DIAGRAM: "img",
    Modality.TABLE: "tbl",
    Modality.CODE: "code",
}


def _normalize_relpath(relpath: str) -> str:
    return str(relpath).replace("\\", "/").lstrip("/")


def make_id(
    modality: Modality,
    relpath: str,
    *,
    page_no: int | None = None,
    qualname: str | None = None,
) -> str:
    """Build the deterministic, human-readable id for a row.

    Formats (see PLAN.md §2.1):
      pdf_page: pdf:{relpath}#p{page_no}
      diagram:  img:{relpath}
      table:    tbl:{relpath}
      code:     code:{relpath}#{qualname}
    """
    relpath = _normalize_relpath(relpath)
    prefix = _ID_PREFIX[modality]

    if modality is Modality.PDF_PAGE:
        if page_no is None:
            raise ValueError("pdf_page id requires page_no")
        return f"{prefix}:{relpath}#p{page_no}"

    if modality is Modality.CODE:
        if not qualname:
            raise ValueError("code id requires qualname")
        return f"{prefix}:{relpath}#{qualname}"

    return f"{prefix}:{relpath}"


@dataclass
class Row:
    id: str
    modality: Modality
    content_text: str
    text_source: TextSource
    source_path: str
    # Two provider-specific embedding spaces (see EMBEDDING_MIGRATION_PLAN.md):
    # vector_cohere is Cohere Embed v4 over the page/diagram image; vector_openai
    # is OpenAI text-embedding-3-small over table/code/caption text. A row must
    # populate at least one; diagram/scanned-page rows populate both.
    vector_cohere: list[float] | None = None
    vector_openai: list[float] | None = None
    thumbnail_ref: str = ""
    metadata: str = "{}"

    def __post_init__(self) -> None:
        if self.vector_cohere is None and self.vector_openai is None:
            raise ValueError("at least one of vector_cohere/vector_openai must be set")
        if self.vector_cohere is not None and len(self.vector_cohere) != config.COHERE_EMBED_DIM:
            raise ValueError(
                f"vector_cohere length {len(self.vector_cohere)} != "
                f"config.COHERE_EMBED_DIM ({config.COHERE_EMBED_DIM})"
            )
        if self.vector_openai is not None and len(self.vector_openai) != config.OPENAI_EMBED_DIM:
            raise ValueError(
                f"vector_openai length {len(self.vector_openai)} != "
                f"config.OPENAI_EMBED_DIM ({config.OPENAI_EMBED_DIM})"
            )
        if not self.content_text:
            raise ValueError("content_text must not be empty (see PLAN.md decision 5)")


ARROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("modality", pa.string()),
        pa.field("content_text", pa.string()),
        pa.field("text_source", pa.string()),
        pa.field("vector_cohere", pa.list_(pa.float32(), config.COHERE_EMBED_DIM)),
        pa.field("vector_openai", pa.list_(pa.float32(), config.OPENAI_EMBED_DIM)),
        pa.field("source_path", pa.string()),
        pa.field("thumbnail_ref", pa.string()),
        pa.field("metadata", pa.string()),
    ]
)


def rows_to_table(rows: list[Row]) -> pa.Table:
    columns: dict[str, list] = {name: [] for name in ARROW_SCHEMA.names}
    for row in rows:
        columns["id"].append(row.id)
        columns["modality"].append(row.modality.value)
        columns["content_text"].append(row.content_text)
        columns["text_source"].append(row.text_source.value)
        columns["vector_cohere"].append(row.vector_cohere)
        columns["vector_openai"].append(row.vector_openai)
        columns["source_path"].append(row.source_path)
        columns["thumbnail_ref"].append(row.thumbnail_ref)
        columns["metadata"].append(row.metadata)
    return pa.table(columns, schema=ARROW_SCHEMA)
