"""PDF page ingestion (rasterized, ColPali-style) and diagram PNG ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
from PIL import Image

from mmsearch import config
from mmsearch.clients.protocols import Captioner, EmbedInput, EmbeddingClient
from mmsearch.ingest.thumbnails import write_thumbnail
from mmsearch.schema import Modality, Row, TextSource, make_id


def ingest_pdf(
    pdf_path: Path,
    corpus_root: Path,
    embedding_client: EmbeddingClient,
    captioner: Captioner,
    thumbnails_dir: Path = config.THUMBNAILS_DIR,
) -> list[Row]:
    relpath = pdf_path.relative_to(corpus_root).as_posix()

    rows: list[Row] = []
    doc = fitz.open(pdf_path)
    try:
        n_pages = doc.page_count
        for index, page in enumerate(doc):
            page_no = index + 1
            png_bytes = page.get_pixmap().tobytes("png")

            vectors = embedding_client.embed_documents([EmbedInput(image_bytes=png_bytes)])
            vector = vectors[0]

            text_layer = page.get_text().strip()
            if text_layer:
                content_text = page.get_text()
                text_source = TextSource.PDF_TEXT_LAYER
            else:
                content_text = captioner.caption(png_bytes)
                text_source = TextSource.VLM_CAPTION

            thumb_relpath = f"{relpath}#p{page_no}.png"
            thumbnail_ref = write_thumbnail(png_bytes, thumb_relpath, thumbnails_dir=thumbnails_dir)

            metadata = json.dumps({"page_no": page_no, "n_pages": n_pages})

            rows.append(
                Row(
                    id=make_id(Modality.PDF_PAGE, relpath, page_no=page_no),
                    modality=Modality.PDF_PAGE,
                    content_text=content_text,
                    text_source=text_source,
                    vector=vector,
                    source_path=relpath,
                    thumbnail_ref=thumbnail_ref,
                    metadata=metadata,
                )
            )
    finally:
        doc.close()

    return rows


def ingest_diagram(
    image_path: Path,
    corpus_root: Path,
    embedding_client: EmbeddingClient,
    captioner: Captioner,
    thumbnails_dir: Path = config.THUMBNAILS_DIR,
) -> Row:
    relpath = image_path.relative_to(corpus_root).as_posix()
    image_bytes = image_path.read_bytes()

    vectors = embedding_client.embed_documents([EmbedInput(image_bytes=image_bytes)])
    vector = vectors[0]

    content_text = captioner.caption(image_bytes)

    with Image.open(image_path) as img:
        width, height = img.size

    thumbnail_ref = write_thumbnail(image_bytes, relpath, thumbnails_dir=thumbnails_dir)

    metadata = json.dumps(
        {
            "width": width,
            "height": height,
            "caption_model": type(captioner).__name__,
        }
    )

    return Row(
        id=make_id(Modality.DIAGRAM, relpath),
        modality=Modality.DIAGRAM,
        content_text=content_text,
        text_source=TextSource.VLM_CAPTION,
        vector=vector,
        source_path=relpath,
        thumbnail_ref=thumbnail_ref,
        metadata=metadata,
    )
