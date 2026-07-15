"""Tree-sitter symbol-aware code ingestion (PLAN.md §(b)).

Chunks a source file into one Row per top-level function and per method
(function nested directly inside a class body), rather than fixed-size
splits. Each chunk is embedded as a context header (file path, language,
enclosing class) followed by the exact source slice for that symbol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import get_parser

from mmsearch.clients.protocols import EmbeddingClient, EmbedInput
from mmsearch.schema import Modality, Row, TextSource, make_id

# Only Python is a hard requirement; other languages are a stretch goal.
_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
}


@dataclass(frozen=True)
class _Symbol:
    start_byte: int
    end_byte: int
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    qualname: str
    kind: str  # "function" | "method"
    enclosing_class: str | None


def ingest_code_file(path: Path, corpus_root: Path, embedding_client: EmbeddingClient) -> list[Row]:
    """Parse a code file into symbol-aware chunks and embed each as a Row."""
    language = _LANGUAGE_BY_SUFFIX.get(path.suffix)
    if language is None:
        raise ValueError(f"unsupported code file extension: {path.suffix!r}")

    relpath = path.relative_to(corpus_root).as_posix()
    source_bytes = path.read_bytes()

    parser = get_parser(language)
    tree = parser.parse(source_bytes)
    symbols = _find_symbols(tree.root_node)

    if not symbols:
        return []

    context_bodies = [
        _build_context_and_body(symbol, relpath, language, source_bytes) for symbol in symbols
    ]
    vectors = embedding_client.embed_documents([EmbedInput(text=cb) for cb in context_bodies])

    rows = []
    for symbol, context_body, vector in zip(symbols, context_bodies, vectors):
        metadata = json.dumps(
            {
                "lang": language,
                "qualname": symbol.qualname,
                "kind": symbol.kind,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
            }
        )
        rows.append(
            Row(
                id=make_id(Modality.CODE, relpath, qualname=symbol.qualname),
                modality=Modality.CODE,
                content_text=context_body,
                text_source=TextSource.CODE_SOURCE,
                vector=vector,
                source_path=relpath,
                thumbnail_ref="",
                metadata=metadata,
            )
        )
    return rows


def _find_symbols(root_node) -> list[_Symbol]:
    """Top-level function_definitions, plus method function_definitions
    nested directly inside a class_definition's body. Classes themselves
    never produce a symbol/row.
    """
    symbols: list[_Symbol] = []
    for child in root_node.children:
        if child.type == "function_definition":
            name = _node_name(child)
            symbols.append(
                _Symbol(
                    start_byte=child.start_byte,
                    end_byte=child.end_byte,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    qualname=name,
                    kind="function",
                    enclosing_class=None,
                )
            )
        elif child.type == "class_definition":
            class_name = _node_name(child)
            body = _class_body(child)
            if body is None:
                continue
            for member in body.children:
                if member.type == "function_definition":
                    method_name = _node_name(member)
                    symbols.append(
                        _Symbol(
                            start_byte=member.start_byte,
                            end_byte=member.end_byte,
                            start_line=member.start_point[0] + 1,
                            end_line=member.end_point[0] + 1,
                            qualname=f"{class_name}.{method_name}",
                            kind="method",
                            enclosing_class=class_name,
                        )
                    )
    return symbols


def _node_name(node) -> str:
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8")
    raise ValueError(f"{node.type} node has no identifier child")


def _class_body(class_node):
    for child in class_node.children:
        if child.type == "block":
            return child
    return None


def _build_context_and_body(symbol: _Symbol, relpath: str, language: str, source_bytes: bytes) -> str:
    body_text = source_bytes[symbol.start_byte : symbol.end_byte].decode("utf-8")
    header_lines = [f"# file: {relpath}", f"# language: {language}"]
    if symbol.enclosing_class is not None:
        header_lines.append(f"# class: {symbol.enclosing_class}")
    return "\n".join(header_lines) + "\n" + body_text
