"""Tree-sitter symbol-aware code ingestion (PLAN.md §(b)).

Chunks a source file into one Row per top-level function, per method
(function nested directly inside a class body), and per zero-method class
(a pure data-container -- e.g. a @dataclass -- indexed by its docstring and
field signatures rather than a method body), instead of fixed-size splits.
Each chunk is embedded as a context header (file path, language, enclosing
class) followed by the exact source slice for that symbol. Decorators are
unwrapped so decorated functions/classes are still found, and included in
the embedded source slice as useful context (e.g. "@dataclass").
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
    kind: str  # "function" | "method" | "class"
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
                vector_openai=vector,
                source_path=relpath,
                thumbnail_ref="",
                metadata=metadata,
            )
        )
    return rows


def _find_symbols(root_node) -> list[_Symbol]:
    """Top-level function_definitions, method function_definitions nested
    directly inside a class body, and -- only when a class has zero methods
    -- one class-level symbol built from its docstring and field signatures.
    A class with at least one method never also produces a class-level
    symbol; a class with neither a docstring nor any field assignment (e.g.
    just `pass`) produces nothing, since there would be no content to embed.
    """
    symbols: list[_Symbol] = []
    for child in root_node.children:
        resolved = _unwrap_decorated(child)
        if resolved is None:
            continue
        def_node, span_start_byte, span_start_point = resolved

        if def_node.type == "function_definition":
            symbols.append(_function_symbol(def_node, span_start_byte, span_start_point))
        elif def_node.type == "class_definition":
            class_name = _node_name(def_node)
            methods = _find_methods(def_node, class_name)
            if methods:
                symbols.extend(methods)
            else:
                class_symbol = _class_symbol(def_node, class_name, span_start_byte, span_start_point)
                if class_symbol is not None:
                    symbols.append(class_symbol)
    return symbols


def _unwrap_decorated(node):
    """Resolve a possibly-decorated definition node.

    Returns (def_node, span_start_byte, span_start_point) where def_node is
    the underlying function_definition/class_definition, and span_start_*
    comes from the decorator wrapper when present, so the decorator line(s)
    are included in the embedded source slice as context. Returns None for
    any other node type.
    """
    if node.type == "decorated_definition":
        inner = next(
            (c for c in node.children if c.type in ("function_definition", "class_definition")),
            None,
        )
        if inner is None:
            return None
        return inner, node.start_byte, node.start_point
    if node.type in ("function_definition", "class_definition"):
        return node, node.start_byte, node.start_point
    return None


def _function_symbol(def_node, span_start_byte: int, span_start_point) -> _Symbol:
    return _Symbol(
        start_byte=span_start_byte,
        end_byte=def_node.end_byte,
        start_line=span_start_point[0] + 1,
        end_line=def_node.end_point[0] + 1,
        qualname=_node_name(def_node),
        kind="function",
        enclosing_class=None,
    )


def _find_methods(class_node, class_name: str) -> list[_Symbol]:
    body = _class_body(class_node)
    if body is None:
        return []
    methods = []
    for member in body.children:
        resolved = _unwrap_decorated(member)
        if resolved is None:
            continue
        def_node, span_start_byte, span_start_point = resolved
        if def_node.type != "function_definition":
            continue
        methods.append(
            _Symbol(
                start_byte=span_start_byte,
                end_byte=def_node.end_byte,
                start_line=span_start_point[0] + 1,
                end_line=def_node.end_point[0] + 1,
                qualname=f"{class_name}.{_node_name(def_node)}",
                kind="method",
                enclosing_class=class_name,
            )
        )
    return methods


def _class_symbol(class_node, class_name: str, span_start_byte: int, span_start_point) -> _Symbol | None:
    """Build a class-level symbol from a zero-method class's docstring and/or
    field assignments. Returns None if the class has neither (nothing to embed).
    """
    body = _class_body(class_node)
    if body is None:
        return None
    has_content = any(member.type in ("string", "assignment") for member in body.children)
    if not has_content:
        return None
    return _Symbol(
        start_byte=span_start_byte,
        end_byte=class_node.end_byte,
        start_line=span_start_point[0] + 1,
        end_line=class_node.end_point[0] + 1,
        qualname=class_name,
        kind="class",
        enclosing_class=None,
    )


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
