from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mmsearch.schema import Modality, TextSource


@dataclass(frozen=True)
class SearchResult:
    id: str
    modality: Modality
    score: float
    snippet: str
    thumbnail_ref: str
    source_path: str
    text_source: TextSource


class SearchFn(Protocol):
    def __call__(self, query: str, k: int = ...) -> list[SearchResult]: ...
