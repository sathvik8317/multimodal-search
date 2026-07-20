"""`mmsearch ingest <path>` entrypoint (PLAN.md module structure).

The real Cohere client and the real (torch-backed) local captioner are only
imported inside _run_ingest_command's production path, when no fake is
injected -- so importing this module, or running its tests, never loads
torch or touches the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lancedb.table import Table

from mmsearch import db
from mmsearch.clients.protocols import Captioner, EmbeddingClient
from mmsearch.ingest.base import IngestStats, ingest_corpus


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmsearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a corpus directory into the LanceDB index")
    ingest_parser.add_argument("path", type=Path, help="Root directory of the corpus to ingest")

    return parser


def format_report(stats: IngestStats) -> str:
    lines = [f"Ingested {stats.files_processed} files, wrote {stats.rows_written} rows."]

    lines.append("By modality:")
    for modality in sorted(stats.rows_by_modality):
        lines.append(f"  {modality}: {stats.rows_by_modality[modality]}")

    lines.append("By text_source:")
    for text_source in sorted(stats.rows_by_text_source):
        lines.append(f"  {text_source}: {stats.rows_by_text_source[text_source]}")

    if stats.skipped:
        lines.append(f"Skipped {len(stats.skipped)} files:")
        for relpath, reason in stats.skipped:
            lines.append(f"  {relpath}: {reason}")

    return "\n".join(lines)


def _run_ingest_command(
    path: Path,
    *,
    embedding_client: EmbeddingClient | None = None,
    captioner: Captioner | None = None,
    table: Table | None = None,
) -> int:
    if embedding_client is None:
        from mmsearch.clients.cohere import CohereClient

        embedding_client = CohereClient()
    if captioner is None:
        from mmsearch.clients.captioner_local import LocalCaptioner

        captioner = LocalCaptioner()
    if table is None:
        table = db.open_table()

    stats = ingest_corpus(path, embedding_client, captioner, table)
    db.ensure_fts_index(table)
    print(format_report(stats))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _run_ingest_command(args.path)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
