from pathlib import Path

from mmsearch import db
from mmsearch.clients.fakes import FakeCaptioner, FakeEmbeddingClient
from mmsearch.clients.protocols import Embedders
from mmsearch.ingest.base import IngestStats
from mmsearch.ingest.cli import _run_ingest_command, build_arg_parser, format_report, main

EMBEDDERS = Embedders(image=FakeEmbeddingClient(), text=FakeEmbeddingClient())


# --- build_arg_parser -----------------------------------------------------------------

def test_ingest_subcommand_parses_path():
    parser = build_arg_parser()
    args = parser.parse_args(["ingest", "corpus/"])
    assert args.command == "ingest"
    assert args.path == Path("corpus/")


def test_missing_command_is_a_parse_error():
    parser = build_arg_parser()
    try:
        parser.parse_args([])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code != 0


# --- format_report ----------------------------------------------------------------------

def test_format_report_includes_totals_and_breakdowns():
    stats = IngestStats(
        rows_written=4,
        rows_by_modality={"pdf_page": 2, "table": 1, "code": 1},
        rows_by_text_source={"pdf_text_layer": 2, "table_markdown": 1, "code_source": 1},
        files_processed=4,
        skipped=[],
    )
    report = format_report(stats)
    assert "4 rows" in report or "wrote 4 rows" in report
    assert "pdf_page: 2" in report
    assert "table: 1" in report
    assert "code: 1" in report
    assert "pdf_text_layer: 2" in report


def test_format_report_lists_skipped_files_with_reasons():
    stats = IngestStats(skipped=[("README.md", "unsupported extension '.md'")])
    report = format_report(stats)
    assert "README.md" in report
    assert "unsupported extension '.md'" in report


def test_format_report_omits_skipped_section_when_nothing_skipped():
    stats = IngestStats(rows_written=1, rows_by_modality={"table": 1}, rows_by_text_source={"table_markdown": 1})
    report = format_report(stats)
    assert "Skipped" not in report


# --- _run_ingest_command (wiring, with injected fakes -- no real network/torch) --------------

def test_run_ingest_command_ingests_and_returns_zero(tmp_path, capsys):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "table.csv").write_text("a,b\n1,2\n")
    table = db.open_table(uri=tmp_path / "lancedb")

    exit_code = _run_ingest_command(
        corpus_root,
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
    )

    assert exit_code == 0
    assert table.count_rows() == 1
    captured = capsys.readouterr()
    assert "table: 1" in captured.out


def test_run_ingest_command_builds_fts_index(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "table.csv").write_text("needle,other\nfoundit,x\n")
    table = db.open_table(uri=tmp_path / "lancedb")

    _run_ingest_command(
        corpus_root,
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
    )

    results = table.search("needle", query_type="fts").to_list()
    assert len(results) == 1


# --- main() dispatch ----------------------------------------------------------------------

def test_main_dispatches_ingest_command_to_run_ingest_command(monkeypatch, tmp_path):
    calls = []

    def fake_run(path, **kwargs):
        calls.append(path)
        return 0

    monkeypatch.setattr("mmsearch.ingest.cli._run_ingest_command", fake_run)

    exit_code = main(["ingest", str(tmp_path)])

    assert exit_code == 0
    assert calls == [tmp_path]

# .env loading no longer happens in cli.py -- it moved to CohereClient's lazy
# Settings() read (clients/cohere.py), covered by test_settings.py's
# test_dotenv_file_is_honored_cwd_relative and test_cohere_client.py.
