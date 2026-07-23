"""Rebuild data/lancedb from scratch, excluding rows sourced from the two PDFs
corpus/README.md excludes from redistribution (arXiv license: hosting rights
granted to arXiv, not to third parties).

Re-run this after any local re-ingest and before committing data/lancedb --
ingest regenerates all rows, including the excluded ones, from the developer's
local corpus/.

table.delete() is deliberately not used here: it only writes a deletion vector,
leaving the row's bytes recoverable from the fragment file and older version
manifests (verified against a throwaway table -- see DEPLOYMENT_PLAN.md). This
rebuilds a brand-new table containing only the retained rows instead, so the
excluded text is never written into this table's history at all.
"""

from __future__ import annotations

import shutil

import lancedb
import pyarrow as pa

from mmsearch import config, db

EXCLUDED_SOURCE_PATHS = {
    "specs/2106.09685v2.pdf",
    "specs/NIPS-2017-attention-is-all-you-need-Paper.pdf",
}


def main() -> None:
    table = db.open_table()
    arrow_table = table.to_arrow()

    mask = pa.array(
        [sp not in EXCLUDED_SOURCE_PATHS for sp in arrow_table.column("source_path").to_pylist()]
    )
    filtered = arrow_table.filter(mask)
    removed = arrow_table.num_rows - filtered.num_rows
    print(f"Retaining {filtered.num_rows} of {arrow_table.num_rows} rows ({removed} excluded).")

    rebuilt_uri = config.LANCEDB_URI.with_name(config.LANCEDB_URI.name + "_rebuilt")
    if rebuilt_uri.exists():
        shutil.rmtree(rebuilt_uri)

    rebuilt_db = lancedb.connect(str(rebuilt_uri))
    rebuilt_table = rebuilt_db.create_table(config.TABLE_NAME, data=filtered)
    db.ensure_fts_index(rebuilt_table)

    shutil.rmtree(config.LANCEDB_URI)
    rebuilt_uri.rename(config.LANCEDB_URI)
    print(f"Rebuilt {config.LANCEDB_URI} with {filtered.num_rows} rows.")


if __name__ == "__main__":
    main()
