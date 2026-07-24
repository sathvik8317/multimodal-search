"""One-time seed: copy the local committed data/lancedb index to R2.

Run this once after setting the R2 env vars (MMSEARCH_LANCEDB_URI,
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION) and
before the first /upload. server.py's db.open_table() call refuses to
auto-create a missing R2 table (create_if_missing=False when lancedb_uri is
set -- see db.py) by design: a misconfigured uri/credentials should fail
loud on the live server, not silently serve an empty index. This script is
the one place that's allowed to create it, run manually, once, by a human
who's checked the env vars are right.

Idempotent: does nothing if the R2 table already has rows, so it's safe to
re-run.
"""

from __future__ import annotations

from mmsearch import db
from mmsearch.settings import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.lancedb_uri:
        raise SystemExit("MMSEARCH_LANCEDB_URI is not set -- nothing to seed")

    local_table = db.open_table()  # the committed local index, read-only source
    local_data = local_table.to_arrow()
    print(f"Local index has {local_data.num_rows} rows.")

    storage_options = settings.r2_storage_options()
    try:
        remote_table = db.open_table(
            uri=settings.lancedb_uri, storage_options=storage_options, create_if_missing=False
        )
    except RuntimeError:
        print(f"R2 table not found at {settings.lancedb_uri} -- creating it.")
        remote_table = db.open_table(
            uri=settings.lancedb_uri, storage_options=storage_options, create_if_missing=True
        )

    remote_count = remote_table.count_rows()
    if remote_count > 0:
        print(f"R2 table already has {remote_count} rows -- not re-seeding (idempotent).")
        return

    remote_table.add(local_data)
    db.ensure_fts_index(remote_table)
    print(f"Seeded {local_data.num_rows} rows to {settings.lancedb_uri}.")


if __name__ == "__main__":
    main()
