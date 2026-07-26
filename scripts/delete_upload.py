"""Delete uploaded rows (moderation).

Usage:
    python scripts/delete_upload.py alice        # delete only alice's rows
    python scripts/delete_upload.py --all-uploads # delete every uploaded row

Every /upload row carries a "uploads/<uploader>/..." source_path (see
ingest/upload.py) -- that's the entire moderation surface the shared+tagged
content model promises (UPLOAD_PLAN.md): no UI, just this predicate delete,
run by the owner. Connects to whichever table is live (R2 if
MMSEARCH_LANCEDB_URI is set, local otherwise), same as the running server.
"""

from __future__ import annotations

import argparse

from mmsearch import config, db
from mmsearch.ingest.upload import _sanitize_uploader
from mmsearch.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "uploader", nargs="?", default=None, help="delete only this uploader's rows"
    )
    group.add_argument(
        "--all-uploads", action="store_true", help="delete every uploaded row, all uploaders"
    )
    args = parser.parse_args()

    settings = get_settings()
    table = db.open_table(
        uri=settings.lancedb_uri or config.LANCEDB_URI,
        storage_options=settings.r2_storage_options(),
        create_if_missing=False,
    )

    if args.all_uploads:
        predicate = "source_path LIKE 'uploads/%'"
    else:
        # Same sanitization ingest_uploaded_file applies when it writes the
        # row, so this matches exactly what's actually stored.
        predicate = f"source_path LIKE 'uploads/{_sanitize_uploader(args.uploader)}/%'"

    before = table.count_rows()
    table.delete(predicate)
    after = table.count_rows()
    print(f"Deleted {before - after} row(s) matching: {predicate}")


if __name__ == "__main__":
    main()
