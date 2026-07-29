"""
Back up the live DB. Safe to run anytime, including while other commands
are using the DB (WAL mode).

Usage:
    uv run python scripts/backup_db.py                # standard online backup
    uv run python scripts/backup_db.py --vacuum        # also defragments, smaller file, a bit slower
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import backup_db, vacuum_into


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vacuum", action="store_true",
                        help="Use VACUUM INTO instead -- also defragments (smaller file, a bit slower).")
    parser.add_argument("--backup-dir", default="output/backups")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"No DB found at {DB_PATH} -- nothing to back up.")
        sys.exit(1)

    if args.vacuum:
        vacuum_into(DB_PATH, backup_dir=args.backup_dir)
    else:
        backup_db(DB_PATH, backup_dir=args.backup_dir)


if __name__ == "__main__":
    main()
