"""
Restore a backup produced by backup_db()/vacuum_into() back over the live DB.

Usage:
    uv run python scripts/restore_db.py output/backups/civitai_20260729T081530Z.db
    uv run python scripts/restore_db.py --list          # see what's available to restore
"""
import argparse
import glob
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH


def list_backups(backup_dir="output/backups"):
    paths = sorted(glob.glob(os.path.join(backup_dir, "*.db")))
    if not paths:
        print(f"No backups found in {backup_dir}/")
        return
    print(f"Backups in {backup_dir}/:")
    for p in paths:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  {p}  ({size_mb:.1f} MB)")


def restore(backup_path, db_path=DB_PATH, yes=False):
    if not os.path.exists(backup_path):
        print(f"Backup file not found: {backup_path}")
        sys.exit(1)

    if os.path.exists(db_path):
        if not yes:
            resp = input(
                f"This will overwrite {db_path} with {backup_path}. "
                f"The current DB will be safety-copied first. Continue? [y/N] "
            )
            if resp.strip().lower() != "y":
                print("Cancelled.")
                return
        # Never overwrite blind: snapshot whatever's currently live before
        # touching it, in case the restore was a mistake.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        pre_restore_copy = db_path.replace(".db", f"_pre_restore_{stamp}.db")
        shutil.copy2(db_path, pre_restore_copy)
        print(f"Safety-copied current DB to {pre_restore_copy}")

        # WAL/SHM sidecar files belong to the old DB's journal -- stale ones
        # left next to a freshly-restored file can confuse SQLite, so clear them.
        for ext in ("-wal", "-shm"):
            sidecar = db_path + ext
            if os.path.exists(sidecar):
                os.remove(sidecar)

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    shutil.copy2(backup_path, db_path)
    print(f"Restored {backup_path} -> {db_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup_path", nargs="?", help="Path to the backup .db file to restore")
    parser.add_argument("--list", action="store_true", help="List available backups and exit")
    parser.add_argument("--backup-dir", default="output/backups", help="Directory to list backups from (with --list)")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    if args.list or not args.backup_path:
        list_backups(args.backup_dir)
        if not args.backup_path:
            return

    restore(args.backup_path, yes=args.yes)


if __name__ == "__main__":
    main()
