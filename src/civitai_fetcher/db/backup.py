"""
Backup helpers. Both approaches are safe to run against a live DB (WAL mode,
set in connection.get_connection) -- no need to stop anything first.
"""
import os
import sqlite3
from datetime import datetime, timezone


def backup_db(db_path, backup_dir="output/backups"):
    """
    SQLite's own online backup API -- copies the DB page-by-page into a new
    file, consistent even if something else is writing to it concurrently.
    Timestamps the filename so successive backups never overwrite each other.
    """
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_path = os.path.join(backup_dir, f"civitai_{stamp}.db")

    src = sqlite3.connect(db_path)
    dest = sqlite3.connect(dest_path)
    with dest:
        src.backup(dest)
    src.close()
    dest.close()
    print(f"[db] backup written to {dest_path}")
    return dest_path


def vacuum_into(db_path, backup_dir="output/backups"):
    """
    Alternative: VACUUM INTO also defragments as it copies, so the backup
    file can end up smaller than the live DB. Slightly slower than
    backup_db() on a large DB, but a good pick for a periodic/cron backup
    where a bit of extra time doesn't matter.
    """
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_path = os.path.join(backup_dir, f"civitai_{stamp}_vacuum.db")

    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM INTO ?", (dest_path,))
    conn.close()
    print(f"[db] vacuum backup written to {dest_path}")
    return dest_path
