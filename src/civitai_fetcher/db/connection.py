"""
SQLite connection + migration runner.

WAL mode is turned on so reads/writes don't block each other (matters once
enrichment passes and refresh runs might overlap a bit), and so the
`.backup`/`VACUUM INTO` approaches in backup.py stay safe to run concurrently
with a live process.

Migrations are plain numbered .sql files in db/migrations/, applied in
filename order, tracked in schema_version. Add 002_*.sql, 003_*.sql etc. for
future changes -- never edit 001_initial.sql once it's shipped anywhere.
"""
import glob
import os
import sqlite3

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")


def get_connection(path):
    """
    Open a connection with sane defaults for this workload (single-process,
    occasional concurrent reads during enrichment/backup).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _current_version(conn):
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if not tables:
        return 0
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return row["v"] or 0


def migrate(conn):
    """
    Apply any migration files with a version number higher than what's
    already been applied. Safe to call every run -- no-op if nothing's new.
    """
    current = _current_version(conn)
    migration_files = sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "*.sql")))

    applied = 0
    for path in migration_files:
        fname = os.path.basename(path)
        version = int(fname.split("_", 1)[0])
        if version <= current:
            continue
        with open(path) as f:
            sql = f.read()
        conn.executescript(sql)
        conn.commit()
        applied += 1
        print(f"[db] applied migration {fname}")

    if applied == 0:
        print(f"[db] schema up to date (version {current})")
    return conn
