"""
Shared "also write to the DB" step for the CLI commands (models/images/creators).

Kept deliberately best-effort: the JSON file is still the primary output
every downstream tool (patterns_cli, etc.) reads, so a DB problem should
never stop a run or lose the JSON the user actually asked for. Mirrors the
same never-fatal philosophy already used for --enrich-meta and the resolver
cache elsewhere in this codebase.
"""
from .connection import get_connection, migrate
from .store import upsert_entries


def write_to_db(ranked, db_path):
    """
    Upsert `ranked` (the same list about to be json.dump()'d) into the DB.
    Prints a one-line summary; swallows and reports any failure rather than
    raising, so it never turns a successful fetch into a failed run.
    """
    try:
        conn = get_connection(db_path)
        migrate(conn)
        new_c, upd_c, skip_c, blocked_c = upsert_entries(conn, ranked)
        notes = []
        if skip_c:
            notes.append(f"{skip_c} skipped (checkpoint spam)")
        if blocked_c:
            notes.append(f"{blocked_c} skipped (blocked creator)")
        note_str = f", {', '.join(notes)}" if notes else ""
        print(f"[db] upserted into {db_path}: {new_c} new, {upd_c} updated{note_str}")
        conn.close()
    except Exception as e:
        print(f"[db] WARNING: write to {db_path} failed, JSON output above is unaffected: {e}")