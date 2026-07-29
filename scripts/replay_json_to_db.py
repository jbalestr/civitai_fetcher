"""
Replay existing civitai_output*.json files into the DB — no API calls.

Usage:
    uv run python scripts/replay_json_to_db.py output/*.json
    uv run python scripts/replay_json_to_db.py output/civitai_output_Week_28Jul26_1130.json

Safe to run repeatedly / on overlapping files: upsert_entries() is keyed by
imageId, so replaying the same file twice just re-updates the same rows
(image_stats is the exception -- it's append-only, but INSERT OR IGNORE on
the (imageId, fetched_at) key means replaying the same file twice doesn't
duplicate history rows either, since fetched_at is derived per-file below).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection, migrate, upsert_entries


def _fetched_at_for_file(path):
    """
    Use the file's own mtime as the fetched_at stamp for image_stats, so
    reaction-count history reflects when each JSON dump was actually
    produced, not "now" (which would be wrong/misleading for old files).
    """
    from datetime import datetime, timezone
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def replay_file(conn, path):
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"  skipping {path}: not a list of image entries (issues/meta files aren't replayable)")
        return 0, 0

    fetched_at = _fetched_at_for_file(path)
    new_c, upd_c = upsert_entries(conn, data, fetched_at=fetched_at)
    print(f"  {path}: {len(data)} entries -> {new_c} new, {upd_c} updated")
    return new_c, upd_c


def main():
    if len(sys.argv) < 2:
        print("Usage: replay_json_to_db.py <file.json | glob pattern> [more files...]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        matched = glob.glob(arg)
        paths.extend(matched if matched else [arg])
    paths = sorted(set(paths))

    # skip the issues/meta sidecar files json.dump also writes -- they're
    # diagnostics, not image records, and replay_file already guards against
    # them, but filtering here keeps the printed plan honest up front.
    candidate_paths = [p for p in paths if "_issues_" not in p and not p.endswith("_meta.json")]
    skipped = set(paths) - set(candidate_paths)
    if skipped:
        print(f"Skipping {len(skipped)} non-image-record file(s): {', '.join(sorted(skipped))}")

    conn = get_connection(DB_PATH)
    migrate(conn)

    total_new, total_upd = 0, 0
    print(f"Replaying {len(candidate_paths)} file(s) into {DB_PATH}...")
    for path in candidate_paths:
        n, u = replay_file(conn, path)
        total_new += n
        total_upd += u

    print(f"Done: {total_new} new row(s), {total_upd} updated row(s) total.")


if __name__ == "__main__":
    main()
