"""
Sanity-check for duplicate imageIds.

- In the DB: imageId is the PRIMARY KEY on `images`, so true duplicates are
  physically impossible there -- this just confirms it and shows row count.
- Across JSON files in an output folder: the same imageId can legitimately
  appear in multiple files (overlapping fetch date ranges), and that's fine
  since upsert_entries() just re-updates the same row. This reports how
  much overlap exists, in case it's more than expected.

Usage:
    uv run python scripts/check_duplicate_imageids.py            # DB check only
    uv run python scripts/check_duplicate_imageids.py output/*.json  # + JSON overlap check
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection


def check_db():
    conn = get_connection(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    distinct = conn.execute("SELECT COUNT(DISTINCT imageId) FROM images").fetchone()[0]
    print(f"DB: {total} row(s), {distinct} distinct imageId(s) "
          f"-> {'OK, no duplicates possible (PRIMARY KEY)' if total == distinct else 'unexpected mismatch!'}")


def check_json_files(paths):
    seen = defaultdict(list)  # imageId -> [file, file, ...]
    skipped = []

    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((path, str(e)))
            continue

        if not isinstance(data, list):
            skipped.append((path, "not a list of entries"))
            continue

        for entry in data:
            image_id = entry.get("imageId")
            if image_id is not None:
                seen[image_id].append(os.path.basename(path))

    dupes = {iid: files for iid, files in seen.items() if len(files) > 1}
    total_entries = sum(len(files) for files in seen.values())

    print(f"\nJSON files: {len(paths)} file(s) read ({len(skipped)} skipped)")
    print(f"Total entries: {total_entries}, distinct imageIds: {len(seen)}, "
          f"appearing in >1 file: {len(dupes)}")

    if dupes:
        print("\nExamples (imageId -> files it appeared in):")
        for iid, files in list(dupes.items())[:15]:
            print(f"  {iid}: {files}")
        if len(dupes) > 15:
            print(f"  ... and {len(dupes) - 15} more")

    if skipped:
        print("\nSkipped files:")
        for path, reason in skipped[:10]:
            print(f"  {path}: {reason}")


def main():
    check_db()

    if len(sys.argv) > 1:
        paths = []
        for arg in sys.argv[1:]:
            paths.extend(glob.glob(arg))
        paths = [p for p in paths if "_issues" not in p and "_meta" not in p]
        check_json_files(paths)


if __name__ == "__main__":
    main()