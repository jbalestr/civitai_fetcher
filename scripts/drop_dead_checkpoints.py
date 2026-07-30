"""
Clear out references to checkpoints Civitai has confirmed deleted -- i.e.
modelVersionIds the resolver cache already tried and got a 404 for (cached
as {"modelId": None, ...}), as opposed to ones simply never attempted yet.

Each dead modelVersionId is classified as one of:
  - PURE NOISE: only used by images that have no modelId at all (i.e. it's
    the sole reason those images can't be attributed to a model). Fully
    swept: images.modelVersionId cleared, image_resources rows deleted,
    the orphaned resources stub deleted.
  - STILL REFERENCED: at least one image with a real, known modelId (e.g.
    from a model-scope fetch) still points at this exact checkpoint
    version. Left completely alone -- that image's modelVersionId is
    genuine historical provenance (it really was generated with that
    checkpoint version), the checkpoint page being gone later doesn't make
    that untrue. Partially clearing this case would also break the
    modelVersionId -> resources foreign key, since that image still needs
    a resources row to point to.

This only touches modelVersionIds the cache confirms were actually
resolved-and-failed -- anything not yet in the cache is left alone (run
resolve_remaining_modelids.py first so "confirmed dead" vs "not yet tried"
is accurate).

Usage:
    uv run python scripts/drop_dead_checkpoints.py           # dry run, shows counts
    uv run python scripts/drop_dead_checkpoints.py --apply   # actually deletes
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH, RESOLVER_CACHE_PATH
from civitai_fetcher.db import get_connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually delete/clear. Without this, dry-run only.")
    args = parser.parse_args()

    if not os.path.exists(RESOLVER_CACHE_PATH):
        print(f"No cache found at {RESOLVER_CACHE_PATH} -- nothing to check against.")
        return

    with open(RESOLVER_CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)
    versions = cache.get("versions", {})

    # confirmed dead: cache has an entry for this version, but modelId is None
    # (i.e. _resolve_version actually tried and got a 404/failure, not just
    # "never looked up")
    dead_ids = [int(vid) for vid, info in versions.items() if info.get("modelId") is None]
    print(f"{len(dead_ids)} modelVersionId(s) confirmed dead (resolved, but model no longer exists).")

    if not dead_ids:
        return

    conn = get_connection(DB_PATH)
    placeholders = ",".join("?" * len(dead_ids))

    # split into "pure noise" (safe to fully sweep) vs "still referenced by
    # an attributed image" (leave entirely alone) -- a dead id is only
    # sweepable if NO image pointing at it already has a real modelId
    still_referenced = {
        r[0] for r in conn.execute(
            f"SELECT DISTINCT modelVersionId FROM images "
            f"WHERE modelVersionId IN ({placeholders}) AND modelId IS NOT NULL",
            dead_ids,
        ).fetchall()
    }
    sweepable_ids = [vid for vid in dead_ids if vid not in still_referenced]

    print(f"{len(still_referenced)} of these are still relied on by an already-attributed image -- "
          f"left entirely alone (genuine historical provenance, not noise)")
    print(f"{len(sweepable_ids)} are pure noise -- safe to fully clear")

    if not sweepable_ids:
        print("\nNothing to do.")
        return

    sweep_placeholders = ",".join("?" * len(sweepable_ids))
    affected_images = conn.execute(
        f"SELECT COUNT(*) FROM images WHERE modelVersionId IN ({sweep_placeholders})", sweepable_ids
    ).fetchone()[0]
    affected_image_resources = conn.execute(
        f"SELECT COUNT(*) FROM image_resources WHERE modelVersionId IN ({sweep_placeholders})", sweepable_ids
    ).fetchone()[0]
    affected_resources = conn.execute(
        f"SELECT COUNT(*) FROM resources WHERE modelVersionId IN ({sweep_placeholders})", sweepable_ids
    ).fetchone()[0]

    print(f"Would clear images.modelVersionId on {affected_images} image(s)")
    print(f"Would delete {affected_image_resources} image_resources row(s)")
    print(f"Would delete {affected_resources} orphaned resources row(s)")

    if not args.apply:
        print("\nDry run only -- re-run with --apply to actually make these changes.")
        return

    conn.execute(f"UPDATE images SET modelVersionId = NULL WHERE modelVersionId IN ({sweep_placeholders})", sweepable_ids)
    conn.execute(f"DELETE FROM image_resources WHERE modelVersionId IN ({sweep_placeholders})", sweepable_ids)
    conn.execute(f"DELETE FROM resources WHERE modelVersionId IN ({sweep_placeholders})", sweepable_ids)
    conn.commit()
    print("\nDone.")


if __name__ == "__main__":
    main()