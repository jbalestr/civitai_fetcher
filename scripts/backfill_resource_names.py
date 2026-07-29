"""
Backfill name/versionName/creatorUsername for `resources` rows that predate
the generation.py field-naming fix and/or the resolve-order fix -- i.e. rows
that came from an --enrich-meta run under the old code and never got a
proper resolve.py pass.

This DOES make live API calls (one per not-yet-cached modelVersionId/modelId,
same as a normal resolve pass) -- it just does it against what's already in
the DB, without needing to re-fetch or re-run any CLI command.

Usage:
    uv run python scripts/backfill_resource_names.py
    uv run python scripts/backfill_resource_names.py --dry-run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection, migrate
from civitai_fetcher.services.enrichment.resolve import (
    enrich_resources, load_cache as load_resolver_cache, save_cache as save_resolver_cache,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be resolved/updated without writing to the DB.")
    args = parser.parse_args()

    conn = get_connection(DB_PATH)
    migrate(conn)

    rows = conn.execute(
        "SELECT modelVersionId FROM resources WHERE name IS NULL OR creatorUsername IS NULL"
    ).fetchall()
    version_ids = [r["modelVersionId"] for r in rows]

    if not version_ids:
        print("Nothing to backfill -- every resource row already has name + creatorUsername.")
        return

    print(f"Found {len(version_ids)} resource row(s) missing name and/or creatorUsername.")
    if args.dry_run:
        print("--dry-run: not calling the API or writing anything. IDs that would be resolved:")
        print(" ", version_ids[:20], "..." if len(version_ids) > 20 else "")
        return

    # Reuse resolve.py's own batching/caching/concurrency wholesale, rather than
    # re-implementing the two-hop version->model->creator lookup here -- build
    # the minimal fake "entries" shape it expects (a list of results, each with
    # meta.civitaiResources) with just the modelVersionIds that need resolving.
    fake_entries = [{"meta": {"civitaiResources": [{"modelVersionId": vid} for vid in version_ids]}}]

    load_resolver_cache()
    resolved = enrich_resources(fake_entries)
    save_resolver_cache()

    resolved_resources = resolved[0]["meta"]["civitaiResources"]

    updated = 0
    still_missing = []
    for res in resolved_resources:
        vid = res["modelVersionId"]
        name = res.get("name")
        version_name = res.get("versionName")
        creator = res.get("creatorUsername")
        if name is None and creator is None:
            still_missing.append(vid)
            continue
        conn.execute(
            """UPDATE resources SET
                 name = COALESCE(?, name),
                 versionName = COALESCE(?, versionName),
                 creatorUsername = COALESCE(?, creatorUsername)
               WHERE modelVersionId = ?""",
            (name, version_name, creator, vid),
        )
        updated += 1
    conn.commit()

    print(f"Updated {updated} resource row(s).")
    if still_missing:
        print(f"{len(still_missing)} modelVersionId(s) still unresolved (likely deleted/private models): "
              f"{still_missing[:20]}{'...' if len(still_missing) > 20 else ''}")


if __name__ == "__main__":
    main()
