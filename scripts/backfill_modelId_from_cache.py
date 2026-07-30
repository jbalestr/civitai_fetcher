"""
Backfill resources.modelId (and name/versionName/creatorUsername) purely
from the existing civitai_resolver_cache.json -- no API calls, no need to
re-run enrichment. Since that cache already holds every modelVersionId
resolved across all your past runs, this catches every checkpoint that's
already known, and only reports the (hopefully small) remainder that
genuinely still needs a live resolve.

Usage:
    uv run python scripts/backfill_modelid_from_cache.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH, RESOLVER_CACHE_PATH
from civitai_fetcher.db import get_connection


def main():
    if not os.path.exists(RESOLVER_CACHE_PATH):
        print(f"No cache found at {RESOLVER_CACHE_PATH} -- nothing to backfill from.")
        return

    with open(RESOLVER_CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)
    versions = cache.get("versions", {})
    models = cache.get("models", {})
    print(f"Loaded cache: {len(versions)} versions, {len(models)} models")

    conn = get_connection(DB_PATH)

    # every resources row still missing modelId
    rows = conn.execute(
        "SELECT modelVersionId FROM resources WHERE modelId IS NULL"
    ).fetchall()
    print(f"{len(rows)} resources row(s) missing modelId")

    resolved = 0
    still_missing = 0

    for r in rows:
        vid = r["modelVersionId"]
        info = versions.get(str(vid))
        if not info or info.get("modelId") is None:
            still_missing += 1
            continue

        model_id = info["modelId"]
        model_name = info.get("name")
        version_name = info.get("versionName")
        creator_username = models.get(str(model_id))

        conn.execute(
            """INSERT INTO models (modelId, modelName, modelUrl) VALUES (?, ?, NULL)
               ON CONFLICT(modelId) DO UPDATE SET
                 modelName = COALESCE(models.modelName, excluded.modelName)""",
            (model_id, model_name),
        )
        conn.execute(
            """UPDATE resources SET
                 modelId = ?,
                 name = COALESCE(name, ?),
                 versionName = COALESCE(versionName, ?),
                 creatorUsername = COALESCE(creatorUsername, ?)
               WHERE modelVersionId = ?""",
            (model_id, model_name, version_name, creator_username, vid),
        )
        resolved += 1

    conn.commit()
    print(f"Resolved {resolved} from cache, {still_missing} still unresolved "
          f"(not yet in {RESOLVER_CACHE_PATH} -- would need a live resolve to fill in).")


if __name__ == "__main__":
    main()