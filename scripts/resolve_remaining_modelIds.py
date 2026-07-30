"""
Resolve resources.modelId for rows the cache-only backfill couldn't cover
-- hits the Civitai API for whatever's still unresolved, same as a normal
enrichment pass, then writes the results straight to the DB.

Run backfill_modelid_from_cache.py FIRST -- this only needs to touch the
network for the remainder that script reported as "still unresolved".

Usage:
    uv run python scripts/resolve_remaining_modelids.py
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection
from civitai_fetcher.services.enrichment.resolve import (
    _cache, _resolve_version, _resolve_creator, VERSIONS, MODELS, load_cache, save_cache,
)


def main():
    load_cache()
    conn = get_connection(DB_PATH)

    rows = conn.execute("SELECT modelVersionId FROM resources WHERE modelId IS NULL").fetchall()
    version_ids = [r["modelVersionId"] for r in rows]
    print(f"{len(version_ids)} resources row(s) still missing modelId -- resolving via API...")

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_resolve_version, vid): vid for vid in version_ids}
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 200 == 0:
                print(f"  {i}/{len(version_ids)} versions resolved...")

    model_ids = {
        _cache.get(VERSIONS, vid, {}).get("modelId")
        for vid in version_ids
        if _cache.get(VERSIONS, vid, {}).get("modelId") is not None
    }
    print(f"Resolving {len(model_ids)} unique creator model(s)...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_resolve_creator, mid): mid for mid in model_ids}
        for future in as_completed(futures):
            future.result()

    save_cache()

    resolved = 0
    for vid in version_ids:
        info = _cache.get(VERSIONS, vid, {})
        model_id = info.get("modelId")
        if model_id is None:
            continue
        model_name = info.get("name")
        version_name = info.get("versionName")
        creator_username = _cache.get(MODELS, model_id)

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
    print(f"Resolved {resolved}/{len(version_ids)} (the rest genuinely failed/no data at the API -- "
          f"likely deleted or private models).")


if __name__ == "__main__":
    main()