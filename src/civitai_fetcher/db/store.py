"""
Upsert + query helpers for the normalised schema (see migrations/001_initial.sql).

Takes entries in the same shape images_cli/creators_cli already produce (see
README "Output shape") -- no upstream fetch/enrichment code needs to change,
this just replaces the final json.dump() step.
"""
import csv
import json
import os
import zlib
from datetime import datetime, timezone

from ..core.config import SPAM_LOG_PATH, CREATOR_FILTERS_PATH, BLOCKED_LOG_PATH
from ..services.quality import is_checkpoint_spam
from ..services.creator_filters import load_creator_filters, is_blocked


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _media_type(entry):
    # existing entries may have "mediaType" (username-scope fetch) or
    # nothing at all (model-scope fetch predates that field) -- default to
    # "image", the overwhelming majority case.
    return entry.get("mediaType") or entry.get("type") or "image"


def _log_checkpoint_spam(entry, civitai_resources, log_path=SPAM_LOG_PATH):
    """
    Append-only record of every entry upsert_entries discards as
    checkpoint spam, so there's a durable trail of what got thrown away
    and why -- separate from the DB itself, since the whole point is these
    never get a row there. Writes a CSV header once if the file is new;
    every call after that just appends one line, so this is safe to call
    from concurrent/repeated runs without clobbering earlier entries.
    """
    checkpoint_count = sum(1 for r in civitai_resources if r.get("type") == "checkpoint")
    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["loggedAt", "imageId", "postId", "posterUsername", "checkpointCount", "mediaType"])
        writer.writerow([
            _now(), entry.get("imageId"), entry.get("postId"), entry.get("posterUsername"),
            checkpoint_count, _media_type(entry),
        ])
    print(f"  [spam] discarding imageId={entry.get('imageId')} postId={entry.get('postId')} "
          f"({checkpoint_count} checkpoints) -- logged to {log_path}")


def _log_blocked_creator(entry, log_path=BLOCKED_LOG_PATH):
    """
    Same append-only pattern as _log_checkpoint_spam, for images discarded
    because their posterUsername is in creator_filters.txt.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["loggedAt", "imageId", "postId", "posterUsername", "mediaType"])
        writer.writerow([
            _now(), entry.get("imageId"), entry.get("postId"), entry.get("posterUsername"), _media_type(entry),
        ])
    print(f"  [blocked] discarding imageId={entry.get('imageId')} postId={entry.get('postId')} "
          f"posterUsername={entry.get('posterUsername')} -- logged to {log_path}")


def upsert_entries(conn, entries, fetched_at=None):
    """
    Upsert a batch of image entries. Returns (new_count, updated_count, skipped_count).

    - models / resources: plain upsert, since they're small and repeat a lot.
    - images: upsert on imageId. first_seen_at is only set on first insert
      (uses SQLite's "excluded" trick via INSERT OR IGNORE + separate
      UPDATE, rather than ON CONFLICT overwriting it every time).
    - image_resources: replaced wholesale per image on each upsert (an
      image's resource list doesn't change once generated, but this is
      cheap and avoids stale rows if it's ever wrong the first time).
    - image_stats: always an INSERT (append-only history), never updated.
    - raw_meta: compressed with zlib, replaced if the entry is re-fetched
      (meta itself is immutable per image, so this is a no-op after the
      first insert in practice).
    - entries flagged by quality.is_checkpoint_spam (checkpoint-type
      resources beyond a threshold, video/audio exempted) are discarded
      outright, not just stripped -- an uploader gaming discoverability
      makes the whole blob untrustworthy, so nothing from that entry gets
      written at all.
    - entries whose posterUsername is in creator_filters.txt (see
      services.creator_filters) are likewise discarded outright.

    Returns (new_count, updated_count, skipped_spam_count, skipped_blocked_count).
    """
    fetched_at = fetched_at or _now()
    now = _now()
    new_count = 0
    updated_count = 0
    skipped_count = 0
    skipped_blocked_count = 0
    creator_filters = load_creator_filters(CREATOR_FILTERS_PATH)

    for entry in entries:
        if is_blocked(entry.get("posterUsername"), creator_filters):
            _log_blocked_creator(entry)
            skipped_blocked_count += 1
            continue

        meta = entry.get("meta") or {}
        civitai_resources = meta.get("civitaiResources") or []
        if is_checkpoint_spam(civitai_resources, media_type=_media_type(entry)):
            _log_checkpoint_spam(entry, civitai_resources)
            skipped_count += 1
            continue

        model_id = entry.get("modelId")
        if model_id is not None:
            conn.execute(
                """INSERT INTO models (modelId, modelName, modelUrl) VALUES (?, ?, ?)
                   ON CONFLICT(modelId) DO UPDATE SET modelName=excluded.modelName, modelUrl=excluded.modelUrl""",
                (model_id, entry.get("modelName"), entry.get("modelUrl")),
            )

        for res in civitai_resources:
            vid = res.get("modelVersionId")
            if vid is None:
                continue

            # civitaiResources entries carry the resource's own modelId
            # alongside modelVersionId -- capture it even on creator-scope
            # fetches (where entry.get("modelId") above is always None),
            # so a checkpoint used in a creator fetch can still be traced
            # back to its model via resources.modelId.
            res_model_id = res.get("modelId")
            if res_model_id is not None:
                conn.execute(
                    """INSERT INTO models (modelId, modelName, modelUrl) VALUES (?, ?, ?)
                       ON CONFLICT(modelId) DO UPDATE SET
                         modelName=COALESCE(models.modelName, excluded.modelName)""",
                    (res_model_id, res.get("modelName") or res.get("name"), None),
                )

            conn.execute(
                """INSERT INTO resources (modelVersionId, name, versionName, creatorUsername, resource_type, modelId)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(modelVersionId) DO UPDATE SET
                     name=COALESCE(excluded.name, resources.name),
                     versionName=COALESCE(excluded.versionName, resources.versionName),
                     creatorUsername=COALESCE(excluded.creatorUsername, resources.creatorUsername),
                     resource_type=COALESCE(excluded.resource_type, resources.resource_type),
                     modelId=COALESCE(excluded.modelId, resources.modelId)""",
                (vid, res.get("name") or res.get("modelName"), res.get("versionName"),
                 res.get("creatorUsername"), res.get("type"), res_model_id),
            )

        image_id = entry["imageId"]
        existed = conn.execute("SELECT 1 FROM images WHERE imageId = ?", (image_id,)).fetchone()

        # images.modelVersionId (the checkpoint version this image was fetched
        # under) has a foreign key into resources -- but that checkpoint
        # doesn't always show up inside meta.civitaiResources (e.g. meta is
        # null/empty entirely, exactly the case --enrich-meta exists to
        # handle). INSERT OR IGNORE a stub row first so the FK is always
        # satisfiable; the resolve pass above already fills in the real name
        # if/when this same modelVersionId also appears in civitaiResources.
        top_level_version_id = entry.get("modelVersionId")
        if top_level_version_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO resources (modelVersionId) VALUES (?)",
                (top_level_version_id,),
            )

        # An entry counts as enriched if resolve.enrich_resources() has already
        # run on it upstream (every attached resource has a resolved `name`) --
        # or trivially, if it has no resources to resolve in the first place.
        # This means replaying an already-enriched JSON file (e.g. from a run
        # with --enrich-meta) correctly marks rows enriched on the way in,
        # instead of every row starting unenriched regardless of history.
        # resolve.py writes "name"; generation.py's internal-endpoint enrichment
        # (--enrich-meta) writes "modelName" instead -- check both.
        is_enriched = not civitai_resources or all(r.get("name") or r.get("modelName") for r in civitai_resources)
        enriched_value = now if is_enriched else None

        conn.execute(
            """INSERT INTO images (
                 imageId, modelId, modelVersionId, imageUrl, posterUsername, postId, postUrl,
                 width, height, media_type, createdAt, nsfwLevel, prompt, negativePrompt,
                 sampler, steps, cfgScale, first_seen_at, enriched_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(imageId) DO UPDATE SET
                 modelId=excluded.modelId, modelVersionId=excluded.modelVersionId,
                 imageUrl=excluded.imageUrl, posterUsername=excluded.posterUsername,
                 postId=excluded.postId, postUrl=excluded.postUrl, width=excluded.width,
                 height=excluded.height, media_type=excluded.media_type,
                 nsfwLevel=excluded.nsfwLevel, prompt=excluded.prompt,
                 negativePrompt=excluded.negativePrompt, sampler=excluded.sampler,
                 steps=excluded.steps, cfgScale=excluded.cfgScale,
                 enriched_at=COALESCE(images.enriched_at, excluded.enriched_at)""",
            (
                image_id, model_id, entry.get("modelVersionId"), entry.get("imageUrl"),
                entry.get("posterUsername"), entry.get("postId"), entry.get("postUrl"),
                entry.get("width"), entry.get("height"), _media_type(entry),
                entry.get("createdAt"), entry.get("nsfwLevel"), meta.get("prompt"),
                meta.get("negativePrompt"), meta.get("sampler"), meta.get("steps"),
                meta.get("cfgScale"), now, enriched_value,
            ),
        )
        if existed:
            updated_count += 1
        else:
            new_count += 1

        conn.execute("DELETE FROM image_resources WHERE imageId = ?", (image_id,))
        for res in civitai_resources:
            vid = res.get("modelVersionId")
            if vid is None:
                continue
            conn.execute(
                """INSERT OR REPLACE INTO image_resources (imageId, modelVersionId, weight, resource_type)
                   VALUES (?, ?, ?, ?)""",
                (image_id, vid, res.get("weight"), res.get("type")),
            )

        stats = entry.get("stats") or {}
        conn.execute(
            """INSERT OR IGNORE INTO image_stats (
                 imageId, fetched_at, likeCount, heartCount, laughCount, cryCount,
                 commentCount, reactionScore
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                image_id, fetched_at, stats.get("likeCount"), stats.get("heartCount"),
                stats.get("laughCount"), stats.get("cryCount"), stats.get("commentCount"),
                entry.get("reactionScore"),
            ),
        )

        if meta:
            raw = json.dumps(meta).encode("utf-8")
            blob = zlib.compress(raw, level=9)
            conn.execute(
                """INSERT INTO raw_meta (imageId, meta_blob, raw_size, compressed_size)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(imageId) DO UPDATE SET
                     meta_blob=excluded.meta_blob, raw_size=excluded.raw_size,
                     compressed_size=excluded.compressed_size""",
                (image_id, blob, len(raw), len(blob)),
            )

    conn.commit()
    return new_count, updated_count, skipped_count, skipped_blocked_count


def get_raw_meta(conn, image_id):
    """Decompress and return the full original meta payload for one image, or None."""
    row = conn.execute("SELECT meta_blob FROM raw_meta WHERE imageId = ?", (image_id,)).fetchone()
    if not row or row["meta_blob"] is None:
        return None
    return json.loads(zlib.decompress(row["meta_blob"]))


def get_unenriched(conn, limit=None):
    """
    Images that have never been through resource/creator-name enrichment.
    Uses the partial index on enriched_at, so this stays cheap regardless
    of total table size.
    """
    sql = "SELECT * FROM images WHERE enriched_at IS NULL"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def mark_enriched(conn, image_ids, when=None):
    """Stamp a batch of images as enriched, after resolve.enrich_resources() has run on them."""
    when = when or _now()
    conn.executemany(
        "UPDATE images SET enriched_at = ? WHERE imageId = ?",
        [(when, iid) for iid in image_ids],
    )
    conn.commit()


def get_new_since(conn, since_iso):
    """Images first seen at or after `since_iso` (ISO timestamp) -- e.g. since your last run."""
    return conn.execute(
        "SELECT * FROM images WHERE first_seen_at >= ? ORDER BY first_seen_at", (since_iso,)
    ).fetchall()


def compression_stats(conn):
    """Quick visibility into how much raw_meta compression is actually saving."""
    row = conn.execute(
        "SELECT COUNT(*) AS n, SUM(raw_size) AS raw, SUM(compressed_size) AS compressed FROM raw_meta"
    ).fetchone()
    if not row["raw"]:
        return {"count": 0, "raw_bytes": 0, "compressed_bytes": 0, "ratio": None}
    return {
        "count": row["n"],
        "raw_bytes": row["raw"],
        "compressed_bytes": row["compressed"],
        "ratio": round(row["raw"] / row["compressed"], 2) if row["compressed"] else None,
    }