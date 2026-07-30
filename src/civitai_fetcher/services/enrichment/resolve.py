"""
Resolve civitaiResources modelVersionIds (checkpoint/LoRA references inside
meta) to human-readable names and creator usernames.

Civitai's own on-site Generator tool writes meta.civitaiResources with only
numeric modelVersionId — no name, no creator. Everything else (A1111, ComfyUI,
Forge uploads) usually has a "Model"/"resources" name already, so this only
needs to fill genuine gaps, not replace them.

Two-hop lookup, both cached via cache.KeyedDiskCache (persisted to disk
across runs) so repeat IDs — same popular checkpoints/LoRAs appearing
across thousands of images, run after run — are only fetched once, ever:
  modelVersionId -> GET /model-versions/{id} -> modelId, name
  modelId        -> GET /models/{id}         -> creator.username
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from ...core.config import BASE, RESOLVER_CACHE_PATH, MAX_WORKERS
from ...core.client import _get_with_retry
from .cache import KeyedDiskCache

_cache = KeyedDiskCache(RESOLVER_CACHE_PATH)

VERSIONS = "versions"  # modelVersionId (str) -> {"modelId": int, "name": str, "versionName": str}
MODELS = "models"      # modelId (str) -> creator username


def load_cache(path=RESOLVER_CACHE_PATH):
    """Load resolver cache from disk, if present. Call once before resolving."""
    global _cache
    if path != _cache.path:
        _cache = KeyedDiskCache(path)
    _cache.load()


def save_cache(path=RESOLVER_CACHE_PATH):
    """Persist resolver cache to disk. Call once after resolving."""
    _cache.save()


def _resolve_version(version_id):
    if _cache.contains(VERSIONS, version_id):
        return _cache.get(VERSIONS, version_id)
    try:
        r = _get_with_retry(f"{BASE}/model-versions/{version_id}", {})
        data = r.json()
        info = {
            "modelId": data.get("modelId"),
            "name": (data.get("model") or {}).get("name"),
            "versionName": data.get("name"),
        }
    except Exception as e:
        print(f"  [resolve] modelVersionId {version_id} failed: {e}")
        info = {"modelId": None, "name": None}
    _cache.set(VERSIONS, version_id, info)
    return info


def _resolve_creator(model_id):
    if model_id is None:
        return None
    if _cache.contains(MODELS, model_id):
        return _cache.get(MODELS, model_id)
    try:
        r = _get_with_retry(f"{BASE}/models/{model_id}", {})
        data = r.json()
        username = (data.get("creator") or {}).get("username")
    except Exception as e:
        print(f"  [resolve] modelId {model_id} failed: {e}")
        username = None
    _cache.set(MODELS, model_id, username)
    return username


def enrich_resources(results, max_workers=MAX_WORKERS):
    """
    Mutates each entry's meta.civitaiResources in place, adding "name",
    "versionName", and "creatorUsername" alongside the existing
    "modelVersionId"/"type"/"weight".

    Also resolves entry["modelVersionId"] (the checkpoint the image was
    fetched under, per the top-level API field) even when
    meta.civitaiResources is empty or entirely missing -- this is the
    common case for creator/username-scope fetches without --enrich-meta,
    where no generation meta ever came through, so the top-level field is
    the ONLY place a checkpoint is recorded at all. Previously this field
    was resolved nowhere, so store.py could only ever write a bare
    unresolved FK stub for it (modelVersionId with no modelId/name),
    leaving these images unable to roll up to their model. When resolved
    and not already represented in civitaiResources, it's injected as a
    synthetic checkpoint-type resource so store.py's existing
    civitaiResources-based insert logic picks it up with no separate code
    path needed.

    Two rounds, each concurrent: resolve versions -> modelIds first, since
    creator lookups need the modelId that only the version lookup provides.
    Cache hits (already-known IDs) are skipped without spawning a request.
    """
    all_version_ids = set()
    for entry in results:
        meta = entry.get("meta") or {}
        for res in meta.get("civitaiResources") or []:
            vid = res.get("modelVersionId")
            if vid:
                all_version_ids.add(vid)
        top_vid = entry.get("modelVersionId")
        if top_vid:
            all_version_ids.add(top_vid)

    to_fetch = [v for v in all_version_ids if not _cache.contains(VERSIONS, v)]
    print(f"Resolving {len(all_version_ids)} unique resource versions ({len(to_fetch)} not cached)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_resolve_version, vid): vid for vid in to_fetch}
        for future in as_completed(futures):
            future.result()

    model_ids = {
        _cache.get(VERSIONS, vid, {}).get("modelId")
        for vid in all_version_ids
        if _cache.get(VERSIONS, vid, {}).get("modelId") is not None
    }
    to_fetch_models = [m for m in model_ids if not _cache.contains(MODELS, m)]
    print(f"Resolving {len(model_ids)} unique creator models ({len(to_fetch_models)} not cached)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_resolve_creator, mid): mid for mid in to_fetch_models}
        for future in as_completed(futures):
            future.result()

    for entry in results:
        meta = entry.get("meta") or {}
        civitai_resources = meta.get("civitaiResources") or []
        for res in civitai_resources:
            vid = res.get("modelVersionId")
            if not vid:
                continue
            info = _cache.get(VERSIONS, vid, {})
            res["name"] = info.get("name")
            res["versionName"] = info.get("versionName")
            res["creatorUsername"] = _cache.get(MODELS, info.get("modelId"))

        top_vid = entry.get("modelVersionId")
        already_present = any(r.get("modelVersionId") == top_vid for r in civitai_resources)
        if top_vid and not already_present:
            info = _cache.get(VERSIONS, top_vid, {})
            civitai_resources.append({
                "type": "checkpoint",
                "modelVersionId": top_vid,
                "modelId": info.get("modelId"),
                "name": info.get("name"),
                "versionName": info.get("versionName"),
                "creatorUsername": _cache.get(MODELS, info.get("modelId")),
                "weight": 1,
            })
            meta["civitaiResources"] = civitai_resources
            entry["meta"] = meta

    return results