"""
Resolve civitaiResources modelVersionIds (checkpoint/LoRA references inside
meta) to human-readable names and creator usernames.

Civitai's own on-site Generator tool writes meta.civitaiResources with only
numeric modelVersionId — no name, no creator. Everything else (A1111, ComfyUI,
Forge uploads) usually has a "Model"/"resources" name already, so this only
needs to fill genuine gaps, not replace them.

Two-hop lookup, both cached (in-memory, and persisted to disk across runs via
load_cache()/save_cache()) so repeat IDs — same popular checkpoints/LoRAs
appearing across thousands of images, run after run — are only fetched once,
ever:
  modelVersionId -> GET /model-versions/{id} -> modelId, name
  modelId        -> GET /models/{id}         -> creator.username
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import BASE, RESOLVER_CACHE_PATH, MAX_WORKERS
from .client import _get_with_retry

_version_cache = {}  # modelVersionId (str) -> {"modelId": int, "name": str}
_model_cache = {}     # modelId (str) -> creator username


def load_cache(path=RESOLVER_CACHE_PATH):
    """Load resolver caches from disk, if present. Call once before resolving."""
    global _version_cache, _model_cache
    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
        _version_cache = data.get("versions", {})
        _model_cache = data.get("models", {})
        print(f"Loaded resolver cache: {len(_version_cache)} versions, {len(_model_cache)} models")
    except Exception as e:
        print(f"  [resolve] failed to load cache at {path}: {e}")


def save_cache(path=RESOLVER_CACHE_PATH):
    """Persist resolver caches to disk. Call once after resolving."""
    try:
        with open(path, "w") as f:
            json.dump({"versions": _version_cache, "models": _model_cache}, f, indent=2)
        print(f"Saved resolver cache: {len(_version_cache)} versions, {len(_model_cache)} models")
    except Exception as e:
        print(f"  [resolve] failed to save cache at {path}: {e}")


def _resolve_version(version_id):
    key = str(version_id)
    if key in _version_cache:
        return _version_cache[key]
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
    _version_cache[key] = info
    return info


def _resolve_creator(model_id):
    if model_id is None:
        return None
    key = str(model_id)
    if key in _model_cache:
        return _model_cache[key]
    try:
        r = _get_with_retry(f"{BASE}/models/{model_id}", {})
        data = r.json()
        username = (data.get("creator") or {}).get("username")
    except Exception as e:
        print(f"  [resolve] modelId {model_id} failed: {e}")
        username = None
    _model_cache[key] = username
    return username


def enrich_resources(results, max_workers=MAX_WORKERS):
    """
    Mutates each entry's meta.civitaiResources in place, adding "name",
    "versionName", and "creatorUsername" alongside the existing
    "modelVersionId"/"type"/"weight". Cheap no-op for entries that don't
    have civitaiResources.

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

    to_fetch = [v for v in all_version_ids if str(v) not in _version_cache]
    print(f"Resolving {len(all_version_ids)} unique resource versions ({len(to_fetch)} not cached)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_resolve_version, vid): vid for vid in to_fetch}
        for future in as_completed(futures):
            future.result()

    model_ids = {
        _version_cache[str(vid)]["modelId"]
        for vid in all_version_ids
        if _version_cache.get(str(vid), {}).get("modelId") is not None
    }
    to_fetch_models = [m for m in model_ids if str(m) not in _model_cache]
    print(f"Resolving {len(model_ids)} unique creator models ({len(to_fetch_models)} not cached)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_resolve_creator, mid): mid for mid in to_fetch_models}
        for future in as_completed(futures):
            future.result()

    for entry in results:
        meta = entry.get("meta") or {}
        for res in meta.get("civitaiResources") or []:
            vid = res.get("modelVersionId")
            if not vid:
                continue
            info = _version_cache.get(str(vid), {})
            res["name"] = info.get("name")
            res["versionName"] = info.get("versionName")
            res["creatorUsername"] = _model_cache.get(str(info.get("modelId")))

    return results