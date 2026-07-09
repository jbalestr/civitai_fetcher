import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import BASE, SITE, HEADERS, MAX_WORKERS

MAX_RETRIES = 5


def _get_with_retry(url, params):
    """GET with retry/backoff on 429 (rate limit) and 503 (transient)."""
    retries = 0
    while True:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 5))
            time.sleep(wait)
            continue
        if r.status_code == 503:
            retries += 1
            if retries > MAX_RETRIES:
                r.raise_for_status()
            time.sleep(min(2 ** retries, 30))
            continue
        r.raise_for_status()
        return r


def get_popular_models(limit=10, sort="Most Downloaded"):
    r = _get_with_retry(f"{BASE}/models", {"limit": limit, "sort": sort})
    return r.json()["items"]


def get_recent_images_with_meta(model_version_id, target_count, page_limit=100, max_pages=5, nsfw="X"):
    """
    Cursor-paginate through a single model VERSION's images (newest first) until
    we've collected `target_count` images that carry metadata, or run out of
    pages/max_pages.

    Querying by modelVersionId (not modelId) is what actually surfaces meta'd
    images for some checkpoints — filtering by the whole model can come back
    empty even after deep pagination, seemingly because images get associated
    with a specific version rather than the parent model.

    nsfw="X": without an explicit nsfw param, Civitai's /images endpoint silently
    excludes NSFW items regardless of your actual access level (documented API
    quirk, civitai/civitai#1277). "X" is the highest content tier and reliably
    returns the full range, not just X-rated images specifically.
    """
    collected = []
    cursor = None
    pages_fetched = 0

    while True:
        params = {
            "modelVersionId": model_version_id,
            "sort": "Newest",
            "limit": page_limit,
            "withMeta": "true",
        }
        if nsfw:
            params["nsfw"] = nsfw
        if cursor:
            params["cursor"] = cursor

        r = _get_with_retry(f"{BASE}/images", params)
        payload = r.json()
        items = payload.get("items", [])
        collected.extend(items)
        pages_fetched += 1

        next_cursor = payload.get("metadata", {}).get("nextCursor")
        enough = len(collected) >= target_count
        no_more_pages = not next_cursor
        hit_page_cap = pages_fetched >= max_pages

        if enough or no_more_pages or hit_page_cap:
            break
        cursor = next_cursor

    return collected[:target_count] if len(collected) > target_count else collected


def fetch_model_images(model, images_per_model, max_pages=5, nsfw="X"):
    model_id = model["id"]
    model_name = model["name"]
    versions = model.get("modelVersions") or []
    entries = []

    for version in versions:
        version_id = version.get("id")
        if not version_id or len(entries) >= images_per_model:
            continue

        remaining = images_per_model - len(entries)
        try:
            images = get_recent_images_with_meta(
                version_id, target_count=remaining, max_pages=max_pages, nsfw=nsfw
            )
        except requests.HTTPError as e:
            print(f"  {model_name} ({model_id}) version {version_id} skipped, error: {e}")
            continue

        for img in images:
            # defensive check, withMeta should already exclude these server-side
            if not img.get("meta"):
                continue
            entries.append({
                # --- static / simple fields first ---
                "modelId": model_id,
                "modelName": model_name,
                "modelVersionId": version_id,
                "modelUrl": f"{SITE}/models/{model_id}",
                "imageId": img["id"],
                "imageUrl": img["url"],
                "posterUsername": img.get("username"),
                "postId": img.get("postId"),
                "postUrl": f"{SITE}/posts/{img['postId']}" if img.get("postId") else None,
                "width": img.get("width"),
                "height": img.get("height"),
                "createdAt": img.get("createdAt"),
                "nsfwLevel": img.get("nsfwLevel"),
                "stats": img.get("stats"),
                # --- dynamic generation metadata last ---
                "meta": img.get("meta"),
            })

    print(f"  {model_name} ({model_id}): {len(entries)} images with meta")
    return entries


def fetch_all(model_count=10, images_per_model=20, max_workers=MAX_WORKERS, max_pages=5, nsfw="X"):
    """Fetch images+meta for the top `model_count` popular models, concurrently."""
    results = []
    models = get_popular_models(limit=model_count)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_model_images, m, images_per_model, max_pages, nsfw): m
            for m in models
        }
        for future in as_completed(futures):
            results.extend(future.result())

    return results
