import time
from datetime import datetime, timedelta, timezone

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


def get_popular_models(limit=10, sort="Most Downloaded", types=None, max_lora_versions=None, only_ids=None):
    """
    only_ids: if given (list/set of model IDs), skips the popularity query
    entirely and fetches exactly these models by ID — e.g. the shortlist
    from probe.py's has_more_pages==True set. types/max_lora_versions are
    ignored when only_ids is used, since you've already curated the list.
    """
    if only_ids:
        items = []
        for mid in only_ids:
            r = _get_with_retry(f"{BASE}/models/{mid}", {})
            items.append(r.json())
        return items

    # Civitai caps /models at 100 per page. A raw incrementing "page" int is
    # unreliable on this endpoint (observed: it silently kept returning page 1's
    # results on every call, ballooning items with duplicates) — use the cursor
    # from metadata.nextCursor instead, same as the /images endpoint, plus a
    # dedupe pass as a safety net in case a page ever repeats.
    PAGE_SIZE = 100
    items = []
    seen_ids = set()
    cursor = None
    while len(items) < limit:
        params = {"limit": min(PAGE_SIZE, limit - len(items)), "sort": sort}
        if types:
            params["types"] = types
        if cursor:
            params["cursor"] = cursor
        r = _get_with_retry(f"{BASE}/models", params)
        payload = r.json()
        page_items = payload.get("items", [])
        if not page_items:
            break  # ran out of results before hitting the requested limit

        new_items = [m for m in page_items if m["id"] not in seen_ids]
        if not new_items:
            break  # every item on this "page" was already seen — stop rather than loop forever
        for m in new_items:
            seen_ids.add(m["id"])
        items.extend(new_items)

        next_cursor = payload.get("metadata", {}).get("nextCursor")
        if not next_cursor:
            break
        cursor = next_cursor

    if max_lora_versions is not None:
        kept = []
        for m in items:
            if m.get("type") == "LORA" and len(m.get("modelVersions") or []) > max_lora_versions:
                print(f"  Skipping {m['name']} ({m['id']}) — LORA with "
                      f"{len(m['modelVersions'])} versions (style/concept pack, likely)")
                continue
            kept.append(m)
        items = kept

    return items


def get_recent_images_with_meta(model_version_id, since, page_limit=100, max_pages=20, nsfw="X"):
    """
    Cursor-paginate through a single model VERSION's images (newest first),
    keeping only images with createdAt >= `since`. Stops as soon as a page's
    items fall entirely outside the window (since results are newest-first,
    everything after that point is guaranteed older too), when pagination
    runs out, or when max_pages is hit.

    Returns (collected, hit_page_cap) so callers can tell whether the window
    was fully captured or truncated by the page limit.

    nsfw="X": without an explicit nsfw param, Civitai's /images endpoint silently
    excludes NSFW items regardless of your actual access level (documented API
    quirk, civitai/civitai#1277). "X" is the highest content tier and reliably
    returns the full range, not just X-rated images specifically.
    """
    collected = []
    cursor = None
    pages_fetched = 0
    hit_page_cap = False

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

        in_window = [img for img in items if img.get("createdAt", "") >= since]
        collected.extend(in_window)

        pages_fetched += 1
        next_cursor = payload.get("metadata", {}).get("nextCursor")
        past_window = bool(items) and not in_window  # page had items, none survived the filter
        no_more_pages = not next_cursor
        hit_page_cap = pages_fetched >= max_pages

        if past_window or no_more_pages or hit_page_cap:
            break
        cursor = next_cursor

    return collected, hit_page_cap


def fetch_model_images(model, since, max_pages=20, nsfw="X", max_versions=None):
    """
    Fetch all meta'd images created since `since` (ISO timestamp) across a
    model's versions. If max_versions is set, only the newest N versions are
    queried (skips ancient/inactive versions that otherwise pad results).
    """
    model_id = model["id"]
    model_name = model["name"]
    versions = model.get("modelVersions") or []
    if max_versions:
        versions = versions[:max_versions]

    entries = []
    any_version_hit_cap = False

    for version in versions:
        version_id = version.get("id")
        if not version_id:
            continue
        try:
            images, hit_cap = get_recent_images_with_meta(
                version_id, since=since, max_pages=max_pages, nsfw=nsfw
            )
        except requests.HTTPError as e:
            print(f"  {model_name} ({model_id}) version {version_id} skipped, error: {e}")
            continue

        if hit_cap:
            any_version_hit_cap = True

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

    cap_note = " [hit max_pages cap on at least one version — window may be incomplete, consider raising max_pages]" \
        if any_version_hit_cap else ""
    print(f"  {model_name} ({model_id}): {len(entries)} images with meta since {since}{cap_note}")
    return entries


def probe_model_activity(model, since, page_limit=50, nsfw="X", download_rank=None):
    """
    Cheap single-page check: does this model's newest version have MORE than
    one page of recent meta'd images? Costs exactly one API call per model
    (only the latest version, only page 1) — orders of magnitude cheaper than
    a full fetch, useful for triaging a large candidate pool down to the
    models that are actually active right now before doing the expensive
    full pull on just those.

    Returns dict with modelId, modelName, type, page1_count, has_more_pages.
    """
    model_id = model["id"]
    model_name = model["name"]
    model_type = model.get("type")
    versions = model.get("modelVersions") or []
    if not versions:
        return {"modelId": model_id, "modelName": model_name, "type": model_type,
                "page1_count": 0, "has_more_pages": False, "download_rank": download_rank}

    latest_version_id = versions[0].get("id")
    params = {
        "modelVersionId": latest_version_id,
        "sort": "Newest",
        "limit": page_limit,
        "withMeta": "true",
    }
    if nsfw:
        params["nsfw"] = nsfw

    try:
        r = _get_with_retry(f"{BASE}/images", params)
        payload = r.json()
    except requests.HTTPError as e:
        print(f"  probe failed for {model_name} ({model_id}): {e}")
        return {"modelId": model_id, "modelName": model_name, "type": model_type,
                "page1_count": 0, "has_more_pages": False, "download_rank": download_rank}

    items = payload.get("items", [])
    in_window = [img for img in items if img.get("createdAt", "") >= since]
    has_more = bool(payload.get("metadata", {}).get("nextCursor")) and len(in_window) == len(items)

    return {
        "modelId": model_id,
        "modelName": model_name,
        "type": model_type,
        "page1_count": len(in_window),
        "has_more_pages": has_more,
        "download_rank": download_rank,
    }


def probe_candidates(candidate_count=100, since_days=30, max_workers=MAX_WORKERS, nsfw="X",
                      types=None, max_lora_versions=None, page_limit=50, only_ids=None):
    """
    Pull a large candidate pool by download rank, then cheaply probe each for
    current activity. Returns a list of probe dicts, unsorted — sort by
    page1_count/has_more_pages afterward to find the real activity threshold.

    page_limit: how deep the single-page check goes. Default 50 is cheap and
    good for the first broad pass (splitting active from dead). Once you've
    got a shortlist of active models (has_more_pages==True), re-run with
    only_ids=<shortlist> and a much bigger page_limit (e.g. 500) — that's
    still one call per model, but now the "big ones" actually separate from
    each other instead of all tying at the same cap.

    only_ids: skip the download-rank query, probe exactly this list instead
    (e.g. the shortlist from a first pass).

    NOTE: there used to be a stop_after_dead_streak early-stopping option here,
    on the assumption that dead models cluster together late in download-rank
    order. Measured against real data: Spearman correlation between download
    rank and current activity was only -0.175, and active models kept showing
    up scattered as far as rank 1000. That assumption doesn't hold, so it was
    removed — always probe the full requested candidate_count.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat().replace("+00:00", "Z")
    models = get_popular_models(limit=candidate_count, types=types, max_lora_versions=max_lora_versions,
                                 only_ids=only_ids)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(probe_model_activity, m, since, page_limit, nsfw, rank): m
            for rank, m in enumerate(models, start=1)
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def fetch_all(model_count=10, since_days=1, max_workers=MAX_WORKERS, max_pages=20, nsfw="X",
              max_versions=None, types=None, max_lora_versions=None, only_ids=None):
    """
    Fetch images+meta for the top `model_count` popular models, concurrently,
    limited to images created in the last `since_days` days.

    If only_ids is given, model_count/types/max_lora_versions are ignored —
    fetches exactly this shortlist (e.g. from probe.py's active-model set).
    """
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat().replace("+00:00", "Z")

    results = []
    models = get_popular_models(limit=model_count, types=types, max_lora_versions=max_lora_versions,
                                 only_ids=only_ids)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_model_images, m, since, max_pages, nsfw, max_versions): m
            for m in models
        }
        for future in as_completed(futures):
            results.extend(future.result())

    return results