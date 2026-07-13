"""
Image-level fetching: pull all images (with generation meta) for a given
set of models, then rank them by community reactions.

This module only knows how to fetch and score *images*. It does not know
how to discover or rank *models* — that's activity.py's job. Deliberately
imports only from client.py, never from activity.py, so nothing done in
here can break probe.py's model-ranking pipeline.

This restores the fetch_all/fetch_model_images/get_recent_images_with_meta
lineage that existed before commit 1df2185 removed it (which is also why
cli.py has been broken since — it still imports fetch_all from a module
that no longer defines it).

Reaction ranking is always done client-side, on the fetched `stats` field,
after fetching everything sorted "Newest". Civitai's own sort=Most Reactions
on /images doesn't reliably combine with withMeta + cursor pagination the
way sort=Newest does — so rather than trust that combination, we fetch the
full window newest-first (same as the old code) and rank reactions ourselves.
"""
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import BASE, SITE, MAX_WORKERS
from .client import _get_with_retry, _log, get_stats, reset_stats, get_popular_models, _wait_all_with_heartbeat


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
        except Exception as e:
            print(f"  {model_name} ({model_id}) version {version_id} skipped, error: {e}", flush=True)
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
                "reactionScore": reaction_score(img.get("stats")),
                # --- dynamic generation metadata last ---
                "meta": img.get("meta"),
            })

    cap_note = " [hit max_pages cap on at least one version — window may be incomplete, consider raising max_pages]" \
        if any_version_hit_cap else ""
    print(f"  {model_name} ({model_id}): {len(entries)} images with meta since {since}{cap_note}", flush=True)
    return entries


def fetch_images_for_models(models, since, max_workers=MAX_WORKERS, max_pages=20, nsfw="X", max_versions=None):
    """
    Fetch images for an already-discovered/ranked list of model dicts —
    e.g. the Week-ranked output of activity.probe_candidates(). This is the
    primitive to use when you want images for activity-ranked models rather
    than plain download-ranked ones; it does zero discovery of its own.
    """
    _log(f"Image fetch: {len(models)} model(s), max_workers={max_workers} "
         f"(max_pages={max_pages}, max_versions={max_versions})...")
    t0 = time.monotonic()
    reset_stats()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_model_images, m, since, max_pages, nsfw, max_versions): m
            for m in models
        }
        done = _wait_all_with_heartbeat(futures.keys(), "Image fetch")
        for future in done:
            results.extend(future.result())
    elapsed = time.monotonic() - t0
    s = get_stats()
    _log(f"Image fetch done: {len(results)} image(s) across {len(models)} model(s) in {elapsed:.2f}s")
    _log(f"  requests={s['requests']} ok={s['ok']} rate_limited_429={s['rate_limited']} "
         f"exceptions={s['exceptions']} gave_up={s['gave_up']}")
    return results


def fetch_all(model_count=10, since_days=1, period="Month", max_workers=MAX_WORKERS, max_pages=20, nsfw="X",
              max_versions=None, types=None, max_lora_versions=None, only_ids=None):
    """
    Standalone convenience: discover models by download rank (like the
    original cli.py did), then fetch their images.

    For activity-ranked discovery instead (recommended for finding what's
    actually active right now, not just historically downloaded — see
    README "Discoveries"), use activity.probe_candidates() to build a
    model list yourself and pass it to fetch_images_for_models() directly.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat().replace("+00:00", "Z")
    models = get_popular_models(limit=model_count, period=period, types=types,
                                 max_lora_versions=max_lora_versions, only_ids=only_ids)
    return fetch_images_for_models(models, since, max_workers=max_workers, max_pages=max_pages,
                                    nsfw=nsfw, max_versions=max_versions)


def reaction_score(stats):
    """
    Combine an image's stats dict into one sortable 'reactions' number.
    Sums whichever reaction-count fields are present rather than assuming
    a fixed schema, since Civitai's /images stats object composition has
    varied slightly across API responses.
    """
    if not stats:
        return 0
    keys = ("likeCount", "heartCount", "laughCount", "cryCount", "commentCount")
    return sum(stats.get(k, 0) or 0 for k in keys)


def sort_by_reactions(entries, top_n=None):
    """
    Sort fetched image entries by combined reaction score, descending.
    Entries already carry a precomputed 'reactionScore' field from
    fetch_model_images; this just orders (and optionally truncates) by it.
    """
    ranked = sorted(entries, key=lambda e: e.get("reactionScore", 0), reverse=True)
    return ranked[:top_n] if top_n else ranked


def sort_by_reactions_per_model(entries, top_n_per_model=10):
    """
    Like sort_by_reactions, but caps the top N PER MODEL instead of one
    global top N — a flat global cutoff always gets crowded out by whichever
    model has the single highest reaction counts (Kreamania style outliers),
    leaving other models with zero representation in the output. Capping
    per-model instead guarantees every fetched model gets a fair, comparable
    slice, which is the point if you're comparing model/checkpoint types
    rather than just chasing the single best image overall.

    Returns entries grouped by model (each model's own images sorted by
    reaction score, descending), models ordered by their own top image's
    reaction score — so the highest-reaction model still appears first, but
    every model gets its full top_n_per_model regardless of how it compares
    to others.
    """
    by_model = {}
    for e in entries:
        by_model.setdefault(e.get("modelId"), []).append(e)

    per_model_top = {
        model_id: sorted(model_entries, key=lambda e: e.get("reactionScore", 0), reverse=True)[:top_n_per_model]
        for model_id, model_entries in by_model.items()
    }
    # Order models by their own best image, so the overall best model still leads,
    # but every model's slice stays intact underneath it.
    ordered_model_ids = sorted(
        per_model_top, key=lambda mid: per_model_top[mid][0].get("reactionScore", 0) if per_model_top[mid] else 0,
        reverse=True,
    )
    result = []
    for mid in ordered_model_ids:
        result.extend(per_model_top[mid])
    return result


def count_resource_usage(entries):
    """
    Tally civitaiResources usage across a set of fetched images — two
    different questions, kept as two separate counters rather than one:

      lora_counts:  how often each LoRA appears overall, regardless of which
                    checkpoint it was paired with. Answers "what's the most
                    popular LoRA right now" across the whole fetched set.

      combo_counts: how often each (checkpoint, LoRA) PAIR appears together.
                    Answers "what's most used specifically with THIS
                    checkpoint" — a LoRA can rank high overall but be tied to
                    one checkpoint, or split its usage across several; this
                    is what tells them apart.

    Uses resolved names (from resolve.enrich_resources, i.e. --resolve-resources)
    when available; falls back to the raw modelVersionId as the label if the
    entry was never enriched, so this still works either way — just less
    readable without resolution.
    """
    lora_counts = Counter()
    combo_counts = Counter()
    for entry in entries:
        checkpoint_name = entry.get("modelName", "Unknown")
        meta = entry.get("meta") or {}
        for res in meta.get("civitaiResources") or []:
            if res.get("type") != "lora":
                continue
            lora_label = res.get("name") or f"modelVersionId:{res.get('modelVersionId')}"
            lora_counts[lora_label] += 1
            combo_counts[(checkpoint_name, lora_label)] += 1
    return lora_counts, combo_counts


def count_bare_checkpoint_usage(entries):
    """
    A third, different question again from count_resource_usage: not "which
    LoRA is popular" but "does this checkpoint even get used WITH a LoRA at
    all, or mostly raw?" Some checkpoints may only look good/produce their
    popular images once paired with a specific LoRA (near-zero bare usage);
    others may be popular entirely on their own (mostly bare). Neither
    lora_counts nor combo_counts answers this — both are silent on
    checkpoints that show up with nothing attached.

    Splits into THREE buckets per checkpoint, not two, because "no LoRA" is
    ambiguous on its own:
      with_lora:      at least one lora/embedding resource attached — genuinely
                       used together.
      bare_resources: civitaiResources is present and non-empty, but contains
                       ONLY the checkpoint (no lora/embedding) — genuinely used
                       on its own.
      no_resources:   civitaiResources is missing or empty entirely. This is
                       NOT the same claim as bare_resources — it usually means
                       the image wasn't generated through Civitai's own on-site
                       generator (e.g. uploaded from A1111/ComfyUI, which don't
                       populate this field), so we genuinely can't tell whether
                       a LoRA was used. Lumping this in with bare_resources
                       would overstate how often a checkpoint is used "raw."

    Returns (with_lora, bare_resources, no_resources, total) — four Counters,
    all keyed by checkpoint name.
    """
    with_lora = Counter()
    bare_resources = Counter()
    no_resources = Counter()
    total = Counter()
    for entry in entries:
        checkpoint_name = entry.get("modelName", "Unknown")
        meta = entry.get("meta") or {}
        resources = meta.get("civitaiResources") or []
        total[checkpoint_name] += 1
        if not resources:
            no_resources[checkpoint_name] += 1
            continue
        has_addon = any(r.get("type") in ("lora", "embedding", "textualinversion") for r in resources)
        if has_addon:
            with_lora[checkpoint_name] += 1
        else:
            bare_resources[checkpoint_name] += 1
    return with_lora, bare_resources, no_resources, total