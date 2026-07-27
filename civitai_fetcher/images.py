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
import threading
import requests
from collections import Counter
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import BASE, SITE, MAX_WORKERS
from .client import _get_with_retry, _log, get_stats, reset_stats, get_popular_models, _wait_all_with_heartbeat


def get_recent_images_with_meta(model_version_id, since, page_limit=100, max_pages=20, nsfw="X",
                                 require_meta=True, stop_event=None, counter=None, counter_lock=None,
                                 limit_total=None):
    """
    Cursor-paginate through a single model VERSION's images (newest first),
    keeping only images with createdAt >= `since`. Stops as soon as a page's
    items fall entirely outside the window (since results are newest-first,
    everything after that point is guaranteed older too), when pagination
    runs out, or when max_pages is hit.

    Returns (collected, hit_page_cap, request_failed) so callers can tell
    whether the window was fully captured, truncated by the page limit, or
    cut short by a persistent request failure (e.g. Civitai's own 503s under
    load) — the latter still returns everything collected up to that point
    rather than raising and losing it.

    nsfw="X": without an explicit nsfw param, Civitai's /images endpoint silently
    excludes NSFW items regardless of your actual access level (documented API
    quirk, civitai/civitai#1277). "X" is the highest content tier and reliably
    returns the full range, not just X-rated images specifically.

    No `withMeta` param is sent here, deliberately, regardless of require_meta.
    Combining `username`/other filters with `withMeta=true` has the same
    documented failure shape as other Civitai filter combos (silently
    returning zero items with a nextCursor that never terminates, rather
    than a clean "no results" — e.g. civitai/civitai#1848, #2134): a creator
    who visibly has meta'd images in the browser can still come back empty
    from this endpoint if withMeta is in the query. Since the plain response
    already includes `meta` when it exists, "has meta or not" is instead
    checked client-side per item after fetching (see fetch_model_images) —
    require_meta only controls whether items without it get dropped, not
    what's sent to the API.

    No `type`/media-type filter is sent here deliberately either — same
    silent-no-op pattern. The JSON fetch is cheap either way, so media-type
    filtering is also applied client-side afterwards (see fetch_model_images)
    — same principle as the createdAt window filter below, which was
    already client-side.

    stop_event/counter/counter_lock/limit_total: an optional shared early-stop
    mechanism across models running in parallel (see fetch_images_for_models).
    Checked between pages (not mid-page) — once the shared counter reaches
    limit_total, every model's pagination winds down on its next loop.
    """
    collected = []
    cursor = None
    pages_fetched = 0
    hit_page_cap = False
    request_failed = False

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        params = {
            "modelVersionId": model_version_id,
            "sort": "Newest",
            "limit": page_limit,
        }
        if nsfw:
            params["nsfw"] = nsfw
        if cursor:
            params["cursor"] = cursor

        t_page = time.monotonic()
        try:
            r = _get_with_retry(f"{BASE}/images", params)
        except requests.exceptions.RequestException as exc:
            _log(f"    page {pages_fetched + 1} (version {model_version_id}, page_size={page_limit}): "
                 f"FAILED after retries — stopping here, keeping {len(collected)} item(s) collected so "
                 f"far. ({exc})")
            request_failed = True
            break
        page_seconds = time.monotonic() - t_page
        payload = r.json()
        items = payload.get("items", [])
        _log(f"    page {pages_fetched + 1} (version {model_version_id}, page_size={page_limit}): "
             f"{page_seconds:.2f}s, {len(r.content)} bytes, {len(items)} item(s)")

        in_window = [img for img in items if img.get("createdAt", "") >= since]
        collected.extend(in_window)

        if counter is not None and in_window:
            with counter_lock:
                counter["n"] += len(in_window)
                if limit_total and counter["n"] >= limit_total and stop_event is not None:
                    stop_event.set()

        pages_fetched += 1
        next_cursor = payload.get("metadata", {}).get("nextCursor")
        past_window = bool(items) and not in_window  # page had items, none survived the filter
        no_more_pages = not next_cursor
        hit_page_cap = pages_fetched >= max_pages
        limit_hit = bool(stop_event is not None and stop_event.is_set())

        if past_window or no_more_pages or hit_page_cap or limit_hit:
            break
        cursor = next_cursor

    return collected, hit_page_cap, request_failed


def fetch_model_images(model, since, max_pages=20, nsfw="X", max_versions=None, media_type=None,
                        require_meta=True, stop_event=None, counter=None, counter_lock=None, limit_total=None,
                        page_size=100):
    """
    Fetch all images created since `since` (ISO timestamp) across a model's
    versions. If max_versions is set, only the newest N versions are queried
    (skips ancient/inactive versions that otherwise pad results).

    media_type: "image" | "video" | "audio" — filtered CLIENT-SIDE against
    each item's own "type" field after fetching (not sent to the API — see
    get_recent_images_with_meta for why).
    require_meta=True (default): only keep images that have generation
    metadata attached (this tool's original purpose). Set False to also
    keep bare images/videos with no meta (e.g. creators who strip prompts).
    stop_event/counter/counter_lock/limit_total: shared early-stop across
    models/versions — see fetch_images_for_models.
    """
    model_id = model["id"]
    model_name = model["name"]
    versions = model.get("modelVersions") or []
    if max_versions:
        versions = versions[:max_versions]

    entries = []
    any_version_hit_cap = False
    any_version_request_failed = False

    for version in versions:
        if stop_event is not None and stop_event.is_set():
            break
        version_id = version.get("id")
        if not version_id:
            continue
        try:
            images, hit_cap, request_failed = get_recent_images_with_meta(
                version_id, since=since, page_limit=page_size, max_pages=max_pages, nsfw=nsfw,
                require_meta=require_meta, stop_event=stop_event, counter=counter,
                counter_lock=counter_lock, limit_total=limit_total,
            )
        except Exception as e:
            print(f"  {model_name} ({model_id}) version {version_id} skipped, error: {e}", flush=True)
            continue

        if hit_cap:
            any_version_hit_cap = True
        if request_failed:
            any_version_request_failed = True

        for img in images:
            if require_meta and not img.get("meta"):
                continue
            if media_type and img.get("type") != media_type:
                continue
            entries.append({
                # --- static / simple fields first ---
                "modelId": model_id,
                "modelName": model_name,
                "modelVersionId": version_id,
                "modelUrl": f"{SITE}/models/{model_id}",
                "baseModel": version.get("baseModel"),
                "imageId": img["id"],
                "imageUrl": img["url"],
                "posterUsername": img.get("username"),
                "postId": img.get("postId"),
                "postUrl": f"{SITE}/posts/{img['postId']}" if img.get("postId") else None,
                "width": img.get("width"),
                "height": img.get("height"),
                "createdAt": img.get("createdAt"),
                "nsfwLevel": img.get("nsfwLevel") if img.get("nsfwLevel") != "None" else None,
                "mediaType": img.get("type"),
                "stats": img.get("stats"),
                "reactionScore": reaction_score(img.get("stats")),
                # --- dynamic generation metadata last ---
                "meta": img.get("meta"),
            })

    cap_note = " [hit max_pages cap on at least one version — window may be incomplete, consider raising max_pages]" \
        if any_version_hit_cap else ""
    fail_note = " [a request failed after retries on at least one version — window may be incomplete]" \
        if any_version_request_failed else ""
    meta_note = "images with meta" if require_meta else "images (meta not required)"
    print(f"  {model_name} ({model_id}): {len(entries)} {meta_note} since {since}{cap_note}{fail_note}", flush=True)
    return entries


def fetch_images_for_models(models, since, max_workers=MAX_WORKERS, max_pages=20, nsfw="X", max_versions=None,
                             media_type=None, require_meta=True, limit_total=None, page_size=100):
    """
    Fetch images for an already-discovered/ranked list of model dicts —
    e.g. the Week-ranked output of activity.probe_candidates(). This is the
    primitive to use when you want images for activity-ranked models rather
    than plain download-ranked ones; it does zero discovery of its own.

    media_type: "image" | "video" | "audio" — restricts to that media type,
    filtered client-side after fetching.
    require_meta=True (default): only keep images with generation metadata.
    Set False for creators who strip/hide their prompts (see
    get_recent_images_with_meta for the Civitai API quirk this works around).

    limit_total: if set, stops fetching (across all models/workers, via a
    shared threading.Event) once roughly this many in-window images have
    been collected. It's a soft cap, checked between pages rather than
    mid-page, so the actual result count may overshoot slightly — trim to
    an exact count client-side afterwards if you need one.
    """
    _log(f"Image fetch: {len(models)} model(s), max_workers={max_workers} "
         f"(max_pages={max_pages}, max_versions={max_versions}, media_type={media_type}, "
         f"require_meta={require_meta}, limit_total={limit_total})...")
    t0 = time.monotonic()
    reset_stats()
    results = []

    stop_event = threading.Event() if limit_total else None
    counter = {"n": 0} if limit_total else None
    counter_lock = threading.Lock() if limit_total else None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_model_images, m, since, max_pages, nsfw, max_versions, media_type,
                        require_meta, stop_event, counter, counter_lock, limit_total, page_size): m
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


def get_images_by_username(username, since, page_limit=100, max_pages=20, nsfw="X", require_meta=True,
                            stop_event=None, counter=None, counter_lock=None, limit_total=None):
    """
    Like get_recent_images_with_meta, but paginates /images filtered by
    UPLOADER username directly (Civitai's own `username` filter on /images)
    instead of by modelVersionId. This is everything the creator has ever
    posted to the gallery — including images made with someone else's
    model/checkpoint — not just the showcase images on models they own.

    No server-side `type` filter here either — see get_recent_images_with_meta
    for why; media-type filtering happens client-side in fetch_images_by_username.

    No `withMeta` param is sent here either, deliberately, regardless of
    require_meta — the combo with `username` is what produced the "0 items,
    nextCursor forever" symptom for creators who visibly do have meta'd
    images in the browser. `meta` is present in the plain response when it
    exists; require_meta only controls whether items without it get dropped,
    client-side, in fetch_images_by_username.
    """
    collected = []
    cursor = None
    pages_fetched = 0
    hit_page_cap = False
    request_failed = False

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        params = {
            "username": username,
            "sort": "Newest",
            "limit": page_limit,
        }
        if nsfw:
            params["nsfw"] = nsfw
        if cursor:
            params["cursor"] = cursor

        t_page = time.monotonic()
        try:
            r = _get_with_retry(f"{BASE}/images", params)
        except requests.exceptions.RequestException as exc:
            _log(f"    page {pages_fetched + 1} (username={username}, page_size={page_limit}): "
                 f"FAILED after retries — stopping here, keeping {len(collected)} item(s) collected so "
                 f"far. ({exc})")
            request_failed = True
            break
        page_seconds = time.monotonic() - t_page
        payload = r.json()
        items = payload.get("items", [])
        _log(f"    page {pages_fetched + 1} (username={username}, page_size={page_limit}): "
             f"{page_seconds:.2f}s, {len(r.content)} bytes, {len(items)} item(s)")

        in_window = [img for img in items if img.get("createdAt", "") >= since]
        collected.extend(in_window)

        if counter is not None and in_window:
            with counter_lock:
                counter["n"] += len(in_window)
                if limit_total and counter["n"] >= limit_total and stop_event is not None:
                    stop_event.set()

        pages_fetched += 1
        next_cursor = payload.get("metadata", {}).get("nextCursor")
        past_window = bool(items) and not in_window
        no_more_pages = not next_cursor
        hit_page_cap = pages_fetched >= max_pages
        limit_hit = bool(stop_event is not None and stop_event.is_set())

        if past_window or no_more_pages or hit_page_cap or limit_hit:
            break
        cursor = next_cursor

    return collected, hit_page_cap, request_failed


def fetch_images_by_username(username, since, max_pages=20, nsfw="X", media_type=None, require_meta=True,
                              limit_total=None, page_size=100):
    """
    Everything a creator has POSTED (uploader username filter), as opposed
    to fetch_images_for_models() which only covers images attached to
    models that creator themselves published. Single paginated stream, no
    per-model fan-out — there's only one username to page through.

    Since there's no owning model here, "modelId"/"modelName" on each entry
    are left as None; the model/LoRA actually used to generate the image
    (if any) shows up resolved inside meta.civitaiResources instead (see
    resolve.enrich_resources).

    media_type: "image" | "video" | "audio" — filtered CLIENT-SIDE against
    each item's own "type" field after fetching (not sent to the API — see
    get_images_by_username for why).
    require_meta=True (default): only keep images with generation metadata.
    Set False for creators who strip/hide their prompts — Civitai's API
    otherwise returns a nextCursor forever with zero items per page for
    those usernames rather than a clean "no results" (see
    get_images_by_username).
    """
    _log(f"Image fetch (uploads scope): username={username} "
         f"(page_size={page_size}, max_pages={max_pages}, media_type={media_type}, "
         f"require_meta={require_meta}, limit_total={limit_total})...")
    t0 = time.monotonic()
    reset_stats()

    stop_event = threading.Event() if limit_total else None
    counter = {"n": 0} if limit_total else None
    counter_lock = threading.Lock() if limit_total else None

    images, hit_cap, request_failed = get_images_by_username(
        username, since=since, page_limit=page_size, max_pages=max_pages, nsfw=nsfw, require_meta=require_meta,
        stop_event=stop_event, counter=counter, counter_lock=counter_lock, limit_total=limit_total,
    )

    entries = []
    for img in images:
        if require_meta and not img.get("meta"):
            continue
        if media_type and img.get("type") != media_type:
            continue
        meta = img.get("meta") or {}
        civitai_resources = meta.get("civitaiResources") or []
        checkpoint = next((r for r in civitai_resources if r.get("type") == "checkpoint"), None)
        model_version_id = checkpoint.get("modelVersionId") if checkpoint else (img.get("modelVersionIds") or [None])[0]
        entries.append({
            "modelId": None,
            "modelName": None,
            "modelVersionId": model_version_id,
            "modelUrl": None,
            "baseModel": img.get("baseModel"),
            "imageId": img["id"],
            "imageUrl": img["url"],
            "posterUsername": img.get("username"),
            "postId": img.get("postId"),
            "postUrl": f"{SITE}/posts/{img['postId']}" if img.get("postId") else None,
            "width": img.get("width"),
            "height": img.get("height"),
            "createdAt": img.get("createdAt"),
            "nsfwLevel": img.get("nsfwLevel") if img.get("nsfwLevel") != "None" else None,
            "mediaType": img.get("type"),
            "stats": img.get("stats"),
            "reactionScore": reaction_score(img.get("stats")),
            "meta": img.get("meta"),
        })

    cap_note = " [hit max_pages cap — window may be incomplete, consider raising max_pages]" if hit_cap else ""
    fail_note = " [a request failed after retries — window may be incomplete, results so far kept]" \
        if request_failed else ""
    elapsed = time.monotonic() - t0
    s = get_stats()
    _log(f"Image fetch done: {len(entries)} image(s) with meta for '{username}' in {elapsed:.2f}s{cap_note}{fail_note}")
    _log(f"  requests={s['requests']} ok={s['ok']} rate_limited_429={s['rate_limited']} "
         f"exceptions={s['exceptions']} gave_up={s['gave_up']}")
    return entries



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