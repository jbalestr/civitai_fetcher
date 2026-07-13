"""
Model-level activity ranking (Tier 1/2 page-cap probe + Tier 3b day-bucketed
velocity). This is the logic behind probe.py.

Deliberately imports only from client.py, never from images.py. That's the
isolation guarantee: nothing done for image-fetching can break this module,
because this module doesn't know images.py exists.
"""
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import BASE, MAX_WORKERS, PHASE_TIMEOUT_SECONDS, MODEL_TIMEOUT_SECONDS, BACKOFF_CAP
from .client import _get_with_retry, _log, get_stats, reset_stats, get_popular_models, _wait_with_heartbeat


def probe_model_activity(model, since, page_limit, nsfw, rank, deep_probe_limit):
    """
    Analyzes an individual model's image velocity by tracing recent image uploads.
    """
    t_start = time.monotonic()
    model_id = model.get("id")
    model_name = model.get("name", "Unknown")
    model_type = model.get("type", "Unknown")

    # baseModel (e.g. "Illustrious", "Pony", "Flux.1 D", "SDXL 1.0") lives on the
    # model *version*, not the model itself — take it from the newest/first version
    # returned by the API (modelVersions is newest-first).
    model_versions = model.get("modelVersions") or []
    base_model = model_versions[0].get("baseModel") if model_versions else None

    result = {
        "modelId": model_id,
        "modelName": model_name,
        "type": model_type,
        "baseModel": base_model,
        "download_rank": rank,
        "page1_count": 0,
        "total_probe_count": 0,
        "has_more_pages": False,
        "probe_status": "ok"
    }

    if not model_id:
        result["probe_status"] = "no_model_id"
        result["probe_seconds"] = round(time.monotonic() - t_start, 3)
        return result

    img_url = f"{BASE}/images"
    params = {
        "modelId": model_id,
        "limit": page_limit,
        "nsfw": nsfw,
        "sort": "Newest"
    }
    response_bytes = 0

    try:
        r = _get_with_retry(img_url, params)
        response_bytes += len(r.content)
        payload = r.json()
        images = payload.get("items", [])

        # Count items matching our timeframe condition. Images are sorted Newest-first
        # (sort="Newest"), so once one falls outside the window every remaining item on
        # this page is guaranteed to be older too — stop scanning immediately.
        recent_images = []
        for img in images:
            if img.get("createdAt", "") >= since:
                recent_images.append(img)
            else:
                break
        result["page1_count"] = len(recent_images)
        result["total_probe_count"] = len(recent_images)

        next_cursor = payload.get("metadata", {}).get("nextCursor")

        # Tier 2 Adaptive Deep Scan if the first page was completely saturated
        if len(recent_images) == page_limit and next_cursor:
            result["has_more_pages"] = True

            # If a strict cap is set and we've already hit it, stop right away
            if deep_probe_limit and result["total_probe_count"] >= deep_probe_limit:
                result["response_bytes"] = response_bytes
                result["probe_seconds"] = round(time.monotonic() - t_start, 3)
                return result

            cursor = next_cursor
            while cursor:
                if time.monotonic() - t_start > MODEL_TIMEOUT_SECONDS:
                    result["probe_status"] = "model_timeout"
                    break
                params["cursor"] = cursor
                r_deep = _get_with_retry(img_url, params)
                response_bytes += len(r_deep.content)
                deep_payload = r_deep.json()
                deep_images = deep_payload.get("items", [])

                if not deep_images:
                    result["has_more_pages"] = False
                    break

                deep_recent = []
                for img in deep_images:
                    if img.get("createdAt", "") >= since:
                        deep_recent.append(img)
                    else:
                        break
                result["total_probe_count"] += len(deep_recent)

                # Check if we broke through the trend window boundary
                if len(deep_recent) < len(deep_images):
                    result["has_more_pages"] = False
                    break

                if deep_probe_limit and result["total_probe_count"] >= deep_probe_limit:
                    break

                cursor = deep_payload.get("metadata", {}).get("nextCursor")
                if not cursor:
                    result["has_more_pages"] = False

    except Exception as e:
        result["probe_status"] = "error"
        print(f"Error probing model {model_id} ({model_name}): {e}", flush=True)

    result["response_bytes"] = response_bytes
    result["probe_seconds"] = round(time.monotonic() - t_start, 3)
    return result


def probe_candidates(candidate_count=100, since_days=30, period="Month", max_workers=MAX_WORKERS, nsfw="X",
                     types=None, max_lora_versions=None, page_limit=50, deep_probe_limit=None, only_ids=None):
    """
    Passes the period flag down to the model finder.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat().replace("+00:00", "Z")
    models = get_popular_models(limit=candidate_count, period=period, types=types,
                                max_lora_versions=max_lora_versions, only_ids=only_ids)

    _log(f"Tier 1/2 probe: {len(models)} model(s), max_workers={max_workers} "
         f"(page_limit={page_limit}, deep_probe_limit={deep_probe_limit})...")
    t0 = time.monotonic()
    reset_stats()
    results = []
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        pool.submit(probe_model_activity, m, since, page_limit, nsfw, rank, deep_probe_limit): m
        for rank, m in enumerate(models, start=1)
    }
    done, not_done = _wait_with_heartbeat(futures.keys(), PHASE_TIMEOUT_SECONDS, "Tier 1/2 probe")
    for future in done:
        results.append(future.result())
    if not_done:
        _log(f"⚠️  Tier 1/2 phase timeout ({PHASE_TIMEOUT_SECONDS}s) hit with {len(not_done)} model(s) still "
             f"stuck in-flight — abandoning them rather than blocking the run. Their worker threads may keep "
             f"running in the background until their own request timeout/retries finally give up.")
        for future in not_done:
            m = futures[future]
            results.append({
                "modelId": m.get("id"), "modelName": m.get("name", "Unknown"), "type": m.get("type", "Unknown"),
                "baseModel": None, "download_rank": None, "page1_count": 0, "total_probe_count": 0,
                "has_more_pages": False, "probe_status": "phase_timeout",
                "probe_seconds": PHASE_TIMEOUT_SECONDS, "response_bytes": 0,
            })
    pool.shutdown(wait=False, cancel_futures=True)
    elapsed = time.monotonic() - t0
    s = get_stats()
    per_model = elapsed / len(models) if models else 0
    print(flush=True)  # end the '.'/'o'/'X' progress line
    _log(f"Tier 1/2 probe done: {len(models)} model(s) in {elapsed:.2f}s "
         f"({per_model*1000:.0f}ms/model wall-clock avg, {max_workers} concurrent)")
    _log(f"  requests={s['requests']} ok={s['ok']} rate_limited_429={s['rate_limited']} "
         f"exceptions={s['exceptions']} gave_up={s['gave_up']}")
    _log(f"  time in network calls: {s['request_seconds']:.2f}s  |  time asleep in backoff: {s['backoff_seconds']:.2f}s "
         f"(backoff time is self-imposed waiting, not network latency — high backoff_seconds means you're being rate-limited)")
    avg_kb = (s["response_bytes"] / s["ok"] / 1024) if s["ok"] else 0
    _log(f"  total response size: {s['response_bytes']/1_048_576:.1f} MB over {s['ok']} responses ({avg_kb:.1f} KB/response avg)")
    if s["rate_limited"] or s["exceptions"]:
        print(f"⚠️  Hit {s['rate_limited']} rate-limit response(s) and {s['exceptions']} network exception(s) "
              f"during Tier 1/2 — this, not raw network speed, is the likely reason a 1000-model run takes minutes "
              f"instead of seconds. Each 429 costs one thread up to {BACKOFF_CAP}s of backoff sleep.", flush=True)
    slowest = sorted(results, key=lambda r: r.get("probe_seconds", 0), reverse=True)[:10]
    if slowest and slowest[0].get("probe_seconds", 0) > 1.0:
        print("  Slowest individual model probes (name, seconds, has_more_pages):", flush=True)
        for r in slowest:
            print(f"    {r['modelName'][:40]:40s} {r.get('probe_seconds', 0):6.2f}s  "
                  f"has_more_pages={r.get('has_more_pages')}", flush=True)
    heaviest = sorted(results, key=lambda r: r.get("response_bytes", 0), reverse=True)[:10]
    avg_bytes = (s["response_bytes"] / len(results)) if results else 0
    if heaviest and avg_bytes and heaviest[0].get("response_bytes", 0) > 3 * avg_bytes:
        print(f"  Heaviest individual payloads (name, KB, vs {avg_bytes/1024:.1f}KB avg/model):", flush=True)
        for r in heaviest:
            print(f"    {r['modelName'][:40]:40s} {r.get('response_bytes', 0)/1024:8.1f} KB", flush=True)
    return results, models, since


def probe_recent_velocity(model, page_limit, nsfw, rank, window_days=3, max_pages=200):
    """
    Tier 3b: measures sustained post velocity over a fixed recent window
    (window_days), bucketed by calendar day — instead of a raw count up to
    an arbitrary cap. A count-based cap can't tell "sustained ~40/day" apart
    from "one viral 280-image day, then silence" if both land near the same
    total. Bucketing by day and reporting max_single_day / burst_ratio
    alongside the average makes that spike visible instead of averaging it
    away. Window is time-bounded, not count-bounded, so a genuinely quiet
    model finishes in one page while a genuinely busy one costs more calls —
    max_pages is a hard safety cap either way.
    """
    model_id = model.get("id")
    model_name = model.get("name", "Unknown")
    t_start = time.monotonic()
    window_since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat().replace("+00:00", "Z")

    result = {
        "modelId": model_id,
        "modelName": model_name,
        "download_rank": rank,
        "window_days": window_days,
        "window_total": 0,
        "velocity_per_day": 0.0,
        "max_single_day": 0,
        "burst_ratio": 0.0,
        "velocity_probe_status": "ok",
    }
    if not model_id:
        result["velocity_probe_status"] = "no_model_id"
        result["velocity_seconds"] = round(time.monotonic() - t_start, 3)
        return result

    img_url = f"{BASE}/images"
    params = {"modelId": model_id, "limit": page_limit, "nsfw": nsfw, "sort": "Newest"}

    day_counts = {}
    cursor = None
    n_pages = 0
    response_bytes = 0
    try:
        for _ in range(max_pages):
            if time.monotonic() - t_start > MODEL_TIMEOUT_SECONDS:
                result["velocity_probe_status"] = "model_timeout"
                break
            if cursor:
                params["cursor"] = cursor
            r = _get_with_retry(img_url, params)
            n_pages += 1
            response_bytes += len(r.content)
            payload = r.json()
            images = payload.get("items", [])
            if not images:
                break

            # Sorted Newest-first: the first image outside the window means every
            # remaining item on this page (and all subsequent pages) is older too.
            reached_end_of_window = False
            for img in images:
                created_at = img.get("createdAt", "")
                if created_at >= window_since:
                    day = created_at[:10]  # YYYY-MM-DD
                    day_counts[day] = day_counts.get(day, 0) + 1
                else:
                    reached_end_of_window = True
                    break

            if reached_end_of_window:
                break

            cursor = payload.get("metadata", {}).get("nextCursor")
            if not cursor:
                break
        else:
            result["velocity_probe_status"] = "max_pages_hit"
    except Exception as e:
        result["velocity_probe_status"] = "error"
        print(f"Error probing velocity for model {model_id} ({model_name}): {e}", flush=True)

    window_total = sum(day_counts.values())
    result["window_total"] = window_total
    result["velocity_per_day"] = round(window_total / window_days, 2) if window_days else float(window_total)
    if day_counts:
        max_day = max(day_counts.values())
        result["max_single_day"] = max_day
        result["burst_ratio"] = round(max_day / result["velocity_per_day"], 2) if result["velocity_per_day"] else 0.0

    result["velocity_seconds"] = round(time.monotonic() - t_start, 3)
    result["velocity_pages"] = n_pages
    result["response_bytes"] = response_bytes
    return result


def add_velocity(results, models, top_n, page_limit, nsfw, window_days=3, max_pages=200, max_workers=MAX_WORKERS):
    """
    Computes probe_recent_velocity for the top_n models (by total_probe_count)
    and merges the velocity fields into each matching result. Everything
    outside top_n is left alone — this is a top-of-the-list refinement, not a
    full re-probe.
    """
    models_by_id = {m.get("id"): m for m in models}
    ranked = sorted(results, key=lambda r: r["total_probe_count"], reverse=True)[:top_n]

    _log(f"Tier 3b velocity: {len(ranked)} model(s), max_workers={max_workers} "
         f"(window_days={window_days}, max_pages={max_pages} — this tier walks back page by page "
         f"per model, so it's the most likely place a busy model burns many sequential requests)")
    t0 = time.monotonic()
    reset_stats()

    velocity_by_id = {}
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        pool.submit(probe_recent_velocity, models_by_id[r["modelId"]], page_limit, nsfw,
                    r["download_rank"], window_days, max_pages): r["modelId"]
        for r in ranked
        if r["modelId"] in models_by_id
    }
    done, not_done = _wait_with_heartbeat(futures.keys(), PHASE_TIMEOUT_SECONDS, "Tier 3b velocity")
    for future in done:
        res = future.result()
        velocity_by_id[res["modelId"]] = res
    if not_done:
        _log(f"⚠️  Tier 3b phase timeout ({PHASE_TIMEOUT_SECONDS}s) hit with {len(not_done)} model(s) still "
             f"stuck in-flight — abandoning them rather than blocking the run. Their velocity fields are left "
             f"unset for this run (falls back to Tier 1/2 values only).")
        for future in not_done:
            mid = futures[future]
            velocity_by_id[mid] = {
                "modelId": mid, "modelName": models_by_id.get(mid, {}).get("name", "Unknown"),
                "window_days": window_days, "window_total": 0, "velocity_per_day": 0.0,
                "max_single_day": 0, "burst_ratio": 0.0, "velocity_probe_status": "phase_timeout",
                "velocity_seconds": PHASE_TIMEOUT_SECONDS, "velocity_pages": 0, "response_bytes": 0,
            }
    pool.shutdown(wait=False, cancel_futures=True)
    elapsed = time.monotonic() - t0
    s = get_stats()
    print(flush=True)
    _log(f"Tier 3b velocity done: {len(velocity_by_id)} model(s) in {elapsed:.2f}s")
    _log(f"  requests={s['requests']} ok={s['ok']} rate_limited_429={s['rate_limited']} "
         f"exceptions={s['exceptions']} gave_up={s['gave_up']}")
    _log(f"  time in network calls: {s['request_seconds']:.2f}s  |  time asleep in backoff: {s['backoff_seconds']:.2f}s")
    avg_kb = (s["response_bytes"] / s["ok"] / 1024) if s["ok"] else 0
    _log(f"  total response size: {s['response_bytes']/1_048_576:.1f} MB over {s['ok']} responses ({avg_kb:.1f} KB/response avg)")
    slowest = sorted(velocity_by_id.values(), key=lambda r: r.get("velocity_seconds", 0), reverse=True)[:10]
    if slowest and slowest[0].get("velocity_seconds", 0) > 1.0:
        print("  Slowest individual velocity probes (name, seconds, pages walked):", flush=True)
        for r in slowest:
            print(f"    {r['modelName'][:40]:40s} {r.get('velocity_seconds', 0):6.2f}s  "
                  f"pages={r.get('velocity_pages', '?')}", flush=True)
    heaviest = sorted(velocity_by_id.values(), key=lambda r: r.get("response_bytes", 0), reverse=True)[:10]
    avg_bytes = (s["response_bytes"] / len(velocity_by_id)) if velocity_by_id else 0
    if heaviest and avg_bytes and heaviest[0].get("response_bytes", 0) > 3 * avg_bytes:
        print(f"  Heaviest individual payloads (name, KB, vs {avg_bytes/1024:.1f}KB avg/model):", flush=True)
        for r in heaviest:
            print(f"    {r['modelName'][:40]:40s} {r.get('response_bytes', 0)/1024:8.1f} KB", flush=True)

    merged = []
    for r in results:
        v = velocity_by_id.get(r["modelId"])
        if v:
            r.update({
                "window_days": v["window_days"], "window_total": v["window_total"],
                "velocity_per_day": v["velocity_per_day"], "max_single_day": v["max_single_day"],
                "burst_ratio": v["burst_ratio"], "velocity_probe_status": v["velocity_probe_status"],
            })
        merged.append(r)
    return merged