"""
Low-level Civitai API client.

This module owns exactly one responsibility: talking to the Civitai API
reliably (retry/backoff, connection pooling, request stats) and discovering
models by download rank. It does NOT know about activity ranking (see
activity.py) or image fetching (see images.py) — both of those build on
top of this module, but never on each other, so a change to one can't
silently break the other.
"""
import random
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
import requests

from .config import (
    BASE, MAX_WORKERS, HEADERS, REQUEST_TIMEOUT, MAX_RETRIES,
    BACKOFF_CAP, BACKOFF_JITTER_MIN, BACKOFF_JITTER_MAX, MODELS_PAGE_SIZE, POOL_MAXSIZE,
)

# Shared session with a pooled connection adapter — plain requests.get() opens a fresh
# TCP+TLS connection every call. Logs showed ~1.3-1.4s average latency per request with
# zero rate-limiting, which is consistent with connection setup cost, not throttling.
# Reusing a pooled Session lets threads reuse warm connections to civitai.com.
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=POOL_MAXSIZE, pool_maxsize=POOL_MAXSIZE)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

_START = time.monotonic()

def _log(msg):
    """Timestamped diagnostic line: wall clock + seconds since process start,
    so you can line up 'slow phase' against 'what was running at that time'."""
    elapsed = time.monotonic() - _START
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts} +{elapsed:7.2f}s] {msg}", flush=True)

# --- request-level stats, updated from every worker thread ---
_stats_lock = threading.Lock()
_stats = {
    "requests": 0,       # total HTTP calls attempted (incl. retries)
    "ok": 0,             # 200s
    "rate_limited": 0,   # 429s hit
    "exceptions": 0,     # RequestException (timeout, connection, etc)
    "gave_up": 0,        # exhausted all retries
    "request_seconds": 0.0,   # time spent inside requests.get() itself
    "backoff_seconds": 0.0,   # time spent sleeping for backoff (not network — self-imposed)
    "response_bytes": 0,      # total bytes received across all successful responses
}

def get_stats():
    with _stats_lock:
        return dict(_stats)

def reset_stats():
    with _stats_lock:
        for k in _stats:
            _stats[k] = 0 if not isinstance(_stats[k], float) else 0.0

def _backoff_sleep(attempt, retry_after=None, cap=BACKOFF_CAP):
    """Sleep before a retry. Prefers the server's explicit Retry-After header
    (Civitai's rate-limit responses include one) over our own guess — avoids
    sleeping 10s when the limit actually resets in 1.5s, or the reverse."""
    if retry_after:
        try:
            sleep_for = min(float(retry_after), cap)
        except (TypeError, ValueError):
            retry_after = None
    if not retry_after:
        jitter = random.uniform(BACKOFF_JITTER_MIN, BACKOFF_JITTER_MAX)
        sleep_for = min((2 ** attempt) * jitter, cap)
    with _stats_lock:
        _stats["backoff_seconds"] += sleep_for
    time.sleep(sleep_for)

def _get_with_retry(url, params, retries=MAX_RETRIES):
    """
    Helper function to execute requests with minimal retry logic.
    Progress markers for long runs:
      '.' — succeeded on the first try
      'o' — succeeded, but only after at least one retry (429 or transient error)
      'X' — exhausted all retries, gave up
    'o' appearing is the early signal you're brushing up against the rate
    limit even though nothing is failing outright yet.
    """
    for i in range(retries):
        with _stats_lock:
            _stats["requests"] += 1
        t0 = time.monotonic()
        try:
            r = _session.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException:
            with _stats_lock:
                _stats["request_seconds"] += time.monotonic() - t0
                _stats["exceptions"] += 1
            if i == retries - 1:
                with _stats_lock:
                    _stats["gave_up"] += 1
                print("X", end="", flush=True)
                raise
            _backoff_sleep(i)
            continue

        with _stats_lock:
            _stats["request_seconds"] += time.monotonic() - t0
        if r.status_code == 200:
            with _stats_lock:
                _stats["ok"] += 1
                _stats["response_bytes"] += len(r.content)
            print("." if i == 0 else "o", end="", flush=True)
            return r
        if r.status_code == 429:
            with _stats_lock:
                _stats["rate_limited"] += 1
            _backoff_sleep(i, retry_after=r.headers.get("Retry-After"))
            continue
        if r.status_code >= 500:
            # Server-side/transient (500, 502, 503, 504) — Civitai's 503 body
            # literally says "temporarily overloaded — please retry", so unlike
            # a genuine client error, backing off and retrying is the right
            # move here, same as a 429.
            with _stats_lock:
                _stats["rate_limited"] += 1
            if i == retries - 1:
                with _stats_lock:
                    _stats["gave_up"] += 1
                print("X", end="", flush=True)
                raise requests.exceptions.RequestException(
                    f"{url} returned {r.status_code} after {retries} retries: {r.text[:500]}"
                )
            _backoff_sleep(i, retry_after=r.headers.get("Retry-After"))
            continue
        # Any other non-200 status (400, 403, 404, etc.) — a genuine client-side
        # error, not transient — retrying blindly just burns attempts and hides
        # the real cause. Surface it immediately instead of falling through to
        # the generic "failed after N retries" message.
        with _stats_lock:
            _stats["gave_up"] += 1
        print("X", end="", flush=True)
        raise requests.exceptions.RequestException(
            f"{url} returned {r.status_code}: {r.text[:500]}"
        )
    with _stats_lock:
        _stats["gave_up"] += 1
    print("X", end="", flush=True)
    raise requests.exceptions.RequestException(f"Failed to fetch from {url} after {retries} retries.")

def get_popular_models(limit=10, sort="Most Downloaded", period="Month", types=None, max_lora_versions=None, only_ids=None):
    """
    period: Window to calculate popularity over: "Day", "Week", "Month", "Year", "AllTime"
    only_ids: if given (list/set of model IDs), skips the popularity query entirely.
    """
    t0 = time.monotonic()
    if only_ids:
        _log(f"Discovery: fetching {len(only_ids)} explicit model ID(s) directly (max_workers={MAX_WORKERS})...")
        items = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_get_with_retry, f"{BASE}/models/{mid}", {}): mid for mid in only_ids}
            for future in as_completed(futures):
                try:
                    items.append(future.result().json())
                except Exception as e:
                    mid = futures[future]
                    print(f"Failed to fetch explicit model ID {mid}: {e}", flush=True)
        _log(f"Discovery done: {len(items)} model(s) in {time.monotonic()-t0:.2f}s")
        return items

    _log(f"Discovery: paging /models (sort={sort}, period={period}) until {limit} candidates...")
    items = []
    seen_ids = set()
    cursor = None
    n_pages = 0
    while len(items) < limit:
        params = {
            "limit": min(MODELS_PAGE_SIZE, limit - len(items)),
            "sort": sort,
            "period": period
        }
        if types:
            params["types"] = types
        if cursor:
            params["cursor"] = cursor

        r = _get_with_retry(f"{BASE}/models", params)
        n_pages += 1
        payload = r.json()
        page_items = payload.get("items", [])
        if not page_items:
            break

        new_items = [m for m in page_items if m.get("id") not in seen_ids]
        if not new_items:
            break
        for m in new_items:
            if m.get("id"):
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
                print(f"  Skipping {m.get('name')} ({m.get('id')}) — LORA with "
                      f"{len(m.get('modelVersions') or [])} versions (style/concept pack, likely)", flush=True)
                continue
            kept.append(m)
        items = kept

    _log(f"Discovery done: {len(items)} model(s) over {n_pages} page(s) in {time.monotonic()-t0:.2f}s "
         f"(this part is sequential/single-threaded — {n_pages} network round-trips, one after another)")
    return items

def _wait_with_heartbeat(futures, phase_timeout, label, heartbeat_interval=10):
    """Like wait(..., timeout=phase_timeout, return_when=ALL_COMPLETED), but prints
    a progress line every heartbeat_interval seconds instead of blocking silently —
    so a long-but-legitimate wait doesn't look identical to a genuine hang.

    Abandons anything still running past phase_timeout (returned in not_done) —
    appropriate for activity probing, where a stuck model shouldn't block an
    otherwise-fast run. NOT appropriate where completeness matters more than a
    bounded runtime (e.g. image fetching) — use _wait_all_with_heartbeat there,
    since abandoning a future here doesn't cancel it, it just discards the result
    once it does finish, wasting the work for nothing."""
    total = len(futures)
    deadline = time.monotonic() + phase_timeout
    done = set()
    not_done = set(futures)
    while not_done:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        slice_done, not_done = wait(not_done, timeout=min(heartbeat_interval, remaining), return_when=ALL_COMPLETED)
        done |= slice_done
        if not_done:
            _log(f"  ...{label} still running: {len(done)}/{total} done, {len(not_done)} in flight "
                 f"(phase timeout in {max(0, deadline - time.monotonic()):.0f}s)")
    return done, not_done


def _wait_all_with_heartbeat(futures, label, heartbeat_interval=10):
    """Like _wait_with_heartbeat, but with no phase timeout — waits for every
    future to complete, no matter how long it takes, printing a heartbeat line
    every heartbeat_interval seconds so a long-but-legitimate wait is still
    visible. Use this wherever losing a result is worse than running long,
    e.g. image fetching, where a model just being slow (503 storms, deep
    pagination) shouldn't cost you the images it already found."""
    total = len(futures)
    done = set()
    not_done = set(futures)
    while not_done:
        slice_done, not_done = wait(not_done, timeout=heartbeat_interval, return_when=ALL_COMPLETED)
        done |= slice_done
        if not_done:
            _log(f"  ...{label} still running: {len(done)}/{total} done, {len(not_done)} in flight "
                 f"(no phase timeout — waiting for all to finish)")
    return done