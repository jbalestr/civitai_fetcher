"""
Per-image generation-data enrichment via Civitai's INTERNAL tRPC endpoint
(generation.getGenerationData), the same call the website itself makes when
you open an image's detail view.

This is deliberately separate from client.py/images.py, which only ever
talk to the documented public /api/v1. Reasons this exists at all:

  - Some creators' images come back with meta=null from /api/v1, even
    though the browser clearly shows a prompt/resources panel for them
    (confirmed by hand, DevTools Network tab, civitai.red/images/<id>).
  - The public endpoint has documented combo-filter bugs (see images.py);
    this isn't one of those — the data is either not exposed via /api/v1
    at all, or is exposed differently than what the site's own detail
    view uses internally.

WHAT THIS NEEDS, AND WHY IT'S DIFFERENT FROM THE REST OF THE TOOL:
  - An authenticated session cookie (__Secure-civ-token), i.e. YOUR login,
    not an API key. Treat it like a password: never commit it, never put
    it directly in code — set it via the CIVITAI_COOKIE env var. If it's
    ever pasted somewhere it could be logged (chat, issue tracker, CI
    output), rotate it (log out + back in) — assume it's compromised.
  - It's undocumented and unofficial. Civitai could change or break this
    at any time without notice, unlike the versioned public API. Treat
    every call as best-effort: catch failures, never let one bad image
    stop the batch.
  - Because of both of the above, this is opt-in only (--enrich-meta in
    creator_cli.py) and deliberately only ever run against an already
    filtered/limited set of entries, never the full fetch — cost should
    scale with --limit, not with how many images a creator has.

RESPONSE FORMAT:
tRPC with `trpc-accept: application/jsonl` streams the response as
newline-delimited JSON chunks using its own dehydration/reference scheme
(chunk N can refer to placeholders resolved by a later chunk). Rather than
implement that protocol precisely, this takes the pragmatic route: parse
every line as JSON, then recursively walk the whole structure looking for
the one dict that has both a "resources" list and a "params" dict — that
shape is unique to the actual payload regardless of how it's wrapped, so
it doesn't matter exactly which chunk it landed in.
"""
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import SITE

TRPC_URL = f"{SITE}/api/trpc/generation.getGenerationData"

# Never hardcode a cookie. Set this in your shell before running:
#   export CIVITAI_COOKIE='__Secure-civ-token=...'   (or the full Cookie header)
# Rotate (log out/in) if it's ever been pasted anywhere that could be logged.
COOKIE = os.environ.get("CIVITAI_COOKIE", "")

_HEADERS = {
    "accept": "*/*",
    "trpc-accept": "application/jsonl",
    "x-client": "web",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

_session = requests.Session()
_stats_lock = threading.Lock()
_stats = {"requests": 0, "ok": 0, "no_data": 0, "errors": 0}
_logged_no_data_sample = False


def _reset_stats():
    global _logged_no_data_sample
    with _stats_lock:
        for k in _stats:
            _stats[k] = 0
        _logged_no_data_sample = False


def _find_payload(node):
    """
    Recursively search a parsed JSON structure for the dict shaped like
    {"resources": [...], "params": {...}} — the actual generation-data
    payload, regardless of which tRPC dehydration chunk it ended up in.
    Returns None if not found anywhere in this chunk.

    Deliberately never descends into a "remixOf" key: that holds a full
    record for a DIFFERENT image (the one this one was remixed from), which
    can itself have its own resources/params. Without this guard, a video
    with no generation params of its own would silently match its remix
    ancestor's params instead — attributing someone else's (or an earlier)
    generation to the wrong image. If the current image has no params of
    its own, that's genuinely "no data", not a reason to borrow one from
    somewhere else in the tree.
    """
    if isinstance(node, dict):
        if "resources" in node and "params" in node and isinstance(node.get("params"), dict):
            return node
        for k, v in node.items():
            if k == "remixOf":
                continue
            found = _find_payload(v)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_payload(v)
            if found is not None:
                return found
    return None


def _to_civitai_resources(resources):
    """Reshape the tRPC resource list into this tool's existing
    meta.civitaiResources shape (type/modelVersionId/modelName/weight),
    so downstream code (resolve.py, output consumers) doesn't need to
    know or care which source a given entry's meta came from."""
    out = []
    for r in resources or []:
        model = r.get("model") or {}
        out.append({
            "type": (model.get("type") or "").lower(),
            "modelVersionId": r.get("id"),
            "modelName": model.get("name"),
            "modelId": model.get("id"),
            "weight": r.get("strength"),
            "air": r.get("air"),
        })
    return out


def fetch_generation_data(image_id, timeout=15, max_retries=2):
    """
    Fetch generation data for one image via the internal tRPC endpoint.
    Returns a dict shaped like the public API's `meta` field (prompt,
    negativePrompt, civitaiResources, etc.) on success, or None if there's
    genuinely nothing there / the request fails after retries.

    Every failure mode is caught here and turned into None — this is
    best-effort, unofficial, and a single bad image must never take down
    a batch (see enrich_generation_data).
    """
    if not COOKIE:
        raise RuntimeError(
            "CIVITAI_COOKIE is not set. Export your session cookie first, e.g.:\n"
            "  export CIVITAI_COOKIE='__Secure-civ-token=<your token>'\n"
            "Grab it from DevTools > Network > any civitai.red request > Request Headers > cookie, "
            "while logged in. Treat it like a password — never commit it."
        )

    params_json = json.dumps({"0": {"json": {"type": "image", "id": image_id, "withPreview": True, "authed": True}}})
    url = f"{TRPC_URL}?batch=1&input={requests.utils.quote(params_json, safe='')}"
    headers = dict(_HEADERS)
    headers["cookie"] = COOKIE
    headers["referer"] = f"{SITE}/images/{image_id}"

    last_err = None
    for attempt in range(max_retries + 1):
        with _stats_lock:
            _stats["requests"] += 1
        try:
            r = _session.get(url, headers=headers, timeout=timeout)
            if r.status_code == 401 or r.status_code == 403:
                raise RuntimeError(
                    f"HTTP {r.status_code} — your CIVITAI_COOKIE is likely expired or invalid. "
                    f"Re-capture it from DevTools while logged in and re-export it."
                )
            r.raise_for_status()

            payload = None
            for line in r.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                found = _find_payload(chunk)
                if found is not None:
                    payload = found
                    break

            if payload is None:
                global _logged_no_data_sample
                with _stats_lock:
                    _stats["no_data"] += 1
                    log_sample = not _logged_no_data_sample
                    if log_sample:
                        _logged_no_data_sample = True
                if log_sample:
                    print(f"  [generation_data] no payload found for image {image_id} (HTTP {r.status_code}) — "
                          f"raw response sample (first no_data of this batch, to check for a disguised auth "
                          f"failure — tRPC often returns HTTP 200 with an error body instead of 401/403):\n"
                          f"    {r.text[:500]!r}", flush=True)
                return None

            params = payload.get("params") or {}
            with _stats_lock:
                _stats["ok"] += 1
            return {
                "prompt": params.get("prompt"),
                "negativePrompt": params.get("negativePrompt"),
                "Model": params.get("Model"),
                "sampler": params.get("sampler"),
                "cfgScale": params.get("cfgScale"),
                "steps": params.get("steps"),
                "seed": params.get("seed"),
                "Size": params.get("Size"),
                "civitaiResources": _to_civitai_resources(payload.get("resources")),
                # keep the raw params too — ADetailer/workflow/ecosystem fields etc.
                # vary a lot and aren't worth naming individually here.
                "raw": params,
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            with _stats_lock:
                _stats["errors"] += 1
            print(f"  [generation_data] image {image_id} failed after {max_retries + 1} attempt(s): {last_err}",
                  flush=True)
            return None


def enrich_generation_data(entries, only_missing=True, max_workers=6):
    """
    Fill in entry["meta"] via the internal endpoint for entries that don't
    already have it (only_missing=True, the default) — or for all entries
    if you explicitly want to re-fetch/override.

    Deliberately low max_workers (6) and deliberately NOT reusing client.py's
    connection pool/session: this hits an unofficial, cookie-authenticated
    endpoint, not the public API — keep it gentle, not parallelized like
    the bulk /api/v1 fetches elsewhere in this tool.

    Call this on an already-filtered/limited list — e.g. after --limit has
    been applied in creator_cli.py — not on a full unbounded fetch. Cost is
    one HTTP request per image, no bulk/paginated equivalent exists.
    """
    _reset_stats()
    targets = [e for e in entries if not only_missing or not e.get("meta")]
    if not targets:
        print("[generation_data] nothing to enrich (all entries already have meta)")
        return entries

    if not COOKIE:
        print("[generation_data] SKIPPED — CIVITAI_COOKIE is not set. Export your session cookie first:\n"
              "  export CIVITAI_COOKIE='__Secure-civ-token=<your token>'\n"
              "Grab it from DevTools > Network > any civitai.red request > Request Headers > cookie, "
              "while logged in. Treat it like a password — never commit it.")
        return entries

    print(f"[generation_data] enriching {len(targets)}/{len(entries)} entries via internal endpoint "
          f"(max_workers={max_workers})...")
    t0 = time.monotonic()
    completed = 0
    last_log = t0
    log_interval_seconds = 3.0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_generation_data, e["imageId"]): e for e in targets}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                data = future.result()
            except Exception as e:
                print(f"  [generation_data] image {entry.get('imageId')} raised unexpectedly: {e}", flush=True)
                continue
            if data is not None:
                entry["meta"] = data
            completed += 1
            now = time.monotonic()
            if now - last_log >= log_interval_seconds or completed == len(targets):
                print(f"  [generation_data] {completed}/{len(targets)} ({now - t0:.1f}s) — "
                      f"ok={_stats['ok']} no_data={_stats['no_data']} errors={_stats['errors']}", flush=True)
                last_log = now

    elapsed = time.monotonic() - t0
    print(f"[generation_data] done in {elapsed:.1f}s — ok={_stats['ok']} no_data={_stats['no_data']} "
          f"errors={_stats['errors']} (of {_stats['requests']} requests)")
    return entries