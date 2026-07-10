import os
import argparse
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE = "https://civitai.com/api/v1"
MAX_WORKERS = 10

def _get_with_retry(url, params, retries=3):
    """
    Helper function to execute requests with minimal retry logic.
    """
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                import time
                time.sleep(2 ** i)
        except requests.exceptions.RequestException:
            if i == retries - 1:
                raise
            import time
            time.sleep(2 ** i)
    raise requests.exceptions.RequestException(f"Failed to fetch from {url} after {retries} retries.")

def probe_model_activity(model, since, page_limit, nsfw, rank, deep_probe_limit):
    """
    Analyzes an individual model's image velocity by tracing recent image uploads.
    """
    model_id = model.get("id")
    model_name = model.get("name", "Unknown")
    model_type = model.get("type", "Unknown")
    
    result = {
        "modelId": model_id,
        "modelName": model_name,
        "type": model_type,
        "download_rank": rank,
        "page1_count": 0,
        "total_probe_count": 0,
        "has_more_pages": False
    }
    
    if not model_id:
        return result
        
    img_url = f"{BASE}/images"
    params = {
        "modelId": model_id,
        "limit": page_limit,
        "nsfw": nsfw,
        "sort": "Newest"
    }
    
    try:
        r = _get_with_retry(img_url, params)
        payload = r.json()
        images = payload.get("items", [])
        
        # Count items matching our timeframe condition
        recent_images = [img for img in images if img.get("createdAt", "") >= since]
        result["page1_count"] = len(recent_images)
        result["total_probe_count"] = len(recent_images)
        
        next_cursor = payload.get("metadata", {}).get("nextCursor")
        
        # Tier 2 Adaptive Deep Scan if the first page was completely saturated
        if len(recent_images) == page_limit and next_cursor:
            result["has_more_pages"] = True
            
            # If a strict cap is set and we've already hit it, stop right away
            if deep_probe_limit and result["total_probe_count"] >= deep_probe_limit:
                return result
                
            cursor = next_cursor
            while cursor:
                params["cursor"] = cursor
                r_deep = _get_with_retry(img_url, params)
                deep_payload = r_deep.json()
                deep_images = deep_payload.get("items", [])
                
                if not deep_images:
                    result["has_more_pages"] = False
                    break
                    
                deep_recent = [img for img in deep_images if img.get("createdAt", "") >= since]
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
        print(f"Error probing model {model_id} ({model_name}): {e}")
        
    return result

def get_popular_models(limit=10, sort="Most Downloaded", period="Month", types=None, max_lora_versions=None, only_ids=None):
    """
    period: Window to calculate popularity over: "Day", "Week", "Month", "Year", "AllTime"
    only_ids: if given (list/set of model IDs), skips the popularity query entirely.
    """
    if only_ids:
        items = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_get_with_retry, f"{BASE}/models/{mid}", {}): mid for mid in only_ids}
            for future in as_completed(futures):
                try:
                    items.append(future.result().json())
                except Exception as e:
                    mid = futures[future]
                    print(f"Failed to fetch explicit model ID {mid}: {e}")
        return items

    PAGE_SIZE = 100
    items = []
    seen_ids = set()
    cursor = None
    while len(items) < limit:
        params = {
            "limit": min(PAGE_SIZE, limit - len(items)), 
            "sort": sort,
            "period": period
        }
        if types:
            params["types"] = types
        if cursor:
            params["cursor"] = cursor
            
        r = _get_with_retry(f"{BASE}/models", params)
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
                      f"{len(m.get('modelVersions') or [])} versions (style/concept pack, likely)")
                continue
            kept.append(m)
        items = kept

    return items

def probe_candidates(candidate_count=100, since_days=30, period="Month", max_workers=MAX_WORKERS, nsfw="X",
                     types=None, max_lora_versions=None, page_limit=50, deep_probe_limit=None, only_ids=None):
    """
    Passes the period flag down to the model finder.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat().replace("+00:00", "Z")
    models = get_popular_models(limit=candidate_count, period=period, types=types, 
                                max_lora_versions=max_lora_versions, only_ids=only_ids)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(probe_model_activity, m, since, page_limit, nsfw, rank, deep_probe_limit): m
            for rank, m in enumerate(models, start=1)
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results