"""
Fetch images for a named model (or an exact model ID), newest first.

Pipeline:
  1. Resolve the model(s):
       --model-id given  -> client.get_popular_models(only_ids=[...]) — exact,
           no search involved, always exactly one hit (or a fetch error).
       --model-name given -> client.get_popular_models(query=...) — Civitai's
           own substring/name search on /models. Can return more than one
           model if the name isn't unique (e.g. "detail" matches several
           LoRAs) — all matches are used unless --exact or --first is passed.
  2. images.fetch_images_for_models() — pull meta'd images across those
     models/versions.

No reaction ranking — output is sorted newest-createdAt-first, same
rationale as creator_cli.py (each underlying stream is already newest-first;
the final sort just guarantees that order holds once multiple models'/
versions' streams are merged).

Use:
    uv run python -m civitai_fetcher.model_cli --model-name "Detail Tweaker"
    uv run python -m civitai_fetcher.model_cli --model-id 58390
"""
import argparse
import json
from datetime import datetime, timedelta, timezone

from .config import OUT_PATH, ISSUES_PATH, IMAGES_MAX_PAGES, IMAGES_NSFW
from .client import get_popular_models
from .images import fetch_images_for_models
from .validate import validate_results
from .resolve import enrich_resources, load_cache, save_cache
from .generation_data import enrich_generation_data


def _page_size_type(value):
    ivalue = int(value)
    if not (50 <= ivalue <= 200):
        raise argparse.ArgumentTypeError(f"--page-size must be between 50 and 200 (got {ivalue})")
    return ivalue


def main():
    parser = argparse.ArgumentParser(
        description="Fetch images for a named model (or exact model ID), newest first."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name",
                        help="Model name to search for (substring, case-insensitive — Civitai's own "
                             "/models `query` filter, e.g. \"Detail Tweaker\"). Can match more than "
                             "one model; every match is used unless --exact or --first is passed.")
    group.add_argument("--model-id", type=int,
                        help="Exact numeric Civitai model ID (the number in civitai.com/models/58390-...). "
                             "Skips name search entirely — always exactly one model.")
    parser.add_argument("--exact", action="store_true",
                        help="Only used with --model-name. Keep only matches whose name is an exact, "
                             "case-insensitive match — use this if a broader search returned unwanted "
                             "extra hits and you want just the one.")
    parser.add_argument("--first", action="store_true",
                        help="Only used with --model-name. If the search returns multiple matches, use "
                             "just the first (Civitai's own relevance order) instead of all of them.")
    parser.add_argument("--max-age-days", type=int, default=3650,
                        help="Only include items created in the last N days (default: 3650, "
                             "i.e. effectively all-time). Combines with --limit as AND, not OR.")
    parser.add_argument("--media-type", choices=["image", "video", "audio", "all"], default="all",
                        help="Only fetch images, videos, or audio posts (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Hard cap on the number of items written, applied after fetching "
                             "(default: no limit, write everything that matched).")
    parser.add_argument("--page-size", type=_page_size_type, default=50,
                        help="Items requested per Civitai API page (50-200, default: 50).")
    parser.add_argument("--enrich-meta", action="store_true",
                        help="For entries still missing generation meta after the normal fetch, try "
                             "Civitai's internal per-image endpoint as a second pass (one HTTP call per "
                             "image — only applied to the final, already-limited set). Requires the "
                             "CIVITAI_COOKIE env var (your logged-in session cookie) — unofficial, "
                             "best-effort, and every failure is caught and skipped, never fatal.")

    # --- internal / advanced flags (hidden from --help) ---
    SUPPRESS = argparse.SUPPRESS
    parser.add_argument("--model-limit", type=int, default=20, help=SUPPRESS)
    parser.add_argument("--max-pages", type=int, default=IMAGES_MAX_PAGES, help=SUPPRESS)
    parser.add_argument("--max-versions", type=int, default=None, help=SUPPRESS)
    parser.add_argument("--nsfw", default=IMAGES_NSFW, help=SUPPRESS)
    parser.add_argument("--resolve-resources", dest="resolve_resources", action="store_true", default=True, help=SUPPRESS)
    parser.add_argument("--no-resolve-resources", dest="resolve_resources", action="store_false", help=SUPPRESS)
    args = parser.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.max_age_days)).isoformat().replace("+00:00", "Z")
    media_type = None if args.media_type == "all" else args.media_type

    # Step 1: resolve the model(s) — exact ID, or a name search.
    if args.model_id is not None:
        models = get_popular_models(only_ids=[args.model_id])
        if not models:
            print(f"No model found for ID {args.model_id}.")
            return
    else:
        models = get_popular_models(limit=args.model_limit, sort="Relevancy", query=args.model_name)
        if not models:
            print(f"No model found matching name '{args.model_name}'. Double check the spelling, "
                  f"or try a shorter/looser substring.")
            return
        if args.exact:
            before = len(models)
            models = [m for m in models if (m.get("name") or "").strip().lower() == args.model_name.strip().lower()]
            if not models:
                print(f"No exact match for '{args.model_name}' among {before} loose match(es) — "
                      f"drop --exact to see the loose matches, or check spelling.")
                return
        elif args.first and len(models) > 1:
            models = models[:1]

    print(f"Found {len(models)} model(s):")
    for m in models:
        print(f"  {m.get('name', 'Unknown')[:60]:60s} ({m.get('type', '?')})  id={m.get('id')}")
    if len(models) > 1 and args.model_name and not args.exact and not args.first:
        print("  (multiple matches — pass --exact for a strict name match, --first to just take the "
              "top hit, or --model-id once you know the one you want)")

    # Step 2: fetch every meta'd image in-window across those models/versions.
    entries = fetch_images_for_models(models, since, max_pages=args.max_pages, nsfw=args.nsfw,
                                       max_versions=args.max_versions, media_type=media_type,
                                       require_meta=False, page_size=args.page_size)
    fetched_count = len(entries)

    # Step 3: dedup by imageId (same image can appear under multiple model
    # versions if a post is shared).
    seen_image_ids: set[int] = set()
    deduped = []
    for e in entries:
        if e["imageId"] not in seen_image_ids:
            deduped.append(e)
            seen_image_ids.add(e["imageId"])
    if len(deduped) < len(entries):
        print(f"Deduped {len(entries) - len(deduped)} duplicate imageId(s)")
    entries = deduped

    # Step 4: newest first (see creator_cli.py for why this re-sort is needed
    # even though each underlying stream is already newest-first).
    ranked = sorted(entries, key=lambda e: e.get("createdAt") or "", reverse=True)

    if args.limit and len(ranked) > args.limit:
        ranked = ranked[:args.limit]

    if args.resolve_resources:
        load_cache()
        ranked = enrich_resources(ranked)
        save_cache()

    if args.enrich_meta:
        missing_before = sum(1 for e in ranked if not e.get("meta"))
        if missing_before:
            ranked = enrich_generation_data(ranked, only_missing=True)
        else:
            print("[generation_data] skipped — every item in the final set already has meta")

    # Tag output filenames with model name/id + run date, same rationale as
    # creator_cli.py — re-runs while tuning flags shouldn't silently overwrite
    # each other within the same day.
    label = str(args.model_id) if args.model_id is not None else args.model_name
    safe_label = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
    stamp = f"{safe_label}_{datetime.now().strftime('%d%b%y').lower()}"
    out_path = OUT_PATH.replace(".json", f"_{stamp}.json").replace("civitai_output_", "")
    issues_path = ISSUES_PATH.replace(".json", f"_{stamp}.json").replace("civitai_output_", "")

    import pathlib; pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ranked, f, indent=2)

    issues = validate_results(ranked)
    if issues:
        with open(issues_path, "w") as f:
            json.dump(issues, f, indent=2)
        print(f"Wrote issues to {issues_path}")

    print(f"\nWrote {len(ranked)} image(s) (of {fetched_count} fetched) for '{label}', "
          f"newest first, to {out_path}")
    print("Most recent 10:")
    for e in ranked[:10]:
        modellabel = (e.get("modelName") or "(no owning model)")[:40]
        print(f"  {e.get('createdAt', '?'):20s}  {modellabel:40s} {e['imageUrl']}")


if __name__ == "__main__":
    main()
