"""
Fetch images from one creator, newest first. Two scopes:

  --scope uploads  (default) — EVERYTHING this creator has posted to the
      gallery, including images made with someone else's checkpoint/LoRA:
        1. images.fetch_images_by_username() — paginate /images filtered
           by uploader username directly. No model discovery step at all.

  --scope models — only images attached to models this creator themselves
      published:
        1. client.get_popular_models(username=...) — list their models.
        2. images.fetch_images_for_models() — pull meta'd images across
           those models/versions.

No reaction ranking — output is just sorted newest-createdAt-first (each
underlying stream is already fetched newest-first; the final sort just
guarantees that order holds once multiple models'/versions' streams are
merged together).

Note: Civitai's API filters by USERNAME (profile name, e.g. "WAI0731"),
not a numeric user ID — there's no numeric-userId filter for /models, and
--scope uploads is the only mode that even accepts one (as `userId` on
/images), which this CLI doesn't expose. If you only have a numeric ID,
resolve it to a username via civitai.com/user/... first.

Use:
    uv run civitai-fetcher creators --username WAI0731
    uv run civitai-fetcher creators --username WAI0731 --scope models
"""
import argparse
import json
from datetime import datetime, timedelta, timezone

from ...core.config import OUT_PATH, ISSUES_PATH, IMAGES_MAX_PAGES, IMAGES_MAX_PAGES_SAFETY_CAP, IMAGES_NSFW
from ...core.client import get_popular_models
from ...services.fetch import fetch_images_for_models, fetch_images_by_username
from ...core.validate import validate_results
from ...services.enrichment.resolve import enrich_resources, load_cache as load_resolver_cache, save_cache as save_resolver_cache
from ...services.enrichment.generation import enrich_generation_data, load_cache as load_generation_cache, save_cache as save_generation_cache


def _page_size_type(value):
    ivalue = int(value)
    if not (50 <= ivalue <= 200):
        raise argparse.ArgumentTypeError(f"--page-size must be between 50 and 200 (got {ivalue})")
    return ivalue


def main():
    parser = argparse.ArgumentParser(
        description="Fetch images from one creator, newest first."
    )
    parser.add_argument("--username", required=True,
                        help="Civitai creator username/profile name (NOT a numeric user ID — "
                             "e.g. the 'WAI0731' in civitai.com/user/WAI0731).")
    parser.add_argument("--scope", choices=["models", "uploads"], default="uploads",
                        help="'uploads' (default): every asset this creator has posted, full stop — "
                             "including images made with someone else's model/checkpoint. "
                             "'models': only images on models this creator published.")
    parser.add_argument("--max-age-days", type=int, default=3650,
                        help="Only include items created in the last N days (default: 3650, "
                             "i.e. effectively all-time). Combines with --limit as AND, not OR: "
                             "an item must be within --max-age-days AND the overall output is "
                             "still capped at --limit — not 'whichever is looser'.")
    parser.add_argument("--types", nargs="+", default=None,
                        help="Restrict to specific model types (e.g. --types Checkpoint LORA). "
                             "Only used with --scope models.")
    parser.add_argument("--media-type", choices=["image", "video", "audio", "all"], default="all",
                        help="Only fetch images, videos, or audio posts (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Target number of items, ANDed with --max-age-days (an item must "
                             "satisfy both). For --scope uploads: stops paging Civitai as soon as "
                             "roughly this many in-window items are collected — checked between "
                             "pages, not mid-page, so once a page is fetched it's kept in full "
                             "rather than thrown away to hit an exact count (result may slightly "
                             "overshoot --limit). For --scope models: applied as a hard cap after "
                             "fetching (early-stop isn't wired up for this scope yet). "
                             "Default: no limit, write everything that matched.")
    parser.add_argument("--page-size", type=_page_size_type, default=50,
                        help="Items requested per Civitai API page (50-200, default: 50). Larger "
                             "pages mean fewer round-trips but bigger individual responses — each "
                             "page's response time/size is logged so this can be tuned from real "
                             "runs.")
    parser.add_argument("--enrich-meta", action="store_true",
                        help="For entries still missing generation meta after the normal fetch, "
                             "try Civitai's internal per-image endpoint as a second pass (one HTTP "
                             "call per image — only applied to the final, already-limited set, so "
                             "cost scales with --limit, not the whole gallery). Requires the "
                             "CIVITAI_COOKIE env var (your logged-in session cookie) — unofficial, "
                             "best-effort, and every failure is caught and skipped, never fatal.")

    # --- internal / advanced flags (hidden from --help) ---
    SUPPRESS = argparse.SUPPRESS
    parser.add_argument("--model-limit", type=int, default=1000, help=SUPPRESS)
    parser.add_argument("--max-pages", type=int, default=None, help=SUPPRESS)
    parser.add_argument("--max-versions", type=int, default=None, help=SUPPRESS)
    parser.add_argument("--nsfw", default=IMAGES_NSFW, help=SUPPRESS)
    parser.add_argument("--resolve-resources", dest="resolve_resources", action="store_true", default=True, help=SUPPRESS)
    parser.add_argument("--no-resolve-resources", dest="resolve_resources", action="store_false", help=SUPPRESS)
    args = parser.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.max_age_days)).isoformat().replace("+00:00", "Z")
    media_type = None if args.media_type == "all" else args.media_type

    # max_pages is an implementation-level safety cap, not something a user
    # should need to think about — but it still needs a value. If the user
    # explicitly passed --max-pages, that always wins. Otherwise: for
    # --scope uploads with --limit set, limit_total already bounds the
    # fetch (see images.py), so there's no need for a page-count ceiling on
    # top of it — drop it (float("inf")). --scope models doesn't have
    # limit_total wired up yet, so it keeps the default cap regardless of
    # --limit, otherwise --limit would stop bounding anything there at all.
    if args.max_pages is not None:
        effective_max_pages = args.max_pages
    elif args.scope == "uploads" and args.limit:
        effective_max_pages = IMAGES_MAX_PAGES_SAFETY_CAP
    else:
        effective_max_pages = IMAGES_MAX_PAGES

    if args.scope == "uploads":
        # No model discovery — one paginated stream over the uploader's
        # own username covers everything they've ever posted. Meta or no
        # meta, keep it all — some creators simply don't have generation
        # metadata attached to (some or all of) their posts, and that's not
        # a reason to drop the image/video itself.
        entries = fetch_images_by_username(args.username, since, max_pages=effective_max_pages, nsfw=args.nsfw,
                                            media_type=media_type, require_meta=False,
                                            page_size=args.page_size, limit_total=args.limit)
        if not entries:
            print(f"No posted images/videos found for '{args.username}' in the given window.")
            return
    else:
        # Step 1: list every model this creator has published — no
        # popularity ranking needed, just everything under their username.
        models = get_popular_models(limit=args.model_limit, sort="Newest", types=args.types, username=args.username)
        if not models:
            print(f"No models found for creator '{args.username}' — double check the username "
                  f"(profile name, not a numeric user ID).")
            return
        print(f"Found {len(models)} model(s) by '{args.username}':")
        for m in models:
            print(f"  {m.get('name', 'Unknown')[:60]:60s} ({m.get('type', '?')})")

        # Step 2: fetch every image in-window across all of those models,
        # meta or no meta (see uploads-scope comment above).
        entries = fetch_images_for_models(models, since, max_pages=effective_max_pages, nsfw=args.nsfw,
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

    # Step 4: newest first. Each version/username stream is already fetched
    # newest-first, but merging multiple streams (across models, or across
    # parallel workers) doesn't guarantee overall order — so re-sort once,
    # combined, on createdAt.
    ranked = sorted(entries, key=lambda e: e.get("createdAt") or "", reverse=True)

    # Belt-and-braces: --max-age-days and --limit are an AND, not an OR — every
    # item written must satisfy both. The age cutoff is already applied while
    # fetching (see images.py) via the `since` check on every page, so this is
    # already guaranteed and not re-checked here.

    # --scope models has no early-stop wired up yet (see fetch_images_for_models
    # call above — limit_total isn't passed), so --limit is applied as a hard
    # cap here instead. --scope uploads already stopped fetching once enough
    # in-window items were collected (limit_total), and keeps whatever full
    # page it landed on rather than being trimmed again here.
    if args.scope == "models" and args.limit and len(ranked) > args.limit:
        ranked = ranked[:args.limit]

    # Resource-name resolution (LoRA/checkpoint lookups) is a network call per
    # unique resource — run it on the final, already-limited set, not on
    # everything fetched before trimming.
    if args.resolve_resources:
        load_resolver_cache()
        ranked = enrich_resources(ranked)
        save_resolver_cache()

    if args.enrich_meta:
        missing_before = sum(1 for e in ranked if not e.get("meta"))
        if missing_before:
            load_generation_cache()
            ranked = enrich_generation_data(ranked, only_missing=True)
            save_generation_cache()
        else:
            print("[generation_data] skipped — every item in the final set already has meta")

    # Tag output filenames with creator + run date/time, same rationale as
    # images_cli.py — re-runs while tuning flags shouldn't silently overwrite
    # each other within the same day.
    # Filename: creator_date.json (date only, no time — re-running same day
    # overwrites, which is fine for this mode since it's "everything" not a
    # tuning run).
    stamp = f"{args.username.lower()}_{datetime.now().strftime('%d%b%y').lower()}"
    if args.scope == "uploads":
        stamp += "_uploads"
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

    print(f"\nWrote {len(ranked)} image(s) (of {fetched_count} fetched) from '{args.username}', "
          f"newest first, to {out_path}")
    print("Most recent 10:")
    for e in ranked[:10]:
        label = (e.get("modelName") or "(no owning model)")[:40]
        print(f"  {e.get('createdAt', '?'):20s}  {label:40s} {e['imageUrl']}")


if __name__ == "__main__":
    main()