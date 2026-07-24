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
    uv run python -m civitai_fetcher.creator_cli --username WAI0731
    uv run python -m civitai_fetcher.creator_cli --username WAI0731 --scope models
"""
import argparse
import json
from datetime import datetime, timedelta, timezone

from .config import OUT_PATH, ISSUES_PATH, IMAGES_MAX_PAGES, IMAGES_NSFW
from .client import get_popular_models
from .images import fetch_images_for_models, fetch_images_by_username
from .validate import validate_results
from .resolve import enrich_resources, load_cache, save_cache


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
                        help="Cap the number of items WRITTEN to the output JSON, applied after "
                             "all filters (--media-type, --max-age-days, --types) and after sorting "
                             "newest-first — i.e. the N most recent items that matched, and each of "
                             "those N must still fall within --max-age-days. "
                             "Default: no limit, write everything that matched.")
    parser.add_argument("--allow-no-meta", action="store_true",
                        help="Also include images/videos with no generation metadata attached "
                             "(default: skipped). Needed for creators who strip/hide their prompts "
                             "entirely — Civitai's API otherwise returns nothing at all for them "
                             "(a nextCursor forever with zero items per page) rather than a clean "
                             "'no results', since withMeta=true excludes everything they've posted.")

    # --- internal / advanced flags (hidden from --help) ---
    SUPPRESS = argparse.SUPPRESS
    parser.add_argument("--model-limit", type=int, default=1000, help=SUPPRESS)
    parser.add_argument("--max-pages", type=int, default=IMAGES_MAX_PAGES, help=SUPPRESS)
    parser.add_argument("--max-versions", type=int, default=None, help=SUPPRESS)
    parser.add_argument("--nsfw", default=IMAGES_NSFW, help=SUPPRESS)
    parser.add_argument("--top-resources", type=int, default=0, help=SUPPRESS)
    parser.add_argument("--resolve-resources", dest="resolve_resources", action="store_true", default=True, help=SUPPRESS)
    parser.add_argument("--no-resolve-resources", dest="resolve_resources", action="store_false", help=SUPPRESS)
    args = parser.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.max_age_days)).isoformat().replace("+00:00", "Z")
    media_type = None if args.media_type == "all" else args.media_type

    if args.scope == "uploads":
        # No model discovery — one paginated stream over the uploader's
        # own username covers everything they've ever posted.
        entries = fetch_images_by_username(args.username, since, max_pages=args.max_pages, nsfw=args.nsfw,
                                            media_type=media_type, require_meta=not args.allow_no_meta)
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

        # Step 2: fetch every image in-window across all of those models.
        entries = fetch_images_for_models(models, since, max_pages=args.max_pages, nsfw=args.nsfw,
                                           max_versions=args.max_versions, media_type=media_type,
                                           require_meta=not args.allow_no_meta)

    if args.resolve_resources:
        load_cache()
        entries = enrich_resources(entries)
        save_cache()

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
    # fetching (see images.py), but re-checking here means this guarantee holds
    # regardless of how entries got here.
    if args.max_age_days:
        before = len(ranked)
        ranked = [e for e in ranked if (e.get("createdAt") or "") >= since]
        if len(ranked) < before:
            print(f"Dropped {before - len(ranked)} item(s) older than --max-age-days={args.max_age_days}")

    if args.limit and len(ranked) > args.limit:
        ranked = ranked[:args.limit]

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

    print(f"\nWrote {len(ranked)} image(s) (of {len(entries)} fetched) from '{args.username}', "
          f"newest first, to {out_path}")
    print("Most recent 10:")
    for e in ranked[:10]:
        label = (e.get("modelName") or "(no owning model)")[:40]
        print(f"  {e.get('createdAt', '?'):20s}  {label:40s} {e['imageUrl']}")


if __name__ == "__main__":
    main()