"""
Fetch images for a named model (or an exact model ID), newest first.

Pipeline:
  1. Resolve the model(s):
       --model-id given   -> client.get_popular_models(only_ids=[...]) — exact,
           no search involved, always exactly one hit (or a fetch error).
       --model-name given -> client.get_popular_models(query=...) — Civitai's
           own fuzzy name search on /models, then filtered down to an exact
           (case-insensitive) name match — a single specific target, same as
           --model-id. Pass --loose to skip that filter and use every fuzzy
           match instead.
  2. images.fetch_images_for_models() — pull meta'd images across those
     models/versions.

No reaction ranking — output is sorted newest-createdAt-first, same
rationale as creator_cli.py (each underlying stream is already newest-first;
the final sort just guarantees that order holds once multiple models'/
versions' streams are merged).

Use:
    uv run civitai-fetcher models --model-name "Detail Tweaker"
    uv run civitai-fetcher models --model-id 58390
"""
import argparse
import json
import re
import sys
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta, timezone

from ...core.config import OUT_PATH, ISSUES_PATH, IMAGES_MAX_PAGES, IMAGES_NSFW, DB_PATH
from ...db import write_to_db
from ...core.client import get_popular_models
from ...services.fetch import fetch_images_for_models
from ...core.validate import validate_results
from ...services.enrichment.resolve import enrich_resources, load_cache as load_resolver_cache, save_cache as save_resolver_cache
from ...services.enrichment.generation import enrich_generation_data, load_cache as load_generation_cache, save_cache as save_generation_cache


def _page_size_type(value):
    ivalue = int(value)
    if not (50 <= ivalue <= 200):
        raise argparse.ArgumentTypeError(f"--page-size must be between 50 and 200 (got {ivalue})")
    return ivalue


def _parse_civitai_url(url):
    """
    Pull (model_id, version_id_or_None, slug_or_None) out of a pasted
    civitai.com/civitai.red model page URL, e.g.:
      https://civitai.red/models/443821/cyberrealistic-pony?modelVersionId=2884631
      https://civitai.com/models/1412827/illustrious-realism-by-klaabu
    version_id is None if the URL didn't have a ?modelVersionId= — that just
    means "use every version" (or combine with --max-versions), same as
    --model-id alone with no --version-id. slug is the human-readable part
    of the path (e.g. "cyberrealistic-pony"), used only for the output
    filename — None if the URL was bare (just /models/<id>).
    Raises ValueError with a plain-English reason on anything that doesn't
    look like a Civitai model URL, so main() can print it and exit cleanly
    rather than crash with a stack trace.
    """
    parsed = urlparse(url)
    if "civitai" not in parsed.netloc:
        raise ValueError(f"'{url}' doesn't look like a civitai.com/civitai.red URL")
    m = re.search(r"/models/(\d+)(?:/([^/?]+))?", parsed.path)
    if not m:
        raise ValueError(f"couldn't find a model ID in '{url}' — expected .../models/<number>/...")
    model_id = int(m.group(1))
    slug = m.group(2)
    version_id = None
    qs = parse_qs(parsed.query)
    if "modelVersionId" in qs:
        try:
            version_id = int(qs["modelVersionId"][0])
        except (ValueError, IndexError):
            pass
    return model_id, version_id, slug


def main():
    parser = argparse.ArgumentParser(
        description="Fetch images for a named model (or exact model ID), newest first."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name",
                        help="Exact model name (case-insensitive), e.g. \"CyberRealistic Pony\" — same "
                             "kind of single, specific target as --model-id, just by name instead of "
                             "number. Internally does a Civitai name search then keeps only the exact "
                             "match; pass --loose to instead keep every fuzzy match.")
    group.add_argument("--model-id", type=int,
                        help="Exact numeric Civitai model ID (the number in civitai.com/models/58390-...). "
                             "Skips name search entirely — always exactly one model.")
    group.add_argument("--url",
                        help="Paste a civitai.com/civitai.red model page URL directly, e.g. "
                             "\"https://civitai.red/models/443821/cyberrealistic-pony?modelVersionId=2884631\". "
                             "Pulls the model ID out automatically, and the version ID too if the URL has "
                             "?modelVersionId= — same effect as passing --model-id and --version-id by hand.")
    parser.add_argument("--loose", action="store_true",
                        help="With --model-name: keep every fuzzy match Civitai's search returns instead "
                             "of just the exact one, and fetch images across all of them. Off by default "
                             "since it can silently pull in unrelated models sharing a common word.")
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
    parser.add_argument("--version-id", type=int, action="append", default=None,
                        help="Pull only from this modelVersionId (repeatable, e.g. "
                             "--version-id 123 --version-id 456) instead of walking every "
                             "version. Use this to target a specific older version once you "
                             "know its ID — overrides --max-versions.")
    parser.add_argument("--max-versions", type=int, default=None,
                        help="Only walk the newest N versions (Civitai lists them newest-first) "
                             "instead of all of them. Ignored if --version-id is given. Useful "
                             "when a model has a long tail of old/rarely-used versions.")
    parser.add_argument("--no-db", action="store_true",
                        help=f"Skip writing to the DB ({DB_PATH}) -- JSON file output only.")
    parser.add_argument("--enrich-meta", action="store_true",
                        help="For entries still missing generation meta after the normal fetch, try "
                             "Civitai's internal per-image endpoint as a second pass (one HTTP call per "
                             "image — only applied to the final, already-limited set). Requires the "
                             "CIVITAI_COOKIE env var (your logged-in session cookie) — unofficial, "
                             "best-effort, and every failure is caught and skipped, never fatal.")

    # --- internal / advanced flags (hidden from --help) ---
    SUPPRESS = argparse.SUPPRESS
    # 100 here (vs. e.g. 20) because this is just the search candidate pool that gets
    # exact-filtered down to one match afterward — cheap (MODELS_PAGE_SIZE=100 means this
    # is usually a single API page) and avoids the real target being pushed out of a
    # smaller top-N window by more-downloaded models sharing a word in the name.
    parser.add_argument("--model-limit", type=int, default=100, help=SUPPRESS)
    parser.add_argument("--max-pages", type=int, default=IMAGES_MAX_PAGES, help=SUPPRESS)
    parser.add_argument("--nsfw", default=IMAGES_NSFW, help=SUPPRESS)
    parser.add_argument("--resolve-resources", dest="resolve_resources", action="store_true", default=True, help=SUPPRESS)
    parser.add_argument("--no-resolve-resources", dest="resolve_resources", action="store_false", help=SUPPRESS)
    args = parser.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.max_age_days)).isoformat().replace("+00:00", "Z")
    media_type = None if args.media_type == "all" else args.media_type

    if args.url:
        try:
            args.model_id, url_version_id, url_slug = _parse_civitai_url(args.url)
        except ValueError as e:
            print(f"Couldn't parse --url: {e}")
            return
        if url_version_id is not None:
            args.version_id = [url_version_id]
        print(f"Parsed --url: model_id={args.model_id}"
              + (f", version_id={url_version_id}" if url_version_id is not None else " (no version in URL — all versions)"))
    else:
        url_slug = None
        url_version_id = None

    # Step 1: resolve the model(s) — exact ID, or a name search.
    if args.model_id is not None:
        models = get_popular_models(only_ids=[args.model_id])
        if not models:
            print(f"No model found for ID {args.model_id}.")
            return
    else:
        # NOT sort="Newest" here (unlike creator_cli.py's username listing, which is a
        # different situation — listing everything a *known* creator made). Newest biases
        # the candidate window toward whatever's freshly published matching those words,
        # and can push an established, popular model (the one you actually meant) out of
        # the top N entirely. Most Downloaded surfaces the well-known match instead.
        models = get_popular_models(limit=args.model_limit, sort="Most Downloaded", period="AllTime",
                                     query=args.model_name)
        if not models:
            print(f"No model found matching name '{args.model_name}'. Double check the spelling, "
                  f"or try a shorter/looser substring.")
            return
        if not args.loose:
            before = len(models)
            exact = [m for m in models if (m.get("name") or "").strip().lower() == args.model_name.strip().lower()]
            if not exact:
                print(f"No exact match for '{args.model_name}' among {before} loose match(es) within the "
                      f"top {args.model_limit} candidates:")
                for m in models[:10]:
                    print(f"  {m.get('name', 'Unknown')[:60]:60s} id={m.get('id')}")
                print("Copy the exact name from above, use --model-id if you know it, or pass --loose to "
                      "fetch across every loose match instead.")
                return
            models = exact

    print(f"Found {len(models)} model(s):")
    for m in models:
        print(f"  {m.get('name', 'Unknown')[:60]:60s} ({m.get('type', '?')})  id={m.get('id')}")

    if args.version_id:
        wanted = set(args.version_id)
        for m in models:
            all_versions = m.get("modelVersions") or []
            kept = [v for v in all_versions if v.get("id") in wanted]
            found_ids = {v.get("id") for v in kept}
            missing = wanted - found_ids
            if missing:
                print(f"  Warning: version id(s) {sorted(missing)} not found on '{m.get('name')}' — ignoring.")
            m["modelVersions"] = kept
        if not any(m.get("modelVersions") for m in models):
            print(f"None of --version-id {args.version_id} matched any version on the resolved model(s).")
            return
        print(f"Restricted to explicit version(s): {sorted(wanted)}")
    else:
        for m in models:
            print(f"    {len(m.get('modelVersions') or [])} version(s) available"
                  + (f", walking newest {args.max_versions}" if args.max_versions else ", walking all")
                  + " (newest first)")

    # Step 2: fetch every meta'd image in-window across those models/versions.
    # limit_total early-stops the fetch itself (checked between pages, via a shared
    # stop event) once roughly --limit images are collected — without this, --limit
    # only trims the result *after* every page for every matched model was already
    # pulled, which is enormously wasteful for a small --limit on a popular model.
    entries = fetch_images_for_models(models, since, max_pages=args.max_pages, nsfw=args.nsfw,
                                       max_versions=None if args.version_id else args.max_versions,
                                       media_type=media_type,
                                       require_meta=False, page_size=args.page_size,
                                       limit_total=args.limit)
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

    # enrich_generation_data() runs BEFORE enrich_resources(), not after: it's
    # the only thing that can add civitaiResources at all for entries whose
    # public-API meta came back null. Running resolve_resources() first (the
    # old order) meant it only ever saw whatever civitaiResources already
    # existed pre-generation_data -- often zero -- so creatorUsername/name
    # never got filled in for anything generation_data added. This order
    # guarantees resolve_resources() sees the complete, final resource list.
    if args.enrich_meta:
        missing_before = sum(1 for e in ranked if not e.get("meta"))
        if missing_before:
            load_generation_cache()
            ranked = enrich_generation_data(ranked, only_missing=True)
            save_generation_cache()
        else:
            print("[generation_data] skipped — every item in the final set already has meta")

    if args.resolve_resources:
        load_resolver_cache()
        ranked = enrich_resources(ranked)
        save_resolver_cache()

    # Tag output filenames with a readable label + run date, same rationale as
    # creator_cli.py — re-runs while tuning flags shouldn't silently overwrite
    # each other within the same day.
    if args.url:
        # slug + model_id (+ version_id, if the URL locked to one) reads far better
        # than a bare number, e.g. "cyberrealistic-pony_443821_v2884631" instead of
        # just "443821" — and still stays unique enough to not collide across models.
        label = "_".join(str(p) for p in [url_slug, args.model_id,
                                           f"v{url_version_id}" if url_version_id else None] if p)
    elif args.model_id is not None:
        label = str(args.model_id)
    else:
        label = args.model_name
    safe_label = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
    stamp = f"{safe_label}_{datetime.now().strftime('%d%b%y').lower()}"
    out_path = OUT_PATH.replace(".json", f"_{stamp}.json").replace("civitai_output_", "")
    issues_path = ISSUES_PATH.replace(".json", f"_{stamp}.json").replace("civitai_output_", "")
    meta_path = out_path.replace(".json", "_meta.json")

    import pathlib; pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ranked, f, indent=2)

    if not args.no_db:
        write_to_db(ranked, DB_PATH)

    # Sidecar, not embedded in the main file — patterns_cli.py and creator_cli.py both
    # expect the output file to be a bare list of entries, so run provenance goes
    # alongside it instead of wrapping/breaking that shape.
    run_meta = {
        "generatingCli": "civitai-fetcher models",
        "commandLine": "uv run civitai-fetcher models " + " ".join(
            f'"{a}"' if " " in a else a for a in sys.argv[1:]
        ),
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "resolvedModels": [{"id": m.get("id"), "name": m.get("name")} for m in models],
        "outputCount": len(ranked),
        "fetchedCount": fetched_count,
    }
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    issues = validate_results(ranked)
    if issues:
        with open(issues_path, "w") as f:
            json.dump(issues, f, indent=2)
        print(f"Wrote issues to {issues_path}")

    print(f"\nWrote {len(ranked)} image(s) (of {fetched_count} fetched) for '{label}', "
          f"newest first, to {out_path}\n(run metadata: {meta_path})")
    print("Most recent 10:")
    for e in ranked[:10]:
        modellabel = (e.get("modelName") or "(no owning model)")[:40]
        print(f"  {e.get('createdAt', '?'):20s}  {modellabel:40s} {e['imageUrl']}")


if __name__ == "__main__":
    main()