"""
Find popular Civitai models by CURRENT activity (not just historical
downloads — see README "Discoveries"), then pull their images ranked by
community reactions.

This is the replacement for the old, broken `civitai-fetcher` CLI
(civitai_fetcher.cli / fetch_all). That command imported a fetch_all that
no longer existed after the codebase split into activity.py (model ranking)
and images.py (image fetching) — see git history around commit 1df2185.

Pipeline:
  1. activity.probe_candidates()  — rank models by download popularity, then
     probe each for recent post-count (Tier 1/2) and optionally sustained
     velocity (Tier 3b) — same logic probe.py uses.
  2. Take the top --top-models by that ranking.
  3. images.fetch_images_for_models() — pull every meta'd image in the
     window for just those models (not the whole candidate pool).
  4. images.sort_by_reactions() — rank pulled images by combined reaction
     count, client-side (see images.py for why this isn't done via the API).
"""
import argparse
import json
from datetime import datetime

from .config import (
    OUT_PATH, ISSUES_PATH,
    IMAGES_PERIOD, IMAGES_SINCE_DAYS, IMAGES_TOP_MODELS, IMAGES_MAX_PAGES, IMAGES_NSFW,
    IMAGES_TOP_REACTIONS,
    PROBE_CANDIDATE_COUNT, PROBE_PAGE_LIMIT, PROBE_DEEP_PROBE_LIMIT, PROBE_TYPES,
    PROBE_VELOCITY_WINDOW_DAYS, PROBE_VELOCITY_MAX_PAGES,
)
from .activity import probe_candidates, add_velocity
from .images import (
    fetch_images_for_models, sort_by_reactions, sort_by_reactions_per_model,
    count_resource_usage, count_bare_checkpoint_usage,
)
from .validate import validate_results
from .resolve import enrich_resources, load_cache, save_cache


def main():
    parser = argparse.ArgumentParser(
        description="Find currently-active popular Civitai models, then fetch their images ranked by reactions."
    )
    parser.add_argument("--period", default=IMAGES_PERIOD, choices=["Day", "Week", "Month", "Year", "AllTime"],
                        help=f"Window to rank model popularity/activity over (default: {IMAGES_PERIOD} — "
                             f"best signal-to-noise in practice, see README)")
    parser.add_argument("--since-days", type=int, default=IMAGES_SINCE_DAYS,
                        help="Image activity/fetch window in days (default matches --period)")
    parser.add_argument("--candidate-count", type=int, default=PROBE_CANDIDATE_COUNT,
                        help="How many models to discover by download rank before activity-ranking them")
    parser.add_argument("--top-models", type=int, default=IMAGES_TOP_MODELS,
                        help="How many activity-ranked models to actually fetch images for")
    parser.add_argument("--types", nargs="+", default=[PROBE_TYPES],
                        help="Restrict to these Civitai model types, e.g. --types Checkpoint LORA")
    parser.add_argument("--max-lora-versions", type=int, default=None,
                        help="Skip LORA-type models with more than N versions (style/concept bundle packs)")
    parser.add_argument("--page-limit", type=int, default=PROBE_PAGE_LIMIT, help="Tier 1/2 activity-probe page depth")
    parser.add_argument("--deep-probe-limit", type=int, default=PROBE_DEEP_PROBE_LIMIT,
                        help="Tier 2 adaptive cap for the activity probe")
    parser.add_argument("--velocity-window-days", type=int, default=PROBE_VELOCITY_WINDOW_DAYS,
                        help="Tier 3b sustained-velocity window, used to break ties at --deep-probe-limit")
    parser.add_argument("--velocity-max-pages", type=int, default=PROBE_VELOCITY_MAX_PAGES,
                        help="Tier 3b safety cap on pages walked per model")
    parser.add_argument("--max-pages", type=int, default=IMAGES_MAX_PAGES,
                        help="Max pages to walk back per model VERSION during image fetch")
    parser.add_argument("--max-versions", type=int, default=None,
                        help="Only fetch images from the newest N versions per model (default: all versions)")
    parser.add_argument("--nsfw", default=IMAGES_NSFW,
                        help="Civitai nsfw param (None/Soft/Mature/X/true/false). Default 'X' avoids silent NSFW filtering")
    parser.add_argument("--top-reactions", type=int, default=IMAGES_TOP_REACTIONS,
                        help="How many top-reaction images to keep GLOBALLY (0 = keep all fetched). "
                             "Ignored if --top-reactions-per-model is set.")
    parser.add_argument("--top-reactions-per-model", type=int, default=None,
                        help="Instead of one global top-N, keep the top N images PER MODEL — guarantees "
                             "every fetched model is represented, useful for comparing models/checkpoint "
                             "types rather than letting one high-reaction model crowd out the rest. "
                             "Overrides --top-reactions when set.")
    parser.add_argument("--top-resources", type=int, default=15,
                        help="How many entries to print in the LoRA/combo usage breakdowns (0 = skip this section)")
    parser.add_argument("--resolve-resources", dest="resolve_resources", action="store_true", default=True,
                        help="Resolve civitaiResources modelVersionIds to names + creator usernames (default: on — "
                             "you'll almost always want readable names, not raw IDs)")
    parser.add_argument("--no-resolve-resources", dest="resolve_resources", action="store_false",
                        help="Skip resource-name resolution — faster, but LoRA/checkpoint names in the usage "
                             "breakdowns and civitaiResources fields stay as raw modelVersionIds")
    args = parser.parse_args()

    # Step 1: rank models by download popularity, then activity (Tier 1/2 + Tier 3b).
    results, models, since = probe_candidates(
        candidate_count=args.candidate_count, since_days=args.since_days, period=args.period,
        nsfw=args.nsfw, types=args.types, max_lora_versions=args.max_lora_versions,
        page_limit=args.page_limit, deep_probe_limit=args.deep_probe_limit,
    )
    if args.velocity_window_days:
        results = add_velocity(
            results, models, top_n=min(args.top_models * 3, len(results)),
            page_limit=args.page_limit, nsfw=args.nsfw,
            window_days=args.velocity_window_days, max_pages=args.velocity_max_pages,
        )
        results = sorted(results, key=lambda r: r.get("velocity_per_day", 0), reverse=True)
    else:
        results = sorted(results, key=lambda r: r.get("total_probe_count", 0), reverse=True)

    top_results = results[:args.top_models]
    top_ids = {r["modelId"] for r in top_results}
    top_models_full = [m for m in models if m.get("id") in top_ids]
    print(f"\nTop {len(top_models_full)} model(s) by current activity — fetching their images now:")
    for r in top_results:
        print(f"  {r['modelName'][:50]:50s} velocity/day={r.get('velocity_per_day', '?'):>8}  "
              f"total_probe_count={r.get('total_probe_count')}")

    # Step 2: fetch every meta'd image in-window for just those top models.
    entries = fetch_images_for_models(
        top_models_full, since, max_pages=args.max_pages, nsfw=args.nsfw, max_versions=args.max_versions,
    )

    if args.resolve_resources:
        load_cache()
        entries = enrich_resources(entries)
        save_cache()

    if args.top_resources:
        lora_counts, combo_counts = count_resource_usage(entries)
        if not args.resolve_resources:
            print("\n(Note: --resolve-resources wasn't used, so LoRAs below are labelled by "
                  "modelVersionId rather than name — pass --resolve-resources for readable labels.)")
        print(f"\nTop {args.top_resources} LoRA(s) by overall usage (across all {len(entries)} fetched images):")
        for lora, count in lora_counts.most_common(args.top_resources):
            print(f"  {count:>5}x  {lora}")
        print(f"\nTop {args.top_resources} checkpoint+LoRA combo(s) (different question — usage tied to a "
              f"specific checkpoint, not the LoRA's overall popularity):")
        for (checkpoint, lora), count in combo_counts.most_common(args.top_resources):
            print(f"  {count:>5}x  {checkpoint[:35]:35s} + {lora}")

        with_lora, bare_resources, no_resources, total_counts = count_bare_checkpoint_usage(entries)
        print(f"\nCheckpoint usage pattern — does this checkpoint need a LoRA, get used bare, or "
              f"lack resource metadata entirely (likely uploaded outside Civitai's own generator)?")
        print(f"  {'checkpoint':35s} {'with_lora':>12} {'bare':>9} {'no_meta':>9}   total")
        for checkpoint, total in total_counts.most_common(args.top_resources):
            wl, br, nr = with_lora.get(checkpoint, 0), bare_resources.get(checkpoint, 0), no_resources.get(checkpoint, 0)
            print(f"  {checkpoint[:35]:35s} {wl:>7}({100*wl/total:4.0f}%) {br:>5}({100*br/total:3.0f}%) "
                  f"{nr:>5}({100*nr/total:3.0f}%)   {total}")

    # Step 3: rank by reactions, client-side (see images.py docstring for why).
    if args.top_reactions_per_model:
        ranked = sort_by_reactions_per_model(entries, top_n_per_model=args.top_reactions_per_model)
    else:
        ranked = sort_by_reactions(entries, top_n=args.top_reactions or None)

    # Tag output filenames with period + run date + run time (not just date, like
    # probe.py) — images_cli runs are more likely to be re-run several times in
    # one sitting while tuning --top-models/--nsfw/etc., so a date-only stamp
    # would still silently overwrite the previous run within the same day.
    # e.g. civitai_output_week_12jul26_2214.json
    suffix = f"_{args.period.lower()}_{datetime.now().strftime('%d%b%y_%H%M').lower()}"
    out_path = OUT_PATH.replace(".json", f"{suffix}.json")
    issues_path = ISSUES_PATH.replace(".json", f"{suffix}.json")

    with open(out_path, "w") as f:
        json.dump(ranked, f, indent=2)

    issues = validate_results(ranked)
    if issues:
        with open(issues_path, "w") as f:
            json.dump(issues, f, indent=2)
        print(f"Wrote issues to {issues_path}")

    print(f"\nWrote {len(ranked)} image(s) (of {len(entries)} fetched), ranked by reactions, to {out_path}")
    if args.top_reactions_per_model:
        print(f"Top {args.top_reactions_per_model} by reactions PER MODEL:")
        last_model = None
        shown_for_model = 0
        for e in ranked:
            if e["modelName"] != last_model:
                last_model = e["modelName"]
                shown_for_model = 0
                print(f"  --- {last_model} ---")
            if shown_for_model >= 3:  # console preview only — full per-model top-N is in the output file
                continue
            print(f"    reactionScore={e['reactionScore']:>5}  {e['imageUrl']}")
            shown_for_model += 1
    else:
        print("Top 10 by reactions:")
        for e in ranked[:10]:
            print(f"  reactionScore={e['reactionScore']:>5}  {e['modelName'][:40]:40s} {e['imageUrl']}")


if __name__ == "__main__":
    main()