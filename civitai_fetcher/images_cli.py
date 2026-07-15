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



# Per-period filter thresholds.
# velocity_per_day is already normalised so the same floor applies across periods.
# probe threshold scales with "consistently active across the period":
#   Day   >= 5   (active today)
#   Week  >= 30  (active most of the week)
#   Month >= 100 (3+ solid weeks)
PERIOD_FILTERS = {
    "Day":   {"min_velocity": 5, "min_probe": 5},
    "Week":  {"min_velocity": 5, "min_probe": 30},
    "Month": {"min_velocity": 5, "min_probe": 100},
}

# velocity window by period — Day uses 1 day, everything else 3 days
PERIOD_VELOCITY_WINDOW = {
    "Day": 1,
    "Week": 3,
    "Month": 3,
}


def main():
    parser = argparse.ArgumentParser(
        description="Find currently-active popular Civitai models, then fetch their images ranked by reactions."
    )
    parser.add_argument("--period", default=IMAGES_PERIOD, choices=["Day", "Week", "Month"],
                        help=f"Window to rank model popularity/activity over (default: {IMAGES_PERIOD})")
    parser.add_argument("--top-reactions-per-model", type=int, default=50,
                        help="Keep the top N images PER MODEL ranked by community reactions (default: 50). "
                             "Every active model contributes equally regardless of overall popularity — "
                             "prevents one dominant checkpoint crowding out the rest. "
                             "Set to 0 to switch to global ranking via --top-reactions instead.")
    parser.add_argument("--top-reactions", type=int, default=IMAGES_TOP_REACTIONS,
                        help="Keep the top N images GLOBALLY across all models (default: 30). "
                             "Only used when --top-reactions-per-model is 0. "
                             "Warning: a single high-reaction model can dominate the output.")

    # --- internal / advanced flags (hidden from --help) ---
    SUPPRESS = argparse.SUPPRESS
    parser.add_argument("--since-days", type=int, default=IMAGES_SINCE_DAYS, help=SUPPRESS)
    parser.add_argument("--candidate-count", type=int, default=PROBE_CANDIDATE_COUNT, help=SUPPRESS)
    parser.add_argument("--types", nargs="+", default=[PROBE_TYPES], help=SUPPRESS)
    parser.add_argument("--max-lora-versions", type=int, default=None, help=SUPPRESS)
    parser.add_argument("--page-limit", type=int, default=PROBE_PAGE_LIMIT, help=SUPPRESS)
    parser.add_argument("--deep-probe-limit", type=int, default=PROBE_DEEP_PROBE_LIMIT, help=SUPPRESS)
    parser.add_argument("--velocity-max-pages", type=int, default=PROBE_VELOCITY_MAX_PAGES, help=SUPPRESS)
    parser.add_argument("--max-pages", type=int, default=IMAGES_MAX_PAGES, help=SUPPRESS)
    parser.add_argument("--max-versions", type=int, default=None, help=SUPPRESS)
    parser.add_argument("--nsfw", default=IMAGES_NSFW, help=SUPPRESS)
    parser.add_argument("--top-resources", type=int, default=15, help=SUPPRESS)
    parser.add_argument("--resolve-resources", dest="resolve_resources", action="store_true", default=True, help=SUPPRESS)
    parser.add_argument("--no-resolve-resources", dest="resolve_resources", action="store_false", help=SUPPRESS)
    # override filters (escape hatch — normally derived from --period)
    parser.add_argument("--min-velocity", type=float, default=None, help=SUPPRESS)
    parser.add_argument("--min-probe", type=int, default=None, help=SUPPRESS)
    args = parser.parse_args()

    # Derive per-period defaults
    thresholds = PERIOD_FILTERS.get(args.period, PERIOD_FILTERS["Week"])
    min_velocity = args.min_velocity if args.min_velocity is not None else thresholds["min_velocity"]
    min_probe = args.min_probe if args.min_probe is not None else thresholds["min_probe"]
    velocity_window_days = PERIOD_VELOCITY_WINDOW.get(args.period, 3)

    # Step 1: rank models by download popularity, then activity (Tier 1/2 + Tier 3b).
    results, models, since = probe_candidates(
        candidate_count=args.candidate_count, since_days=args.since_days, period=args.period,
        nsfw=args.nsfw, types=args.types, max_lora_versions=args.max_lora_versions,
        page_limit=args.page_limit, deep_probe_limit=args.deep_probe_limit,
    )
    results = add_velocity(
        results, models, top_n=len(results),
        page_limit=args.page_limit, nsfw=args.nsfw,
        window_days=velocity_window_days, max_pages=args.velocity_max_pages,
    )
    results = sorted(results, key=lambda r: r.get("velocity_per_day", 0), reverse=True)

    # Filter to consistently active models only
    before = len(results)
    results = [r for r in results
               if r.get("velocity_per_day", 0) >= min_velocity
               and r.get("total_probe_count", 0) >= min_probe]
    print(f"Activity filter (velocity >= {min_velocity}/day, probe >= {min_probe}): "
          f"{len(results)} of {before} models kept")

    top_results = results
    top_ids = {r["modelId"] for r in top_results}
    top_models_full = [m for m in models if m.get("id") in top_ids]
    print(f"\n{len(top_models_full)} model(s) by current activity — fetching their images now:")
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

    import pathlib; pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
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