import argparse
import json

from .config import OUT_PATH, ISSUES_PATH
from .fetch import fetch_all
from .validate import validate_results
from .resolve import enrich_resources, load_cache, save_cache


def main():
    parser = argparse.ArgumentParser(description="Fetch recent images+metadata for popular Civitai models.")
    parser.add_argument("--model-count", type=int, default=10, help="Number of popular models to pull")
    parser.add_argument("--since-days", type=int, default=1, help="Only fetch images created in the last N days")
    parser.add_argument("--max-pages", type=int, default=20, help="Max pages to walk back per model version before giving up")
    parser.add_argument("--max-versions", type=int, default=None, help="Only query the newest N versions per model (default: all versions)")
    parser.add_argument("--types", nargs="+", default=None, help="Restrict to these Civitai model types, e.g. --types Checkpoint LORA (default: all types, including TextualInversion embeddings)")
    parser.add_argument("--max-lora-versions", type=int, default=None, help="Skip LORA-type models with more than N versions (drops style/concept bundle packs; checkpoints are exempt)")
    parser.add_argument("--nsfw", default="X", help="Civitai nsfw param (None/Soft/Mature/X/true/false). Default 'X' avoids silent NSFW filtering — see civitai/civitai#1277")
    parser.add_argument("--resolve-resources", action="store_true", help="Resolve civitaiResources modelVersionIds to names + creator usernames (extra API calls, cached per unique ID)")
    args = parser.parse_args()

    results = fetch_all(
        model_count=args.model_count,
        since_days=args.since_days,
        max_pages=args.max_pages,
        nsfw=args.nsfw,
        max_versions=args.max_versions,
        types=args.types,
        max_lora_versions=args.max_lora_versions,
    )

    if args.resolve_resources:
        load_cache()
        results = enrich_resources(results)
        save_cache()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    issues = validate_results(results)
    if issues:
        with open(ISSUES_PATH, "w") as f:
            json.dump(issues, f, indent=2)
        print(f"Wrote issues to {ISSUES_PATH}")

    print(f"Wrote {len(results)} images to {OUT_PATH}")


if __name__ == "__main__":
    main()