import argparse
import json

from .config import OUT_PATH, ISSUES_PATH
from .fetch import fetch_all
from .validate import validate_results
from .resolve import enrich_resources, load_cache, save_cache


def main():
    parser = argparse.ArgumentParser(description="Fetch recent images+metadata for popular Civitai models.")
    parser.add_argument("--model-count", type=int, default=10, help="Number of popular models to pull")
    parser.add_argument("--images-per-model", type=int, default=20, help="Recent images per model")
    parser.add_argument("--max-pages", type=int, default=5, help="Max pages to walk back per model if page 1 is short on metadata")
    parser.add_argument("--nsfw", default="X", help="Civitai nsfw param (None/Soft/Mature/X/true/false). Default 'X' avoids silent NSFW filtering — see civitai/civitai#1277")
    parser.add_argument("--resolve-resources", action="store_true", help="Resolve civitaiResources modelVersionIds to names + creator usernames (extra API calls, cached per unique ID)")
    args = parser.parse_args()

    results = fetch_all(
        model_count=args.model_count,
        images_per_model=args.images_per_model,
        max_pages=args.max_pages,
        nsfw=args.nsfw,
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
