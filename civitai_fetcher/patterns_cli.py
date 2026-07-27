"""
Postprocessing: given one or more already-downloaded output JSON files
(from creator_cli.py / model_cli.py / images_cli.py), find repeated
workflows/patterns per creator (posterUsername) — the idea being that a
creator tends to land on a recipe (a checkpoint+LoRA stack, a settings
combo, a prompt template) and then reuse it across many posts, rather than
generating every image from scratch.

Deliberately standalone from client.py/images.py — this only reads JSON
already on disk, no network calls, no Civitai API involved at all.

Grouping is by posterUsername, not by input file: a --scope models file
(model_cli.py, or creator_cli.py --scope models) legitimately contains many
different posters using the same model, so grouping by file would blur
different people's workflows together under one label.

Three independent pattern signals, each answering a different question:

  1. Resource stack — which checkpoint+LoRA combination does this creator
     reuse? (built on the same civitaiResources shape as
     images.count_resource_usage, but read from disk rather than a live
     fetch, and grouped per-poster rather than globally).
  2. Settings recipe — sampler/steps/cfgScale/Size/scheduler combo reused
     across posts. A creator landing on "DPM++ 2M, 30 steps, cfg 7,
     896x1152" repeatedly is a workflow signal independent of what
     resources they used.
  3. Prompt template — exact-duplicate prompt/negativePrompt text reused
     across posts (a copy-pasted or lightly-varied template). Deliberately
     EXACT match only, not fuzzy — a fuzzy near-duplicate detector would
     need to reproduce/compare substantial prompt text against itself,
     which is easy to get wrong quietly (false "same template" merges);
     exact match is a strict floor that's always correct when it fires.

Use:
    uv run python -m civitai_fetcher.patterns_cli output/*.json
    uv run python -m civitai_fetcher.patterns_cli output/wai0731_uploads.json --min-images 5
"""
import argparse
import json
from collections import Counter, defaultdict


def _load_entries(paths):
    """Load and concatenate entries from one or more output JSON files.
    Dedupes by imageId across files (the same image can legitimately show
    up in more than one input file, e.g. a creator scan and a model scan
    overlapping) so it isn't double-counted into any pattern's frequency."""
    seen_ids = set()
    entries = []
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  skipped {path}: {e}")
            continue
        n_before = len(entries)
        for e in data:
            iid = e.get("imageId")
            if iid is not None and iid in seen_ids:
                continue
            if iid is not None:
                seen_ids.add(iid)
            entries.append(e)
        print(f"  {path}: {len(data)} entries ({len(entries) - n_before} new)")
    return entries


def _resource_stack(entry):
    """(checkpoint_name, sorted tuple of LoRA/embedding names) for one entry.
    Sorted so the same stack in a different attachment order still counts
    as the same pattern. Uses resolved 'name' when present (see resolve.py
    --resolve-resources), falls back to the raw modelVersionId otherwise —
    still works, just less readable."""
    checkpoint = entry.get("modelName") or "Unknown"
    meta = entry.get("meta") or {}
    addons = []
    for res in meta.get("civitaiResources") or []:
        if res.get("type") not in ("lora", "embedding", "textualinversion"):
            continue
        addons.append(res.get("name") or f"modelVersionId:{res.get('modelVersionId')}")
    return checkpoint, tuple(sorted(addons))


def _settings_recipe(entry):
    """(sampler, steps, cfgScale, Size, scheduler) for one entry — None for
    any field that's missing, so entries with partial meta still group
    together on whatever fields they do share rather than being dropped."""
    meta = entry.get("meta") or {}
    return (
        meta.get("sampler"),
        meta.get("steps"),
        meta.get("cfgScale"),
        meta.get("Size"),
        meta.get("scheduler"),
    )


def analyse_poster(entries, top_n=5):
    """Run all three pattern signals for one poster's entries. Returns a
    dict summary (also used for the optional JSON output)."""
    total = len(entries)
    with_meta = [e for e in entries if e.get("meta")]

    stack_counts = Counter(_resource_stack(e) for e in with_meta)
    # drop the "no addons at all" bucket from the *headline* ranking — a bare
    # checkpoint with an empty tuple isn't a "workflow", it just means no
    # LoRA/embedding was attached that time. Still counted, just not reported
    # as if it were a deliberate recipe.
    stack_counts_nonempty = Counter({k: v for k, v in stack_counts.items() if k[1]})

    recipe_counts = Counter(_settings_recipe(e) for e in with_meta
                             if any(_settings_recipe(e)))

    prompt_counts = Counter(e["meta"]["prompt"] for e in with_meta if e["meta"].get("prompt"))
    neg_counts = Counter(e["meta"]["negativePrompt"] for e in with_meta if e["meta"].get("negativePrompt"))

    return {
        "total_entries": total,
        "entries_with_meta": len(with_meta),
        "top_resource_stacks": stack_counts_nonempty.most_common(top_n),
        "top_settings_recipes": recipe_counts.most_common(top_n),
        "top_repeated_prompts": [(p, c) for p, c in prompt_counts.most_common(top_n) if c > 1],
        "top_repeated_negative_prompts": [(p, c) for p, c in neg_counts.most_common(top_n) if c > 1],
    }


def _fmt_stack(stack):
    checkpoint, addons = stack
    if not addons:
        return checkpoint
    return f"{checkpoint} + {', '.join(addons)}"


def _fmt_recipe(recipe):
    sampler, steps, cfg, size, scheduler = recipe
    parts = []
    if sampler: parts.append(sampler)
    if scheduler: parts.append(scheduler)
    if steps is not None: parts.append(f"{steps} steps")
    if cfg is not None: parts.append(f"cfg {cfg}")
    if size: parts.append(str(size))
    return ", ".join(parts) if parts else "(no settings fields present)"


def print_poster_summary(username, summary, prompt_preview_chars=70):
    total = summary["total_entries"]
    print(f"\n=== {username} — {total} image(s), {summary['entries_with_meta']} with meta ===")

    if summary["top_resource_stacks"]:
        print("  Resource stacks (checkpoint + LoRA/embedding combo):")
        for stack, count in summary["top_resource_stacks"]:
            pct = 100 * count / total
            print(f"    {count:>4}x ({pct:4.1f}%)  {_fmt_stack(stack)}")
    else:
        print("  Resource stacks: none (no meta'd images with an attached LoRA/embedding)")

    if summary["top_settings_recipes"]:
        print("  Settings recipes (sampler/steps/cfg/size/scheduler):")
        for recipe, count in summary["top_settings_recipes"]:
            pct = 100 * count / total
            print(f"    {count:>4}x ({pct:4.1f}%)  {_fmt_recipe(recipe)}")

    if summary["top_repeated_prompts"]:
        print("  Repeated prompts (exact match):")
        for prompt, count in summary["top_repeated_prompts"]:
            preview = prompt[:prompt_preview_chars].replace("\n", " ")
            ellipsis = "..." if len(prompt) > prompt_preview_chars else ""
            print(f"    {count:>4}x  \"{preview}{ellipsis}\"")

    if summary["top_repeated_negative_prompts"]:
        print("  Repeated negative prompts (exact match):")
        for prompt, count in summary["top_repeated_negative_prompts"]:
            preview = prompt[:prompt_preview_chars].replace("\n", " ")
            ellipsis = "..." if len(prompt) > prompt_preview_chars else ""
            print(f"    {count:>4}x  \"{preview}{ellipsis}\"")

    if not (summary["top_resource_stacks"] or summary["top_settings_recipes"]
            or summary["top_repeated_prompts"] or summary["top_repeated_negative_prompts"]):
        print("  No repeated pattern found — this creator's posts look one-off rather than a reused workflow.")


def main():
    parser = argparse.ArgumentParser(
        description="Find repeated per-creator workflows/patterns (resource stacks, settings recipes, "
                    "prompt templates) across one or more downloaded civitai_fetcher output JSON files."
    )
    parser.add_argument("files", nargs="+", help="Output JSON file(s) to analyse (supports shell globs).")
    parser.add_argument("--min-images", type=int, default=3,
                        help="Skip posters with fewer than this many images — not enough data to call "
                             "anything a repeated pattern (default: 3).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many top patterns to show per signal, per poster (default: 5).")
    parser.add_argument("--sort-by", choices=["images", "top-stack", "username"], default="images",
                        help="How to order posters in the output: total image count (default), "
                             "strength of their single most-repeated resource stack, or alphabetically.")
    parser.add_argument("--out", default=None,
                        help="Optional path to also write the full summary as JSON (default: print only).")
    args = parser.parse_args()

    print(f"Loading {len(args.files)} file(s)...")
    entries = _load_entries(args.files)
    print(f"Total unique entries: {len(entries)}")

    by_poster = defaultdict(list)
    unattributed = 0
    for e in entries:
        username = e.get("posterUsername")
        if not username:
            unattributed += 1
            continue
        by_poster[username].append(e)
    if unattributed:
        print(f"  ({unattributed} entries had no posterUsername — excluded from per-poster analysis)")

    before = len(by_poster)
    posters = {u: es for u, es in by_poster.items() if len(es) >= args.min_images}
    if len(posters) < before:
        print(f"Dropped {before - len(posters)} poster(s) below --min-images {args.min_images}")

    summaries = {u: analyse_poster(es, top_n=args.top_n) for u, es in posters.items()}

    if args.sort_by == "images":
        order = sorted(summaries, key=lambda u: summaries[u]["total_entries"], reverse=True)
    elif args.sort_by == "top-stack":
        order = sorted(
            summaries,
            key=lambda u: summaries[u]["top_resource_stacks"][0][1] if summaries[u]["top_resource_stacks"] else 0,
            reverse=True,
        )
    else:
        order = sorted(summaries)

    print(f"\n{len(order)} poster(s) with >= {args.min_images} image(s):")
    for username in order:
        print_poster_summary(username, summaries[username])

    if args.out:
        import pathlib
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # tuples aren't valid JSON keys/values in some shapes (settings recipes are
        # tuples) — json.dump handles tuples-as-lists fine on the value side, this
        # is just here as a comment so it's clear that round-tripping this file
        # back through json.load will give lists, not tuples, back.
        with open(args.out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote full summary to {args.out}")


if __name__ == "__main__":
    main()
