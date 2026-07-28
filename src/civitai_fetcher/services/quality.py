"""
Quality gate for noisy fetch sources — pure functions over already-fetched
image entries, no network calls.

This is deliberately NOT applied inside creator_cli/model_cli's pipeline:
a named creator/model is already a curated, high-signal source, so gating
it on reaction score would just be discarding legitimate low-reaction work
from a good source. It IS meant for images_cli's activity-ranked pipeline,
where the candidate pool is drawn from the general firehose and is mostly
noise — reaction score is how we tell the small fraction anyone actually
cared about apart from the rest.

Split from images.py: these are a different responsibility (filtering/
ranking) to fetch.py (network/pagination), and count_resource_usage /
count_bare_checkpoint_usage are a third responsibility again (usage
statistics, not quality) — kept in the same file for now since they're
small and share no state, but worth re-splitting if either grows.
"""
from collections import Counter


def reaction_score(stats):
    """
    Combine an image's stats dict into one sortable 'reactions' number.
    Sums whichever reaction-count fields are present rather than assuming
    a fixed schema, since Civitai's /images stats object composition has
    varied slightly across API responses.
    """
    if not stats:
        return 0
    keys = ("likeCount", "heartCount", "laughCount", "cryCount", "commentCount")
    return sum(stats.get(k, 0) or 0 for k in keys)


def filter_by_min_reactions(entries, min_score=1):
    """
    The actual noise gate: drop entries below a reaction-score floor.
    Distinct from sort_by_reactions' top_n, which caps how many survive
    regardless of score — this instead drops everything below a threshold
    regardless of how many that leaves. Even min_score=1 removes the bulk
    of zero-reaction uploads nobody has looked at twice.
    """
    return [e for e in entries if e.get("reactionScore", 0) >= min_score]


def sort_by_reactions(entries, top_n=None):
    """
    Sort fetched image entries by combined reaction score, descending.
    Entries already carry a precomputed 'reactionScore' field from
    fetch_model_images; this just orders (and optionally truncates) by it.
    """
    ranked = sorted(entries, key=lambda e: e.get("reactionScore", 0), reverse=True)
    return ranked[:top_n] if top_n else ranked


def sort_by_reactions_per_model(entries, top_n_per_model=10):
    """
    Like sort_by_reactions, but caps the top N PER MODEL instead of one
    global top N — a flat global cutoff always gets crowded out by whichever
    model has the single highest reaction counts (Kreamania style outliers),
    leaving other models with zero representation in the output. Capping
    per-model instead guarantees every fetched model gets a fair, comparable
    slice, which is the point if you're comparing model/checkpoint types
    rather than just chasing the single best image overall.

    Returns entries grouped by model (each model's own images sorted by
    reaction score, descending), models ordered by their own top image's
    reaction score — so the highest-reaction model still appears first, but
    every model gets its full top_n_per_model regardless of how it compares
    to others.
    """
    by_model = {}
    for e in entries:
        by_model.setdefault(e.get("modelId"), []).append(e)

    per_model_top = {
        model_id: sorted(model_entries, key=lambda e: e.get("reactionScore", 0), reverse=True)[:top_n_per_model]
        for model_id, model_entries in by_model.items()
    }
    ordered_model_ids = sorted(
        per_model_top, key=lambda mid: per_model_top[mid][0].get("reactionScore", 0) if per_model_top[mid] else 0,
        reverse=True,
    )
    result = []
    for mid in ordered_model_ids:
        result.extend(per_model_top[mid])
    return result


def aggregate_by_post(entries, method="max"):
    """
    Group entries by postId and aggregate their reactionScore into one
    post-level number, keyed by postId.

    Why this exists: a reaction lands on one image, but people don't
    reliably like every image in a post they liked — spread-the-love
    behaviour (Civitai's like rewards are capped, so users spread likes
    across posts for diversity rather than piling onto one) means a
    single image's raw score is partly an artefact of which image in
    the post they happened to click, not a clean "this image is best"
    signal. The post is the more robust unit for "did this generation
    land", even though individual images still carry the final ranking.

    method="max" (default): the post's score is its single best image's
    score. Preferred over sum for gating/triage — sum lets a large
    variant-dump post (many mediocre images) outrank a post with one
    genuinely strong image, just by volume. Measured on real creator
    data: multi-image posts show a "winner takes most" pattern (top
    image averages ~70% of the post's total score) except in large
    dump-style posts, where max avoids rewarding volume over quality.

    method="sum": total across all images in the post. Available for
    comparison/analysis, not recommended as the default gate.

    Returns {postId: aggregate_score}. Entries without a postId are
    ignored (nothing to group them with).
    """
    by_post = {}
    for e in entries:
        post_id = e.get("postId")
        if post_id is None:
            continue
        by_post.setdefault(post_id, []).append(e.get("reactionScore", 0))

    if method == "sum":
        return {pid: sum(scores) for pid, scores in by_post.items()}
    if method == "max":
        return {pid: max(scores) for pid, scores in by_post.items()}
    raise ValueError(f"Unknown aggregation method: {method!r} (expected 'max' or 'sum')")


def count_resource_usage(entries):
    """
    Tally civitaiResources usage across a set of fetched images — two
    different questions, kept as two separate counters rather than one:

      lora_counts:  how often each LoRA appears overall, regardless of which
                    checkpoint it was paired with. Answers "what's the most
                    popular LoRA right now" across the whole fetched set.

      combo_counts: how often each (checkpoint, LoRA) PAIR appears together.
                    Answers "what's most used specifically with THIS
                    checkpoint" — a LoRA can rank high overall but be tied to
                    one checkpoint, or split its usage across several; this
                    is what tells them apart.

    Uses resolved names (from enrichment.resolve.enrich_resources, i.e.
    --resolve-resources) when available; falls back to the raw
    modelVersionId as the label if the entry was never enriched, so this
    still works either way — just less readable without resolution.
    """
    lora_counts = Counter()
    combo_counts = Counter()
    for entry in entries:
        checkpoint_name = entry.get("modelName", "Unknown")
        meta = entry.get("meta") or {}
        for res in meta.get("civitaiResources") or []:
            if res.get("type") != "lora":
                continue
            lora_label = res.get("name") or f"modelVersionId:{res.get('modelVersionId')}"
            lora_counts[lora_label] += 1
            combo_counts[(checkpoint_name, lora_label)] += 1
    return lora_counts, combo_counts


def count_bare_checkpoint_usage(entries):
    """
    A third, different question again from count_resource_usage: not "which
    LoRA is popular" but "does this checkpoint even get used WITH a LoRA at
    all, or mostly raw?" Some checkpoints may only look good/produce their
    popular images once paired with a specific LoRA (near-zero bare usage);
    others may be popular entirely on their own (mostly bare). Neither
    lora_counts nor combo_counts answers this — both are silent on
    checkpoints that show up with nothing attached.

    Splits into THREE buckets per checkpoint, not two, because "no LoRA" is
    ambiguous on its own:
      with_lora:      at least one lora/embedding resource attached — genuinely
                       used together.
      bare_resources: civitaiResources is present and non-empty, but contains
                       ONLY the checkpoint (no lora/embedding) — genuinely used
                       on its own.
      no_resources:   civitaiResources is missing or empty entirely. This is
                       NOT the same claim as bare_resources — it usually means
                       the image wasn't generated through Civitai's own on-site
                       generator (e.g. uploaded from A1111/ComfyUI, which don't
                       populate this field), so we genuinely can't tell whether
                       a LoRA was used. Lumping this in with bare_resources
                       would overstate how often a checkpoint is used "raw."

    Returns (with_lora, bare_resources, no_resources, total) — four Counters,
    all keyed by checkpoint name.
    """
    with_lora = Counter()
    bare_resources = Counter()
    no_resources = Counter()
    total = Counter()
    for entry in entries:
        checkpoint_name = entry.get("modelName", "Unknown")
        meta = entry.get("meta") or {}
        resources = meta.get("civitaiResources") or []
        total[checkpoint_name] += 1
        if not resources:
            no_resources[checkpoint_name] += 1
            continue
        has_addon = any(r.get("type") in ("lora", "embedding", "textualinversion") for r in resources)
        if has_addon:
            with_lora[checkpoint_name] += 1
        else:
            bare_resources[checkpoint_name] += 1
    return with_lora, bare_resources, no_resources, total
