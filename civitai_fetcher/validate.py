def validate_entry(entry):
    """Return a list of issue strings for one entry, empty if clean."""
    issues = []
    meta = entry.get("meta") or {}

    # size mismatch: meta.Size (generation-time) vs actual file width/height
    # these legitimately differ when an image is upscaled/resized after generation
    meta_size = meta.get("Size")
    if meta_size and "x" in str(meta_size):
        try:
            meta_w, meta_h = (int(x) for x in str(meta_size).lower().split("x"))
            if entry.get("width") != meta_w or entry.get("height") != meta_h:
                issues.append(
                    f"size_mismatch: file={entry.get('width')}x{entry.get('height')} "
                    f"meta={meta_w}x{meta_h}"
                )
        except ValueError:
            issues.append(f"unparseable_meta_size: {meta_size!r}")

    # missing expected static fields
    for field in ("imageId", "imageUrl", "modelId", "modelUrl"):
        if not entry.get(field):
            issues.append(f"missing_{field}")

    # meta present but empty dict (shouldn't happen with withMeta=true, but check anyway)
    if entry.get("meta") == {}:
        issues.append("empty_meta_dict")

    # postId present but postUrl missing (or vice versa)
    if bool(entry.get("postId")) != bool(entry.get("postUrl")):
        issues.append("postId_postUrl_mismatch")

    return issues


def validate_results(results):
    """Run validation over all entries, print a summary, return issues keyed by imageId."""
    all_issues = {}
    for entry in results:
        issues = validate_entry(entry)
        if issues:
            all_issues[entry.get("imageId")] = issues

    # duplicate imageId check across the whole set
    seen = {}
    for entry in results:
        seen[entry.get("imageId")] = seen.get(entry.get("imageId"), 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}

    print(f"\nValidation: {len(all_issues)}/{len(results)} entries flagged")
    for image_id, issues in list(all_issues.items())[:10]:
        print(f"  image {image_id}: {issues}")
    if len(all_issues) > 10:
        print(f"  ... and {len(all_issues) - 10} more")
    if dupes:
        print(f"  duplicate imageIds: {dupes}")

    return all_issues
