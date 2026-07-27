def validate_entry(entry):
    """Return a list of issue strings for one entry, empty if clean."""
    issues = []

    # missing expected static fields — modelId/modelUrl are deliberately
    # excluded here. They're always populated for scope=models entries
    # (set directly from the model dict) and always None for scope=uploads
    # entries (there's no owning model by definition — see creator_cli.py's
    # --scope uploads). So their absence is never actually an anomaly with
    # the current fetch pipeline, just noise from a shape that legitimately
    # doesn't have that field.
    for field in ("imageId", "imageUrl"):
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