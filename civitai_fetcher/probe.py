"""
Probe a large pool of candidate models (by download rank) for CURRENT
activity, cheaply — one API call per model, not a full fetch.

Usage:
    uv run python -m civitai_fetcher.probe --candidate-count 100 --since-days 30

Writes:
    probe_results.csv   — modelId, modelName, type, page1_count, has_more_pages
    probe_distribution.png — sorted bar chart to eyeball the active/dead cutoff
"""

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt

from .fetch import probe_candidates

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Strips emoji/symbol characters for chart labels only — no text font renders these,
# so they just throw warnings and show as boxes. Full names are kept in the CSV.
import unicodedata

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def clean_label(text):
    # Stylized Unicode letters (bold/fraktur/double-struck "mathematical alphanumeric
    # symbols") are actual text content, not decoration — NFKD normalizes them back
    # to plain ASCII (e.g. stylized "Illustrious" -> "Illustrious") instead of
    # deleting the word entirely.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # True decorative symbols (emoji, dingbats) get dropped, not normalized
    return _EMOJI_RE.sub("", text).strip()


def main():
    parser = argparse.ArgumentParser(description="Probe candidate models for current activity.")
    parser.add_argument("--candidate-count", type=int, default=100,
                         help="Size of the download-ranked candidate pool to probe (default: 100)")
    parser.add_argument("--since-days", type=int, default=30,
                         help="Activity window to check against (default: 30)")
    parser.add_argument("--types", nargs="+", default=None,
                         help="Restrict candidate pool to these Civitai model types, e.g. --types Checkpoint LORA")
    parser.add_argument("--max-lora-versions", type=int, default=None,
                         help="Skip LORA-type candidates with more than N versions (style/concept packs)")
    parser.add_argument("--page-limit", type=int, default=50,
                         help="How many images deep to check per model (default: 50). Bump this way up "
                              "(e.g. 500) on a second pass with --only-ids to differentiate active models "
                              "from each other instead of them all tying at the cap.")
    parser.add_argument("--only-ids", nargs="+", type=int, default=None,
                         help="Probe exactly these model IDs instead of a fresh download-ranked pool — "
                              "e.g. feed in the has_more_pages=True shortlist from a first pass.")
    parser.add_argument("--nsfw", default="X")
    args = parser.parse_args()

    results = probe_candidates(candidate_count=args.candidate_count, since_days=args.since_days, nsfw=args.nsfw,
                                types=args.types, max_lora_versions=args.max_lora_versions,
                                page_limit=args.page_limit, only_ids=args.only_ids)
    df = pd.DataFrame(results)
    df["label"] = df["modelName"] + " (" + df["modelId"].astype(str) + ")"
    df = df.sort_values("download_rank").reset_index(drop=True)  # preserve real download-rank order
    df.to_csv("probe_results.csv", index=False)
    print(f"Wrote probe_results.csv ({len(df)} models, sorted by download_rank)")

    # Sanity check the dead-streak assumption: does activity actually cluster
    # by download rank, or is it scattered? Correlation near 1 = clusters well
    # (early-stopping is safe); near 0 = scattered (don't trust dead-streak cutoffs).
    corr = df["download_rank"].corr(df["page1_count"].rank())
    print(f"Spearman correlation (download_rank vs page1_count): {corr:.2f}")
    print("  (closer to -1 = activity clusters at the top of download rank, dead-streak stopping is reliable)")
    print("  (closer to 0 = scattered, don't trust early-stopping — probe the full candidate pool instead)")

    print("\nTop 20 by CURRENT activity (not download rank):")
    activity_view = df.sort_values("page1_count", ascending=False)
    print(activity_view[["label", "type", "download_rank", "page1_count", "has_more_pages"]]
          .head(20).to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_df = activity_view.reset_index(drop=True)
    colors = ["tab:green" if hm else "tab:red" for hm in plot_df["has_more_pages"]]
    ax.bar(range(len(plot_df)), plot_df["page1_count"], color=colors)
    ax.set_title(f"Model Activity Probe — page-1 image count, last {args.since_days} days\n"
                 f"(green = more pages exist beyond this count, red = this is all there is)")
    ax.set_xlabel("Models, ranked by CURRENT activity (most active → least) — not download rank")
    ax.set_ylabel(f"Images in page 1 (max {args.page_limit})")
    ax.set_xticks(range(len(plot_df)))
    df = plot_df  # keep the label-thinning code below unchanged, operating on the activity-sorted view
    n = len(df)
    if n <= 60:
        ax.set_xticklabels(df["label"].apply(clean_label), rotation=90, fontsize=6)
    else:
        # too many bars to label individually — only label every Nth one so it stays readable
        step = max(1, n // 40)
        labels = [clean_label(lbl) if i % step == 0 else "" for i, lbl in enumerate(df["label"])]
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        print(f"  ({n} models — thinned x-axis labels to every {step}th; full names are in probe_results.csv)")
    plt.tight_layout()
    plt.savefig("probe_distribution.png", dpi=150)
    plt.close()
    print("Wrote probe_distribution.png")


if __name__ == "__main__":
    main()