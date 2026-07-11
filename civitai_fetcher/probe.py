import argparse
import pandas as pd
import matplotlib.pyplot as plt
from civitai_fetcher.fetch import probe_candidates, add_velocity

def clean_label(label_str):
    """
    Truncates exceptionally long names for clean chart presentation.
    """
    if len(label_str) > 25:
        return label_str[:22] + "..."
    return label_str

def main():
    parser = argparse.ArgumentParser(description="Probe candidate models for current activity.")
    parser.add_argument("--candidate-count", type=int, default=1000,
                         help="Size of the download-ranked candidate pool to probe (default: 1000)")
    parser.add_argument("--period", default="Month", choices=["Day", "Week", "Month", "Year", "AllTime"],
                         help="Timeframe window for download popularity ranking (default: Month)")
    parser.add_argument("--since-days", type=int, default=30,
                         help="Activity window to check images against (default: 30)")
    parser.add_argument("--types", nargs="+", default="Checkpoint",
                         help="Restrict candidate pool to these Civitai model types, e.g. --types Checkpoint Lora")
    parser.add_argument("--max-lora-versions", type=int, default=None,
                         help="Skip LORA-type candidates with more than N versions")
    parser.add_argument("--page-limit", type=int, default=100,
                         help="How many images deep to check per model on the Tier 1 pass (default: 100)")
    parser.add_argument("--deep-probe-limit", type=int, default=150,
                         help="Max items to check for green models across an adaptive Tier 2 page (default: 150)")
    parser.add_argument("--only-ids", nargs="+", type=int, default=None,
                         help="Probe exactly these model IDs instead of a fresh download-ranked pool.")
    parser.add_argument("--nsfw", default="X")
    parser.add_argument("--velocity-top-n", type=int, default=150,
                         help="Tier 3b: measure sustained daily post velocity (day-bucketed, not a raw count) "
                              "for this many top models by total_probe_count (default: 150, 0 = off)")
    parser.add_argument("--velocity-window-days", type=int, default=3,
                         help="Window size in days for velocity measurement (default: 3)")
    parser.add_argument("--velocity-max-pages", type=int, default=400,
                         help="Hard safety cap on pages fetched per model during velocity measurement (default: 400)")
    args = parser.parse_args()

    results, models, since = probe_candidates(candidate_count=args.candidate_count, since_days=args.since_days, period=args.period,
                                nsfw=args.nsfw, types=args.types, max_lora_versions=args.max_lora_versions,
                                page_limit=args.page_limit, deep_probe_limit=args.deep_probe_limit,
                                only_ids=args.only_ids)

    capped = sum(1 for r in results if r.get("has_more_pages"))
    if capped:
        print(f"\n⚠️  {capped} model(s) hit the Tier 2 --deep-probe-limit ({args.deep_probe_limit}) and are "
              f"tied at that value in total_probe_count — their real ranking is unresolved.\n"
              f"   Use --velocity-top-n (on by default) to resolve them by day-bucketed velocity instead of "
              f"raising --deep-probe-limit — a bigger count cap does NOT reliably resolve this (see README).\n")

    if args.velocity_top_n:
        results = add_velocity(results, models, args.velocity_top_n, args.page_limit, args.nsfw,
                                window_days=args.velocity_window_days, max_pages=args.velocity_max_pages)
        maxed = sum(1 for r in results if r.get("velocity_probe_status") == "max_pages_hit")
        if maxed:
            print(f"\n⚠️  {maxed} model(s) hit --velocity-max-pages ({args.velocity_max_pages}) while measuring "
                  f"velocity — their velocity_per_day is a lower bound, not exact. Raise --velocity-max-pages "
                  f"if you need the precise number for these.\n")

    print()  # newline after the run of '.'/'X'/'o' progress marks
    df = pd.DataFrame(results)
    
    if df.empty:
        print("❌ No models matched the given filters. Output files will not be generated.")
        return

    df["label"] = df["modelName"] + " (" + df["modelId"].astype(str) + ")"
    df = df.sort_values("download_rank").reset_index(drop=True)
    df.to_csv("probe_results.csv", index=False)
    print(f"Wrote probe_results.csv ({len(df)} models, sorted by download_rank over the last {args.period})")

    corr = df["download_rank"].corr(df["total_probe_count"].rank())
    print(f"Spearman correlation (download_rank vs total_probe_count): {corr:.2f}")

    print(f"\nTop 20 by CURRENT activity (Initial pool sorted by {args.period} downloads):")
    activity_view = df.sort_values("total_probe_count", ascending=False)
    print(activity_view[["label", "type", "download_rank", "page1_count", "total_probe_count", "has_more_pages"]]
          .head(20).to_string(index=False))

    if "velocity_per_day" in df.columns:
        vel_view = df[df["velocity_per_day"].notna()].sort_values("velocity_per_day", ascending=False)
        print(f"\nTop 20 by sustained {args.velocity_window_days}-day velocity "
              f"(burst_ratio: peak single day vs the {args.velocity_window_days}-day average — "
              f"high burst_ratio = spiky/viral, not sustained):")
        print(vel_view[["label", "velocity_per_day", "max_single_day", "burst_ratio", "window_total"]]
              .head(20).to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_df = activity_view.reset_index(drop=True)
    colors = ["tab:green" if hm else "tab:red" for hm in plot_df["has_more_pages"]]
    
    ax.bar(range(len(plot_df)), plot_df["total_probe_count"], color=colors)
    ax.set_title(f"Model Activity Probe — Initial Pool: Top Downloads ({args.period})\n"
                 f"(green = more pages exist beyond deep cap, red = volume fully captured)")
    ax.set_xlabel("Models, ranked by CURRENT activity (most active → least)")
    ax.set_ylabel(f"Images Captured (Tier 1 Cap: {args.page_limit}, Tier 2 Cap: {args.deep_probe_limit})")
    
    n = len(plot_df)
    if n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(plot_df["label"].apply(clean_label), rotation=90, fontsize=6)
    else:
        step = max(1, n // 40)
        tick_positions = [i for i in range(n) if i % step == 0]
        labels = [clean_label(plot_df["label"].iloc[i]) for i in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        print(f"  ({n} models — thinned x-axis labels to every {step}th for plot scannability)")
        
    plt.tight_layout()
    plt.savefig("probe_distribution.png", dpi=150)
    plt.close()
    print("Wrote probe_distribution.png")

if __name__ == "__main__":
    main()