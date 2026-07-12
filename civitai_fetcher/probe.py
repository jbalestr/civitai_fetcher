import argparse
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from civitai_fetcher.fetch import probe_candidates, add_velocity, _log
import time
from civitai_fetcher.config import (
    PROBE_CANDIDATE_COUNT, PROBE_PERIOD, PROBE_SINCE_DAYS, PROBE_TYPES,
    PROBE_PAGE_LIMIT, PROBE_DEEP_PROBE_LIMIT, PROBE_NSFW,
    PROBE_VELOCITY_TOP_N, PROBE_VELOCITY_WINDOW_DAYS, PROBE_VELOCITY_MAX_PAGES,
)

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

def clean_label(label_str):
    """
    Truncates exceptionally long names for clean chart presentation.

    Also strips non-ASCII characters (CJK, Japanese, Korean, etc.) — matplotlib's
    default font (DejaVu Sans) can't render most of these, and leaving them in
    triggers a "Glyph missing" UserWarning per unique glyph, every run, for any
    model whose name contains them. This only affects the PNG chart's x-axis
    labels; the real model name is preserved untouched in the CSV and HTML report.
    """
    label_str = _NON_ASCII_RE.sub("", label_str).strip()
    if not label_str:
        label_str = "(unnamed)"
    if len(label_str) > 25:
        return label_str[:22] + "..."
    return label_str

def write_html_report(df, args, path="probe_report.html"):
    """
    A raw CSV needs a spreadsheet; the console top-20 tables and PNG only show
    a slice/shape. This writes one self-contained HTML file (no CDN, no
    internet needed) with every probed model in a sortable, filterable table —
    click a header to sort, type in the box to filter by name/type/baseModel.
    """
    cols = ["download_rank", "label", "type", "baseModel", "page1_count", "total_probe_count",
            "has_more_pages", "window_total", "velocity_per_day", "max_single_day",
            "burst_ratio", "probe_status", "velocity_probe_status"]
    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy()

    # Flag models that tied at the Tier 2 cap — these are the ones whose real
    # ranking depends on velocity (or didn't get resolved if velocity was off/skipped).
    if "total_probe_count" in view.columns:
        capped = view["total_probe_count"] == args.deep_probe_limit
        resolved = view["velocity_per_day"].notna() if "velocity_per_day" in view.columns else False
        view["tier_status"] = "normal"
        view.loc[capped, "tier_status"] = "cap_tied_unresolved"
        view.loc[capped & resolved, "tier_status"] = "cap_tied_resolved_by_velocity"

    records = json.loads(view.to_json(orient="records"))
    columns = list(view.columns)

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Civitai Probe Report</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 20px; background: #111; color: #ddd; }
  h1 { font-size: 18px; }
  #meta { color: #999; margin-bottom: 10px; font-size: 13px; }
  #filter { padding: 6px 10px; width: 320px; margin-bottom: 10px; background: #222; color: #eee; border: 1px solid #444; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { padding: 5px 10px; border-bottom: 1px solid #333; text-align: left; white-space: nowrap; }
  th { cursor: pointer; background: #1a1a1a; position: sticky; top: 0; user-select: none; }
  th:hover { background: #2a2a2a; }
  th.sorted-asc::after { content: " \\25B2"; }
  th.sorted-desc::after { content: " \\25BC"; }
  tr:hover { background: #1c2430; }
  .tag { padding: 1px 6px; border-radius: 3px; font-size: 11px; }
  .tag-green { background: #14361f; color: #6fdc8c; }
  .tag-red { background: #3a1616; color: #e88; }
  .tag-cap { background: #3a3316; color: #e8d488; }
  .tag-vel { background: #16303a; color: #7fd0e8; }
</style></head>
<body>
<h1>Civitai Probe Report</h1>
<div id="meta">__META__</div>
<input id="filter" placeholder="Filter by name / type / baseModel...">
<table id="tbl"><thead><tr>__HEADERS__</tr></thead><tbody></tbody></table>
<script>
const DATA = __DATA__;
const COLUMNS = __COLUMNS__;
let sortCol = "total_probe_count", sortDir = -1;

function fmtCell(col, v) {
  if (v === null || v === undefined) return "";
  if (col === "has_more_pages") return v ? '<span class="tag tag-green">more pages</span>' : '<span class="tag tag-red">fully captured</span>';
  if (col === "tier_status") {
    if (v === "cap_tied_resolved_by_velocity") return '<span class="tag tag-vel">resolved by velocity</span>';
    if (v === "cap_tied_unresolved") return '<span class="tag tag-cap">cap-tied, unresolved</span>';
    return "";
  }
  if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(2);
  return v;
}

function render() {
  const q = document.getElementById("filter").value.toLowerCase();
  let rows = DATA.filter(r => !q || ["label","type","baseModel"].some(c => (r[c]||"").toString().toLowerCase().includes(q)));
  rows.sort((a, b) => {
    let av = a[sortCol], bv = b[sortCol];
    if (av === null || av === undefined) av = -Infinity;
    if (bv === null || bv === undefined) bv = -Infinity;
    if (typeof av === "string") return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  });
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = rows.map(r => "<tr>" + COLUMNS.map(c => `<td>${fmtCell(c, r[c])}</td>`).join("") + "</tr>").join("");
  document.querySelectorAll("th").forEach(th => {
    th.classList.remove("sorted-asc", "sorted-desc");
    if (th.dataset.col === sortCol) th.classList.add(sortDir === 1 ? "sorted-asc" : "sorted-desc");
  });
}

document.querySelectorAll("th").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (sortCol === col) sortDir *= -1; else { sortCol = col; sortDir = -1; }
    render();
  });
});
document.getElementById("filter").addEventListener("input", render);
render();
</script>
</body></html>"""

    headers = "".join(f'<th data-col="{c}">{c}</th>' for c in columns)
    meta = (f"{len(records)} models — sorted by {args.period} downloads at fetch time, "
            f"deep_probe_limit={args.deep_probe_limit}. Click any column header to sort, use the box to filter.")

    html = (html.replace("__HEADERS__", headers)
                .replace("__META__", meta)
                .replace("__DATA__", json.dumps(records))
                .replace("__COLUMNS__", json.dumps(columns)))

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {path} ({len(records)} models — open in a browser, sortable/filterable, "
          f"no internet needed)", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Probe candidate models for current activity.")
    parser.add_argument("--candidate-count", type=int, default=PROBE_CANDIDATE_COUNT,
                         help=f"Size of the download-ranked candidate pool to probe (default: {PROBE_CANDIDATE_COUNT})")
    parser.add_argument("--period", default=PROBE_PERIOD, choices=["Day", "Week", "Month", "Year", "AllTime"],
                         help=f"Timeframe window for download popularity ranking (default: {PROBE_PERIOD})")
    parser.add_argument("--since-days", type=int, default=PROBE_SINCE_DAYS,
                         help=f"Activity window to check images against (default: {PROBE_SINCE_DAYS})")
    parser.add_argument("--types", nargs="+", default=PROBE_TYPES,
                         help="Restrict candidate pool to these Civitai model types, e.g. --types Checkpoint Lora")
    parser.add_argument("--max-lora-versions", type=int, default=None,
                         help="Skip LORA-type candidates with more than N versions")
    parser.add_argument("--page-limit", type=int, default=PROBE_PAGE_LIMIT,
                         help=f"How many images deep to check per model on the Tier 1 pass (default: {PROBE_PAGE_LIMIT})")
    parser.add_argument("--deep-probe-limit", type=int, default=PROBE_DEEP_PROBE_LIMIT,
                         help=f"Max items to check for green models across an adaptive Tier 2 page (default: {PROBE_DEEP_PROBE_LIMIT})")
    parser.add_argument("--only-ids", nargs="+", type=int, default=None,
                         help="Probe exactly these model IDs instead of a fresh download-ranked pool.")
    parser.add_argument("--nsfw", default=PROBE_NSFW)
    parser.add_argument("--velocity-top-n", type=int, default=PROBE_VELOCITY_TOP_N,
                         help=f"Tier 3b: measure sustained daily post velocity (day-bucketed, not a raw count) "
                              f"for this many top models by total_probe_count (default: {PROBE_VELOCITY_TOP_N}, 0 = off)")
    parser.add_argument("--velocity-window-days", type=int, default=PROBE_VELOCITY_WINDOW_DAYS,
                         help=f"Window size in days for velocity measurement (default: {PROBE_VELOCITY_WINDOW_DAYS})")
    parser.add_argument("--velocity-max-pages", type=int, default=PROBE_VELOCITY_MAX_PAGES,
                         help=f"Hard safety cap on pages fetched per model during velocity measurement (default: {PROBE_VELOCITY_MAX_PAGES})")
    args = parser.parse_args()

    run_t0 = time.monotonic()
    _log(f"Run start — candidate_count={args.candidate_count}, page_limit={args.page_limit}, "
         f"deep_probe_limit={args.deep_probe_limit}, velocity_top_n={args.velocity_top_n}")

    results, models, since = probe_candidates(candidate_count=args.candidate_count, since_days=args.since_days, period=args.period,
                                nsfw=args.nsfw, types=args.types, max_lora_versions=args.max_lora_versions,
                                page_limit=args.page_limit, deep_probe_limit=args.deep_probe_limit,
                                only_ids=args.only_ids)

    capped = sum(1 for r in results if r.get("has_more_pages"))
    if capped:
        print(f"\n⚠️  {capped} model(s) hit the Tier 2 --deep-probe-limit ({args.deep_probe_limit}) and are "
              f"tied at that value in total_probe_count — their real ranking is unresolved.\n"
              f"   Use --velocity-top-n (on by default) to resolve them by day-bucketed velocity instead of "
              f"raising --deep-probe-limit — a bigger count cap does NOT reliably resolve this (see README).\n", flush=True)

    if args.velocity_top_n:
        results = add_velocity(results, models, args.velocity_top_n, args.page_limit, args.nsfw,
                                window_days=args.velocity_window_days, max_pages=args.velocity_max_pages)
        maxed = sum(1 for r in results if r.get("velocity_probe_status") == "max_pages_hit")
        if maxed:
            print(f"\n⚠️  {maxed} model(s) hit --velocity-max-pages ({args.velocity_max_pages}) while measuring "
                  f"velocity — their velocity_per_day is a lower bound, not exact. Raise --velocity-max-pages "
                  f"if you need the precise number for these.\n", flush=True)

    print(flush=True)  # newline after the run of '.'/'X'/'o' progress marks
    _log(f"Run total: {time.monotonic() - run_t0:.2f}s wall-clock (discovery + Tier 1/2 + Tier 3b combined — "
         f"see the phase logs above for the breakdown)")
    df = pd.DataFrame(results)
    
    if df.empty:
        print("❌ No models matched the given filters. Output files will not be generated.", flush=True)
        return

    df["label"] = df["modelName"] + " (" + df["modelId"].astype(str) + ")"
    df = df.sort_values("download_rank").reset_index(drop=True)

    # Tag output filenames with period + run date so different test runs
    # (e.g. --period Day vs --period Month) don't overwrite each other.
    # e.g. probe_results_day_12jul26.csv
    suffix = f"_{args.period.lower()}_{datetime.now().strftime('%d%b%y').lower()}"
    results_path = f"probe_results{suffix}.csv"
    report_path = f"probe_report{suffix}.html"
    dist_path = f"probe_distribution{suffix}.png"

    df.to_csv(results_path, index=False)
    print(f"Wrote {results_path} ({len(df)} models, sorted by download_rank over the last {args.period})", flush=True)

    corr = df["download_rank"].corr(df["total_probe_count"].rank())
    print(f"Spearman correlation (download_rank vs total_probe_count): {corr:.2f}", flush=True)

    print(f"\nTop 20 by CURRENT activity (Initial pool sorted by {args.period} downloads):", flush=True)
    activity_view = df.sort_values("total_probe_count", ascending=False)
    print(activity_view[["label", "type", "download_rank", "page1_count", "total_probe_count", "has_more_pages"]]
          .head(20).to_string(index=False), flush=True)

    if "velocity_per_day" in df.columns:
        vel_view = df[df["velocity_per_day"].notna()].sort_values("velocity_per_day", ascending=False)
        print(f"\nTop 20 by sustained {args.velocity_window_days}-day velocity "
              f"(burst_ratio: peak single day vs the {args.velocity_window_days}-day average — "
              f"high burst_ratio = spiky/viral, not sustained):", flush=True)
        print(vel_view[["label", "velocity_per_day", "max_single_day", "burst_ratio", "window_total"]]
              .head(20).to_string(index=False), flush=True)

    write_html_report(df, args, path=report_path)

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
        print(f"  ({n} models — thinned x-axis labels to every {step}th for plot scannability)", flush=True)
        
    plt.tight_layout()
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Wrote {dist_path} (shows the overall green/red distribution shape only — "
          f"open {report_path} to browse/sort/filter individual models)", flush=True)

if __name__ == "__main__":
    main()