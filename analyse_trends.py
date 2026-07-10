"""
Trend analysis for civitai_fetcher output.

Usage:
    python analyze_trends.py civitai_output.json

Produces:
    - images_per_day.png       (histogram: images per day, by model)
    - creators_per_day.png     (histogram: unique creators per day, by model)
    - posts_per_day.png        (histogram: unique posts per day, by model)
    - trend_summary.csv        (raw daily counts, one row per model per day)
"""

import json
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Fallback fonts that include CJK glyphs (model names can contain Chinese/Japanese/Korean text).
# Falls back silently to DejaVu Sans if none of these are installed.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Field name candidates — adjust here if your schema differs
DATE_FIELDS = ["createdAt", "created_at", "publishedAt"]
MODEL_ID_FIELDS = ["modelId", "model_id"]
MODEL_NAME_FIELDS = ["modelName", "model_name"]
IMAGE_FIELDS = ["imageId", "image_id", "id"]
POST_FIELDS = ["postId", "post_id"]
CREATOR_FIELDS = ["posterUsername", "username", "creator", "creatorUsername", "user"]

# Where resource usage (loras/embeddings) lives — checked in this order
RESOURCE_LIST_FIELDS = ["resources", "meta.resources", "civitaiResources", "meta.civitaiResources"]
RESOURCE_TYPE_KEYS = ["type", "modelType"]
RESOURCE_NAME_KEYS = ["modelName", "name", "modelVersionName"]


def get_nested(record, dotted_key):
    parts = dotted_key.split(".")
    value = record
    for p in parts:
        if isinstance(value, dict) and p in value:
            value = value[p]
        else:
            return None
    return value


def extract_resources(record):
    """Return list of (type, name) tuples for resources used in this image."""
    for field in RESOURCE_LIST_FIELDS:
        value = get_nested(record, field)
        if isinstance(value, list) and value:
            out = []
            for res in value:
                if not isinstance(res, dict):
                    continue
                rtype = next((res[k] for k in RESOURCE_TYPE_KEYS if k in res and res[k]), None)
                rname = next((res[k] for k in RESOURCE_NAME_KEYS if k in res and res[k]), None)
                if rtype:
                    out.append((str(rtype).lower(), rname))
            if out:
                return out
    return []



def first_present(record, candidates):
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
        # also check nested "stats" dict, since schema keeps extra fields there
        stats = record.get("stats")
        if isinstance(stats, dict) and key in stats:
            return stats[key]
    return None


def parse_date(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # assume unix timestamp (seconds or ms)
        ts = value / 1000 if value > 1e12 else value
        return datetime.utcfromtimestamp(ts).date()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_trends.py civitai_output.json [max_span_days]")
        print("  max_span_days: optional — exclude models whose data spans more than N days")
        print("                 (a wide span usually means a low-activity model padded out with old images)")
        sys.exit(1)

    path = sys.argv[1]
    max_span_days = int(sys.argv[2]) if len(sys.argv) > 2 else None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else data.get("images", data.get("results", []))
    if not records:
        print("No records found — check the JSON structure and update the field candidates.")
        sys.exit(1)

    creator_field_found = False
    resource_field_found = False
    rows = []
    resource_rows = []
    for r in records:
        date = parse_date(first_present(r, DATE_FIELDS))
        model_id = first_present(r, MODEL_ID_FIELDS)
        model_name = first_present(r, MODEL_NAME_FIELDS)
        model = f"{model_name} ({model_id})" if model_name is not None else str(model_id)
        image_id = first_present(r, IMAGE_FIELDS)
        post_id = first_present(r, POST_FIELDS)
        creator = first_present(r, CREATOR_FIELDS)
        if creator is not None:
            creator_field_found = True
        rows.append(
            {"date": date, "model": model, "model_id": model_id, "model_name": model_name,
             "image_id": image_id, "post_id": post_id, "creator": creator}
        )

        resources = extract_resources(r)
        if resources:
            resource_field_found = True
            for rtype, rname in resources:
                resource_rows.append(
                    {"date": date, "image_id": image_id, "resource_type": rtype, "resource_name": rname}
                )

    df = pd.DataFrame(rows).dropna(subset=["date", "model"])
    res_df = pd.DataFrame(resource_rows).dropna(subset=["date"]) if resource_rows else pd.DataFrame()

    date_range = (
        df.groupby("model")["date"]
        .agg(oldest="min", newest="max", images="count")
        .reset_index()
    )
    date_range["span_days"] = (
        pd.to_datetime(date_range["newest"]) - pd.to_datetime(date_range["oldest"])
    ).dt.days
    date_range = date_range.sort_values("span_days", ascending=False)
    date_range.to_csv("model_date_ranges.csv", index=False)
    print(f"Wrote model_date_ranges.csv — widest span: "
          f"{date_range.iloc[0]['model']} ({date_range.iloc[0]['span_days']} days)")
    print(date_range.head(10).to_string(index=False))

    if max_span_days is not None:
        keep_models = set(date_range[date_range["span_days"] <= max_span_days]["model"])
        dropped = set(date_range["model"]) - keep_models
        if dropped:
            print(f"Excluding {len(dropped)} model(s) with span > {max_span_days} days: {sorted(dropped)}")
        df = df[df["model"].isin(keep_models)]
        res_df = res_df[res_df["date"].isin(df["date"].unique())] if not res_df.empty else res_df

    if not creator_field_found:
        print("Note: no creator/username field found in the data — creators_per_day will be empty.")
        print(f"Checked keys: {CREATOR_FIELDS}. Update CREATOR_FIELDS in the script if yours differs.")

    daily = (
        df.groupby(["date", "model"])
        .agg(
            images=("image_id", "nunique"),
            posts=("post_id", "nunique"),
            creators=("creator", "nunique"),
        )
        .reset_index()
    )
    daily.to_csv("trend_summary.csv", index=False)
    print(f"Wrote trend_summary.csv ({len(daily)} rows)")

    total_rows = len(df)
    unique_images = df.drop_duplicates(subset="image_id")
    print(f"Total rows: {total_rows} | Unique images: {len(unique_images)} "
          f"(duplicates come from images shared across multiple models/resources)")

    def plot_metric(metric, filename, title):
        pivot = daily.pivot_table(index="date", columns="model", values=metric, aggfunc="sum", fill_value=0)
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()

        fig, ax = plt.subplots(figsize=(14, 7))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", markersize=3, linewidth=1.2, label=col)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel(metric.capitalize())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=45)

        if pivot.shape[1] <= 15:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small", title="Model")
        else:
            print(f"  ({pivot.shape[1]} models — legend suppressed, see trend_summary.csv for names)")

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Wrote {filename}")

    plot_metric("images", "images_per_day.png", "Images per Day by Model")
    plot_metric("posts", "posts_per_day.png", "Posts per Day by Model")
    if creator_field_found:
        plot_metric("creators", "creators_per_day.png", "Unique Creators per Day by Model")

    if resource_field_found:
        res_df.to_csv("resource_usage.csv", index=False)
        print(f"Wrote resource_usage.csv ({len(res_df)} rows)")

        daily_res = (
            res_df.groupby(["date", "resource_type"])["image_id"].nunique().reset_index(name="uses")
        )
        pivot = daily_res.pivot_table(index="date", columns="resource_type", values="uses", fill_value=0)
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()

        fig, ax = plt.subplots(figsize=(14, 7))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", markersize=3, linewidth=1.2, label=col)

        ax.set_title("LoRA / Embedding Usage per Day")
        ax.set_xlabel("Date")
        ax.set_ylabel("Images using resource")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=45)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small", title="Resource type")

        plt.tight_layout()
        plt.savefig("resource_usage_per_day.png", dpi=150)
        plt.close()
        print("Wrote resource_usage_per_day.png")
    else:
        print("Note: no resources/loras/embeddings field found — skipped resource_usage_per_day.png")
        print(f"Checked keys: {RESOURCE_LIST_FIELDS}. Update RESOURCE_LIST_FIELDS if your schema differs.")



if __name__ == "__main__":
    main()