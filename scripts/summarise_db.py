"""
Summarise what's in the DB: image counts per model, per creator, and a
breakdown by media type (image/video/audio).

Usage:
    uv run python scripts/summarise_db.py
    uv run python scripts/summarise_db.py --top 10       # cap listed rows
    uv run python scripts/summarise_db.py --csv-dir out  # write full CSVs
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection


def _print_table(rows, columns, top=None):
    to_show = rows[:top] if top else rows
    for r in to_show:
        print("  " + "  ".join(f"{r[c]}" for c in columns))
    if top and len(rows) > top:
        print(f"  ... and {len(rows) - top} more")


def _write_csv(rows, columns, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r[c] for c in columns])
    print(f"  wrote {len(rows)} row(s) to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20, help="Rows to print per section (default 20). 0 = all.")
    parser.add_argument("--csv-dir", help="Optional dir to also write full per-section CSVs.")
    args = parser.parse_args()
    top = args.top or None

    conn = get_connection(DB_PATH)

    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    print(f"Total images: {total_images}\n")

    print("By media type:")
    media_rows = conn.execute(
        """
        SELECT COALESCE(media_type, 'unknown') AS media_type, COUNT(*) AS n
        FROM images
        GROUP BY media_type
        ORDER BY n DESC
        """
    ).fetchall()
    _print_table(media_rows, ["media_type", "n"])
    print()

    print(f"By model (top {top or 'all'}):")
    model_rows = conn.execute(
        """
        SELECT COALESCE(i.modelId, r.modelId) AS modelId,
               COALESCE(m.modelName, m2.modelName) AS modelName,
               COUNT(i.imageId) AS image_count,
               SUM(CASE WHEN i.media_type = 'image' THEN 1 ELSE 0 END) AS images,
               SUM(CASE WHEN i.media_type = 'video' THEN 1 ELSE 0 END) AS videos
        FROM images i
        LEFT JOIN models m ON m.modelId = i.modelId
        LEFT JOIN resources r ON r.modelVersionId = i.modelVersionId
        LEFT JOIN models m2 ON m2.modelId = r.modelId
        GROUP BY COALESCE(i.modelId, r.modelId)
        ORDER BY image_count DESC
        """
    ).fetchall()
    _print_table(model_rows, ["modelId", "modelName", "image_count", "images", "videos"], top)
    print()

    print(f"By creator/poster (top {top or 'all'}):")
    creator_rows = conn.execute(
        """
        SELECT posterUsername, COUNT(*) AS image_count,
               SUM(CASE WHEN media_type = 'image' THEN 1 ELSE 0 END) AS images,
               SUM(CASE WHEN media_type = 'video' THEN 1 ELSE 0 END) AS videos
        FROM images
        GROUP BY posterUsername
        ORDER BY image_count DESC
        """
    ).fetchall()
    _print_table(creator_rows, ["posterUsername", "image_count", "images", "videos"], top)

    if args.csv_dir:
        print()
        _write_csv(media_rows, ["media_type", "n"], os.path.join(args.csv_dir, "by_media_type.csv"))
        _write_csv(model_rows, ["modelId", "modelName", "image_count", "images", "videos"],
                   os.path.join(args.csv_dir, "by_model.csv"))
        _write_csv(creator_rows, ["posterUsername", "image_count", "images", "videos"],
                   os.path.join(args.csv_dir, "by_creator.csv"))


if __name__ == "__main__":
    main()