"""
Flag images with null/blank generation meta -- e.g. uploads from tools that
don't embed prompt/sampler data, or images where meta genuinely never came
through.

"Blank meta" here means ALL of: prompt, negativePrompt, sampler, steps,
cfgScale are NULL on the images row, AND there are no rows in
image_resources for that image (no checkpoint/lora usage recorded either).
Any one of those being present is enough to NOT flag it -- this is meant to
catch the "nothing at all" case, not just "no prompt but resources known".

Usage:
    uv run python scripts/find_null_meta.py                # summary + list
    uv run python scripts/find_null_meta.py --limit 20      # cap listed rows
    uv run python scripts/find_null_meta.py --csv out.csv   # also write CSV
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Cap how many rows are printed.")
    parser.add_argument("--csv", help="Optional path to also write full results as CSV.")
    args = parser.parse_args()

    conn = get_connection(DB_PATH)

    sql = """
        SELECT i.imageId, i.postId, i.posterUsername, i.createdAt, i.imageUrl
        FROM images i
        LEFT JOIN image_resources ir ON ir.imageId = i.imageId
        WHERE i.prompt IS NULL
          AND i.negativePrompt IS NULL
          AND i.sampler IS NULL
          AND i.steps IS NULL
          AND i.cfgScale IS NULL
          AND ir.imageId IS NULL
        ORDER BY i.createdAt DESC
    """
    rows = conn.execute(sql).fetchall()
    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    pct = (len(rows) / total_images * 100) if total_images else 0
    print(f"{len(rows)} / {total_images} images ({pct:.1f}%) have null/blank meta.\n")

    to_print = rows[: args.limit] if args.limit else rows
    for r in to_print:
        print(f"{r['imageId']}  post={r['postId']}  user={r['posterUsername']}  "
              f"created={r['createdAt']}  {r['imageUrl']}")
    if args.limit and len(rows) > args.limit:
        print(f"... and {len(rows) - args.limit} more (use --limit to show more, or --csv for all)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["imageId", "postId", "posterUsername", "createdAt", "imageUrl"])
            for r in rows:
                writer.writerow([r["imageId"], r["postId"], r["posterUsername"], r["createdAt"], r["imageUrl"]])
        print(f"\nWrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()