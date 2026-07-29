"""
Quick, ready-made views into the DB -- no SQL required for the common cases.

Usage:
    uv run python scripts/browse.py summary
    uv run python scripts/browse.py top-images --limit 10
    uv run python scripts/browse.py top-images --model-id 443821 --limit 20
    uv run python scripts/browse.py top-resources --type lora --limit 15
    uv run python scripts/browse.py recent --limit 10
    uv run python scripts/browse.py search "cyberpunk city"
    uv run python scripts/browse.py history 12345678
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection, get_raw_meta, compression_stats


def _print_rows(rows, columns):
    """Simple fixed-width table print -- good enough for a terminal, no extra deps."""
    if not rows:
        print("(no rows)")
        return
    widths = {c: max(len(c), max((len(str(r[c])) if r[c] is not None else 4) for r in rows)) for c in columns}
    # clamp very long columns (e.g. prompt) so the table doesn't wrap unreadably
    widths = {c: min(w, 60) for c, w in widths.items()}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = []
        for c in columns:
            val = "" if r[c] is None else str(r[c])
            if len(val) > widths[c]:
                val = val[: widths[c] - 1] + "…"
            line.append(val.ljust(widths[c]))
        print("  ".join(line))


def cmd_summary(conn, args):
    counts = {
        "images": "SELECT COUNT(*) FROM images",
        "models": "SELECT COUNT(*) FROM models",
        "resources": "SELECT COUNT(*) FROM resources",
        "image_stats rows": "SELECT COUNT(*) FROM image_stats",
        "unenriched images": "SELECT COUNT(*) FROM images WHERE enriched_at IS NULL",
    }
    for label, sql in counts.items():
        print(f"{label:20s} {conn.execute(sql).fetchone()[0]}")
    print(f"{'compression':20s} {compression_stats(conn)}")


def cmd_top_images(conn, args):
    where = ""
    params = []
    if args.model_id:
        where = "WHERE i.modelId = ?"
        params.append(args.model_id)
    sql = f"""
        SELECT i.imageId, m.modelName, i.prompt, s.reactionScore, i.createdAt
        FROM images i
        LEFT JOIN models m ON m.modelId = i.modelId
        LEFT JOIN (
            SELECT imageId, MAX(fetched_at) AS latest FROM image_stats GROUP BY imageId
        ) latest_stats ON latest_stats.imageId = i.imageId
        LEFT JOIN image_stats s ON s.imageId = i.imageId AND s.fetched_at = latest_stats.latest
        {where}
        ORDER BY s.reactionScore DESC
        LIMIT ?
    """
    params.append(args.limit)
    rows = conn.execute(sql, params).fetchall()
    _print_rows(rows, ["imageId", "modelName", "prompt", "reactionScore", "createdAt"])


def cmd_top_resources(conn, args):
    where = ""
    params = []
    if args.type:
        where = "WHERE ir.resource_type = ?"
        params.append(args.type)
    sql = f"""
        SELECT r.modelVersionId, r.name, r.resource_type, r.creatorUsername, COUNT(*) AS uses
        FROM image_resources ir
        JOIN resources r ON r.modelVersionId = ir.modelVersionId
        {where}
        GROUP BY r.modelVersionId
        ORDER BY uses DESC
        LIMIT ?
    """
    params.append(args.limit)
    rows = conn.execute(sql, params).fetchall()
    _print_rows(rows, ["modelVersionId", "name", "resource_type", "creatorUsername", "uses"])


def cmd_recent(conn, args):
    sql = """
        SELECT imageId, modelId, prompt, first_seen_at, createdAt
        FROM images
        ORDER BY first_seen_at DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (args.limit,)).fetchall()
    _print_rows(rows, ["imageId", "modelId", "prompt", "first_seen_at", "createdAt"])


def cmd_search(conn, args):
    sql = """
        SELECT imageId, modelId, prompt, createdAt
        FROM images
        WHERE prompt LIKE ?
        ORDER BY createdAt DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (f"%{args.query}%", args.limit)).fetchall()
    _print_rows(rows, ["imageId", "modelId", "prompt", "createdAt"])


def cmd_history(conn, args):
    sql = """
        SELECT fetched_at, likeCount, heartCount, laughCount, cryCount, commentCount, reactionScore
        FROM image_stats
        WHERE imageId = ?
        ORDER BY fetched_at
    """
    rows = conn.execute(sql, (args.image_id,)).fetchall()
    _print_rows(rows, ["fetched_at", "likeCount", "heartCount", "laughCount", "cryCount", "commentCount", "reactionScore"])


def cmd_meta(conn, args):
    meta = get_raw_meta(conn, args.image_id)
    if meta is None:
        print(f"No raw_meta stored for imageId {args.image_id}")
        return
    import json
    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=DB_PATH, help=f"DB path (default: {DB_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary", help="Row counts + compression stats")

    p = sub.add_parser("top-images", help="Images ranked by latest reactionScore")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--model-id", type=int, default=None)

    p = sub.add_parser("top-resources", help="Most-used LoRAs/checkpoints across all images")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--type", choices=["lora", "checkpoint"], default=None)

    p = sub.add_parser("recent", help="Most recently first-seen images (by DB insert time, not createdAt)")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("search", help="Search prompts for a substring")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("history", help="Reaction/stats history for one image over time")
    p.add_argument("image_id", type=int)

    p = sub.add_parser("meta", help="Print the full raw (decompressed) meta blob for one image")
    p.add_argument("image_id", type=int)

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"No DB found at {args.db}")
        sys.exit(1)

    conn = get_connection(args.db)
    {
        "summary": cmd_summary,
        "top-images": cmd_top_images,
        "top-resources": cmd_top_resources,
        "recent": cmd_recent,
        "search": cmd_search,
        "history": cmd_history,
        "meta": cmd_meta,
    }[args.command](conn, args)


if __name__ == "__main__":
    main()
