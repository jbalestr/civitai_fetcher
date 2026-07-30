"""
One-off cleanup: delete images already in the DB that match the
checkpoint-spam pattern (>1 checkpoint-type resource), now that ingestion
filters these out going forward. Run once against your real DB.

Usage:
    uv run python delete_checkpoint_spam.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection

conn = get_connection(DB_PATH)

spam_ids = [
    r[0] for r in conn.execute(
        """
        SELECT imageId FROM image_resources
        WHERE resource_type = 'checkpoint'
        GROUP BY imageId
        HAVING COUNT(*) > 1
        """
    ).fetchall()
]

print(f"Found {len(spam_ids)} spammed image(s) to delete.")

if spam_ids:
    placeholders = ",".join("?" * len(spam_ids))
    conn.execute(f"DELETE FROM image_resources WHERE imageId IN ({placeholders})", spam_ids)
    conn.execute(f"DELETE FROM image_stats WHERE imageId IN ({placeholders})", spam_ids)
    conn.execute(f"DELETE FROM raw_meta WHERE imageId IN ({placeholders})", spam_ids)
    conn.execute(f"DELETE FROM images WHERE imageId IN ({placeholders})", spam_ids)
    conn.commit()
    print(f"Deleted {len(spam_ids)} image(s) and their related rows.")
else:
    print("Nothing to delete.")

print("Remaining images:", conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])