import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from civitai_fetcher.core.config import DB_PATH
from civitai_fetcher.db import get_connection

conn = get_connection(DB_PATH)

# Of the images with no modelId, how many have a modelVersionId at all?
print("images with modelId IS NULL, broken down by modelVersionId presence:")
for r in conn.execute("""
    SELECT
      CASE WHEN modelVersionId IS NULL THEN 'no modelVersionId' ELSE 'has modelVersionId' END AS bucket,
      COUNT(*) AS n
    FROM images WHERE modelId IS NULL
    GROUP BY bucket
""").fetchall():
    print(" ", r["bucket"], r["n"])

# Of those WITH a modelVersionId, how many resolve via resources.modelId?
print("\nOf images with modelId IS NULL and modelVersionId set, resources.modelId resolvable?")
for r in conn.execute("""
    SELECT
      CASE WHEN r.modelId IS NULL THEN 'resources.modelId NULL' ELSE 'resolved' END AS bucket,
      COUNT(*) AS n
    FROM images i
    LEFT JOIN resources r ON r.modelVersionId = i.modelVersionId
    WHERE i.modelId IS NULL AND i.modelVersionId IS NOT NULL
    GROUP BY bucket
""").fetchall():
    print(" ", r["bucket"], r["n"])