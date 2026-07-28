import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else input("Path to output JSON: ")
with open(path, encoding="utf-8") as f:
    entries = json.load(f)

by_id = {e["imageId"]: e for e in entries}
cluster = [136601380, 136601158, 136600848, 136600851, 136600579, 136600194,
           136600004, 136599799, 136599422, 136599191, 136598719]
for iid in cluster:
    e = by_id.get(iid, {})
    print(f"{iid}: postId={e.get('postId')}  createdAt={e.get('createdAt')}")