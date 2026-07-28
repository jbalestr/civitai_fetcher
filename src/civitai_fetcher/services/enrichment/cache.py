"""
Shared keyed disk-cache for enrichment lookups.

Both resolve.py (resource/creator name resolution) and generation.py
(internal-API meta backfill) enrich data that's immutable once fetched —
a modelVersionId's name doesn't change, and a given image's generation
data doesn't change once the image exists. Pulled out as one class instead
of resolve.py's original hand-rolled dict + load/save pair, so generation.py
gets caching too (it previously had none, despite hitting a riskier
authenticated internal endpoint on every single run).

Not for anything time-sensitive or mutating (reaction scores, activity
counts) — those must NOT be cached this way, since a cache hit here is
permanent for the process lifetime and persisted to disk indefinitely.

Namespaced so one on-disk file can hold multiple independent caches
(resolve.py needs two: versions and models) without key collisions.
"""
import json
import os


class KeyedDiskCache:
    """
    A small set of namespaced dict caches, persisted to a single JSON file.

    Usage:
        cache = KeyedDiskCache("civitai_resolver_cache.json")
        cache.load()
        cache.get("versions", "12345")          # None if not cached
        cache.set("versions", "12345", {...})
        cache.save()
    """

    def __init__(self, path):
        self.path = path
        self._data = {}

    def load(self):
        """Load all namespaces from disk, if the file exists. No-op otherwise."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                self._data = json.load(f)
            sizes = ", ".join(f"{len(v)} {k}" for k, v in self._data.items())
            print(f"Loaded cache {self.path}: {sizes or 'empty'}")
        except Exception as e:
            print(f"  [cache] failed to load {self.path}: {e}")

    def save(self):
        """Persist all namespaces to disk."""
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)
            sizes = ", ".join(f"{len(v)} {k}" for k, v in self._data.items())
            print(f"Saved cache {self.path}: {sizes or 'empty'}")
        except Exception as e:
            print(f"  [cache] failed to save {self.path}: {e}")

    def get(self, namespace, key, default=None):
        return self._data.get(namespace, {}).get(str(key), default)

    def contains(self, namespace, key):
        return str(key) in self._data.get(namespace, {})

    def set(self, namespace, key, value):
        self._data.setdefault(namespace, {})[str(key)] = value

    def namespace_size(self, namespace):
        return len(self._data.get(namespace, {}))
