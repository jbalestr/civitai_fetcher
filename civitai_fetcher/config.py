import os

BASE = "https://civitai.com/api/v1"  # API host is unaffected by the .com/.red split
SITE = "https://civitai.red"          # human-facing links: .com is SFW-only, .red serves all

# Optional bearer token for higher rate limits. Set CIVITAI_API_TOKEN in your environment,
# never commit a token to the repo.
API_TOKEN = os.environ.get("CIVITAI_API_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

MAX_WORKERS = 8
OUT_PATH = "civitai_output.json"
ISSUES_PATH = "civitai_output_issues.json"
RESOLVER_CACHE_PATH = "civitai_resolver_cache.json"
