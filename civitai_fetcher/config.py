import os

BASE = "https://civitai.com/api/v1"  # API host is unaffected by the .com/.red split
SITE = "https://civitai.red"          # human-facing links: .com is SFW-only, .red serves all

# Optional bearer token for higher rate limits. Set CIVITAI_API_TOKEN in your environment,
# never commit a token to the repo.
API_TOKEN = os.environ.get("CIVITAI_API_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

MAX_WORKERS = 24            # bumped from 8 — logs showed 0 rate-limit hits, so there was headroom
POOL_MAXSIZE = 32            # HTTP connection pool size — should be >= MAX_WORKERS so no thread blocks waiting for a free connection
OUT_PATH = "civitai_output.json"
ISSUES_PATH = "civitai_output_issues.json"
RESOLVER_CACHE_PATH = "civitai_resolver_cache.json"

# --- HTTP / retry tuning (_get_with_retry in fetch.py) ---
REQUEST_TIMEOUT = 15        # seconds, per request
MAX_RETRIES = 3             # attempts before giving up
BACKOFF_CAP = 30            # seconds, ceiling on exponential backoff sleep
BACKOFF_JITTER_MIN = 0.5
BACKOFF_JITTER_MAX = 1.5

# --- get_popular_models pagination ---
MODELS_PAGE_SIZE = 100      # items requested per /models page while paging to candidate_count

# --- probe.py CLI defaults (Tier 1 / Tier 2 / Tier 3b activity probe) ---
PROBE_CANDIDATE_COUNT = 1000
PROBE_PERIOD = "Month"
PROBE_SINCE_DAYS = 30
PROBE_TYPES = "Checkpoint"
PROBE_PAGE_LIMIT = 100          # Tier 1 page depth
PROBE_DEEP_PROBE_LIMIT = 150    # Tier 2 adaptive cap
PROBE_NSFW = "X"
PROBE_VELOCITY_TOP_N = 150      # Tier 3b: 0 disables
PROBE_VELOCITY_WINDOW_DAYS = 3
PROBE_VELOCITY_MAX_PAGES = 400

# --- cli.py defaults (fetch_all pipeline) ---
FETCH_MODEL_COUNT = 10
FETCH_SINCE_DAYS = 1
FETCH_MAX_PAGES = 20
FETCH_NSFW = "X"

# --- images_cli.py defaults (activity-ranked discovery -> image fetch -> reaction rank) ---
# Week strikes the best signal-to-noise balance in practice: Day's candidate pool is
# too thin (~15 models) to be representative, Month's (~350) buries the images you
# actually want to see under a lot of low-signal noise before you get to them.
IMAGES_PERIOD = "Week"
IMAGES_SINCE_DAYS = 7
IMAGES_TOP_MODELS = 20      # how many activity-ranked models (by velocity_per_day) to pull images for
IMAGES_MAX_PAGES = FETCH_MAX_PAGES
IMAGES_NSFW = FETCH_NSFW
IMAGES_TOP_REACTIONS = 30   # how many top-reaction images to keep/print, 0 = keep all

# --- phase-level safety valve ---
# Two layers: MODEL_TIMEOUT_SECONDS bounds a single model (checked between pages,
# so a model gives up on itself before eating the shared budget); PHASE_TIMEOUT_SECONDS
# is a much shorter backstop for the rare case many models stall at once.
MODEL_TIMEOUT_SECONDS = 30
PHASE_TIMEOUT_SECONDS = 60