# civitai-fetcher

Pull recent images + generation metadata for Civitai's most popular models, concurrently,
validated, ready to feed into a downstream vector DB (Qdrant) pipeline.

## Status

⚠️ **`civitai-fetcher` (the main CLI) is currently broken.** `cli.py` imports `fetch_all` from
`fetch.py`, but `fetch.py` has no `fetch_all` function — only `get_popular_models`,
`probe_model_activity`, `probe_candidates`. Running the CLI raises `ImportError` on startup.
Needs a `fetch_all` implementation (or `cli.py` repointed) before this is usable.

Working: model discovery, concurrent activity probing, metadata filtering, resource resolution,
trend analysis. **Not yet done:** Qdrant loader / embedding step, and the `fetch_all` fix above.

## Tools in this repo

There are three separate entry points, not one:

| Script | Purpose |
|---|---|
| `civitai_fetcher.cli` | Main fetcher — pull images + meta for popular models. **Currently broken, see above.** |
| `civitai_fetcher.probe` | Ranks a download-popular candidate pool by *current* image activity, writes CSV + chart |
| `analyse_trends.py` | Standalone script — trend charts from an existing fetcher output JSON |

## Usage

### Main fetcher (broken — see Status)

```bash
uv sync
uv run python -m civitai_fetcher.cli --model-count 10 --since-days 1 --resolve-resources
```

Flags (from `cli.py`): `--model-count`, `--since-days`, `--max-pages`, `--max-versions`,
`--types` (e.g. `--types Checkpoint LORA`), `--max-lora-versions`, `--nsfw` (default `X`, avoids
silent NSFW filtering), `--resolve-resources` (resolves `civitaiResources` modelVersionIds to
names + creator usernames via `resolve.py`, cached to `civitai_resolver_cache.json`).

To use an API token (optional, raises rate limits):

```bash
export CIVITAI_API_TOKEN=your_token_here   # bash/zsh
$env:CIVITAI_API_TOKEN = "your_token_here" # PowerShell
```

Writes:
- `civitai_output.json` — one flat record per image
- `civitai_output_issues.json` — only written if validation flags anything
- `civitai_resolver_cache.json` — only if `--resolve-resources` used

### Activity probe (working)

```bash
uv run python -m civitai_fetcher.probe
```

Zero-arg default now does the full pipeline in one reasonable-length run: discover a
download-ranked candidate pool, probe each for recent post activity (Tier 2), then measure
sustained daily velocity for the top 150 (Tier 3b) to properly rank the models that tie at the
Tier 2 cap. Writes `probe_results.csv` and `probe_distribution.png`.

Defaults (all tuned from real runs, see **Discoveries** below):
`--candidate-count 1000`, `--period Month`, `--since-days 30`, `--page-limit 100`,
`--deep-probe-limit 300`, `--velocity-top-n 150`, `--velocity-window-days 3`,
`--velocity-max-pages 400`. `--only-ids` probes specific model IDs instead of a fresh ranked pool.

Progress markers while it runs: `.` = clean request, `o` = succeeded after a retry (early
rate-limit warning), `X` = gave up after 3 retries.

The CLI prints a warning (with the specific fix) whenever a limit is actually constraining your
results — candidate pool capped by Civitai's own ranking cursor, Tier 2 ties unresolved, or Tier
3b velocity hitting `--velocity-max-pages`. If you don't see a warning, that limit isn't your
bottleneck right now.

### Trend analysis (working, standalone)

```bash
uv run python analyse_trends.py civitai_output.json [max_span_days]
```

Reads a fetcher output JSON and writes `trend_summary.csv`, `images_per_day.png`,
`posts_per_day.png`, `creators_per_day.png` (if a creator field is present), and
`resource_usage_per_day.png` (if `civitaiResources`/`meta.civitaiResources` is present).
Optional `max_span_days` excludes models whose data spans an unusually wide date range
(usually a low-activity model padded out with old images).

## Output shape (per image)

```json
{
  "modelId": 123,
  "modelName": "...",
  "modelUrl": "https://civitai.red/models/123",
  "imageId": 456,
  "imageUrl": "https://image.civitai.com/...",
  "postId": 789,
  "postUrl": "https://civitai.red/posts/789",
  "width": 832,
  "height": 1216,
  "createdAt": "...",
  "nsfwLevel": 1,
  "stats": { "..." : "..." },
  "meta": { "prompt": "...", "steps": 20, "sampler": "...", "cfgScale": 7, "seed": 123, "Size": "832x1216" }
}
```

Image-centric, not model-centric: one entry per image, model fields repeated per row.
This suits filtering/aggregation and eventual embedding better than nesting images under models.

## Known data quirks

- `meta.Size` (generation-time size) can legitimately differ from `width`/`height` (actual file)
  when an image is upscaled or resized after generation. Flagged, not treated as an error.
- `civitai.com` is SFW-only since the April 2026 split; `civitai.red` serves both SFW and NSFW.
  Human-facing links (`modelUrl`, `postUrl`) use `.red`. The API host (`civitai.com/api/v1`) is
  unaffected by the split and still serves both.
- Civitai auto-tags images internally (character/landscape/gender-style content tags), but this
  isn't exposed on the public `/api/v1/images` schema — only `nsfw`/`nsfwLevel`/`browsingLevel`.
  If you need those tags, options are prompt-text heuristics or running your own CLIP classifier
  as a separate enrichment pass over `imageUrl`.

## Discoveries (probe tuning — read before re-tuning any of this)

Findings from real runs against the live API, kept here so we don't re-derive them from scratch.

**`total_probe_count` measures posts, not generations.** Tier 2 counts images posted in the
window; it's a proxy for generation activity, not a direct measure — we don't know the
post:generation ratio, and can't from this API.

**Download rank and current activity correlate weakly.** Spearman correlation between
`download_rank` and `total_probe_count` has consistently come out around -0.40 to -0.45 across
runs — a real signal, but download-popularity is a mediocre proxy for what's active *right now*.
This is the whole reason the probe exists rather than just reading the downloads-ranked list.

**Raising the Tier 2 count cap does not resolve ties — the top of the distribution is a
long/power-law tail.** Tested directly: at `--deep-probe-limit 150`, 137/345 models were tied at
the cap. Raising the cap to 1000 only shrank that to 37 — and the pattern (`≥150→137, ≥200→113,
≥300→85, ≥500→61, ≥750→45, ≥1000→37`) doesn't converge, it just keeps peeling off a similar
fraction. **No cap will fully resolve this.** A "Tier 3" that re-probed the top N with a bigger
count cap was built, tested, confirmed to have this diminishing-returns problem, and was then
**removed from the code** (`refine_top_candidates` / `--refine-top-n` / `--refine-deep-limit` no
longer exist). Don't rebuild it — see the next point for what actually works.

**Day-bucketed velocity is the right metric, not a bigger count cap.** Instead of counting posts
up to an arbitrary cap, `probe_recent_velocity` (Tier 3b, `--velocity-top-n`) walks back a fixed
*time* window (`--velocity-window-days`, default 3) and buckets posts by calendar day. This is
bounded by time, not count, so it's cheap regardless of how active a model is, and it directly
answers "how many posts/day, sustained" instead of "did you hit my arbitrary ceiling." In one
real run this found a model tied with 18 others at the Tier 2 cap that was actually running
**946 images/day — 2.5x the next-highest model** once measured properly. That ranking was
completely invisible to Tier 2 alone.

**A single time-windowed average can hide a viral one-day spike — `burst_ratio` catches it.**
Two models with the same `velocity_per_day` can have totally different shapes: one steady, one a
single viral day followed by silence. `probe_recent_velocity` also reports `max_single_day` and
`burst_ratio` (peak day ÷ window average). `burst_ratio` near 1.0 = sustained; well above 1 =
spiky/viral, treat the average with caution. Verified with synthetic sustained-vs-spike test data
before trusting it against the real API.

**`--candidate-count` has an API-side ceiling, not a code-side one.** Requesting
`--candidate-count 1000` for `--period Month --types Checkpoint` returned only 345 models —
`get_popular_models`'s pagination loop exited because Civitai's own ranking cursor ran out
(empty page or no `nextCursor`), confirmed by `download_rank` topping out exactly at 345 with
zero duplicate `modelId`s. Raising `--candidate-count` further does nothing for that specific
`period`+`types` combo. To get more candidates: widen `--period` (e.g. `AllTime` has a deeper
cursor than `Month`) or drop `--types`. The CLI now warns automatically when this happens, with
the specific fix, instead of silently returning fewer models than requested.

**A `0` in `total_probe_count`/`page1_count` can mean two different things, and used to be
ambiguous.** Genuinely zero recent posts vs. the probe silently erroring (deleted model, bad ID,
transient network fault) both produced the same all-zero result dict. Fixed by adding
`probe_status` (`ok` / `error` / `no_model_id`) — check this before treating a `0` as a real
signal.

**Defaults were tuned up from a run that showed clear headroom.** The `--candidate-count 1000
--velocity-top-n 150` run above completed with almost no `o` markers (retry-then-succeed) and
zero `X` markers (exhausted retries) — meaning the API had capacity to spare at those settings.
Current defaults (`--candidate-count 1000`, `--page-limit 100`, `--deep-probe-limit 300`,
`--velocity-max-pages 400`, `--velocity-top-n 150` on by default) are roughly double what was
proven safe, specifically so a **zero-argument run** (`python -m civitai_fetcher.probe`) does
something useful in one shot without hand-tuning. If you ever see the cap-hit warnings mentioned
above during a default run, that's the signal these need raising again — don't guess, read the
warning, it names the exact flag.

## Roadmap

- [ ] Fix `civitai-fetcher` CLI: implement/restore `fetch_all` in `fetch.py` (or repoint `cli.py`)
- [ ] Normalise categorical fields (`sampler`, `civitaiResources`) for consistent grouping
- [ ] Qdrant loader: embed `prompt` (+ optionally `negativePrompt`), rest as filterable payload
- [ ] Optional CLIP pass for character/landscape + gender classification