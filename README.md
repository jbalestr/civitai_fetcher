# civitai-fetcher

Find Civitai models by **current activity** (not just historical downloads), then pull their
images ranked by community reactions — concurrently, validated, ready to feed into a downstream
vector DB (Qdrant) pipeline.

## Status

Working end-to-end: model discovery, concurrent activity probing (download rank → recent-activity
rank → sustained-velocity rank), image fetching with generation metadata, reaction ranking,
resource-name resolution, resource-usage analysis, trend analysis. **Not yet done:** the Qdrant
loader / embedding step (see Roadmap).

The old `civitai_fetcher.cli` (`fetch_all`) was broken for a while — see `Architecture` below for
what replaced it. `civitai_fetcher.cli` is now a deprecated shim that just points you at
`civitai_fetcher.images_cli`, the real thing.

## Tools in this repo

| Script | Purpose |
|---|---|
| `civitai_fetcher.images_cli` | **Main entry point.** Rank models by current activity, fetch their images, rank by reactions, break down resource (LoRA/checkpoint) usage. |
| `civitai_fetcher.probe` | Rank a download-popular candidate pool by *current* image activity, writes CSV + HTML report + chart. `images_cli` builds on this. |
| `analyse_trends.py` | Standalone script — trend charts from an existing `images_cli`/old `cli` output JSON |
| `civitai_fetcher.cli` | Deprecated shim — points you at `images_cli` instead of doing anything itself |

## Architecture

The package is deliberately split so that image-fetching work can never break model-ranking work,
after that happening once (see Discoveries):

```
client.py     — pure Civitai API primitives: retry/backoff HTTP, request stats, model discovery.
                Nothing here knows about activity ranking or image fetching.
activity.py   — Tier 1/2/3b model activity ranking (the logic behind probe.py).
                Imports ONLY from client.py.
images.py     — image fetching + reaction scoring + resource-usage analysis.
                Imports ONLY from client.py, never from activity.py.
probe.py      — CLI for activity.py. Its import graph cannot reach images.py — structurally
                impossible for image-fetching changes to break this.
images_cli.py — CLI that composes activity.py (to build a Week-ranked model shortlist) with
                images.py (to fetch + rank their images).
resolve.py    — resolves civitaiResources modelVersionIds to names/creators, cached to disk.
validate.py   — flags suspicious/incomplete records in fetched output.
```

## Usage

### images_cli — find active models, fetch their images, rank by reactions

```bash
uv sync
uv run python -m civitai_fetcher.images_cli --top-models 10
```

Pipeline:
1. `activity.probe_candidates()` — rank models by download popularity, then by recent activity
   (Tier 1/2), then by sustained velocity (Tier 3b) — same logic `probe.py` uses.
2. Take the top `--top-models` by that ranking.
3. `images.fetch_images_for_models()` — pull every image with generation metadata in the window,
   for just those top models (not the whole candidate pool).
4. Rank fetched images by combined reaction count, client-side (see "Why reactions are ranked
   client-side" below).
5. Print LoRA/checkpoint resource-usage breakdowns.

Key flags:
- `--period` (default `Week` — see Discoveries for why), `--since-days` (defaults to match period)
- `--top-models` (default 20) — how many activity-ranked models to actually fetch images for
- `--top-reactions` (default 30) — keep only the top N images GLOBALLY, ranked by reactions
- `--top-reactions-per-model` — instead of one global top-N, keep the top N **per model**, so one
  high-reaction model (e.g. a Kreamania-style outlier) can't crowd every other model out of the
  output. Overrides `--top-reactions` when set. Use `--top-reactions 0` to keep everything fetched
  (no truncation at all — needed for feeding a full corpus into Qdrant later; fine until output
  gets into the 100MB+ range, prune then).
- `--resolve-resources` (**on by default**) — resolves `civitaiResources` modelVersionIds to
  names + creator usernames, cached to `civitai_resolver_cache.json`. Use `--no-resolve-resources`
  to skip (faster, but LoRA/checkpoint names in the usage breakdowns stay as raw IDs).
- `--top-resources` (default 15) — how many entries in the LoRA/combo/bare-checkpoint breakdowns
- `--types`, `--max-lora-versions`, `--max-pages`, `--max-versions`, `--nsfw`,
  `--candidate-count`, `--page-limit`, `--deep-probe-limit`, `--velocity-window-days`,
  `--velocity-max-pages` — same meaning as the equivalent `probe.py` flags

To use an API token (optional, raises rate limits):

```bash
export CIVITAI_API_TOKEN=your_token_here   # bash/zsh
$env:CIVITAI_API_TOKEN = "your_token_here" # PowerShell
```

Writes (all tagged with period + run date + run **time**, so successive same-day runs while
tuning flags never silently overwrite each other):
- `civitai_output_<period>_<ddMMMyy>_<HHMM>.json` — ranked image records
- `civitai_output_issues_<period>_<ddMMMyy>_<HHMM>.json` — only written if validation flags anything
- `civitai_resolver_cache.json` — persistent across runs, only touched if resolving resources

#### Resource-usage breakdowns

Three genuinely different questions, printed as three separate sections — don't conflate them:

- **LoRA usage overall** — which LoRA appears most across ALL fetched images, regardless of which
  checkpoint it's paired with.
- **Checkpoint+LoRA combo usage** — which LoRA is most used specifically *with a given checkpoint*.
  A LoRA can dominate the overall list while actually being tied to just one checkpoint — the
  combo view is what tells the two apart. In practice this list gets dominated by whichever
  checkpoint had the most images fetched, since it's a raw count, not normalised per-checkpoint.
- **Bare checkpoint usage** — does a checkpoint get used WITH a LoRA at all, or mostly on its own?
  Split into three buckets, not two: `with_lora`, `bare` (resource metadata present, genuinely no
  LoRA), and `no_meta` (civitaiResources missing/empty entirely — usually means the image wasn't
  generated through Civitai's own on-site generator, e.g. an A1111/ComfyUI upload; this is a
  metadata gap, NOT evidence the checkpoint was used bare, so don't conflate `no_meta` with `bare`).

#### Why reactions are ranked client-side, not via the API

Civitai's own `sort=Most Reactions` on `/images` doesn't reliably combine with `withMeta` + cursor
pagination the way `sort=Newest` does. So `images.py` always fetches the full window newest-first
(same approach the original working `fetch_all` used, restored — see Discoveries) and ranks by a
combined `reactionScore` (sum of like/heart/laugh/cry/comment counts) afterward, itself.

### probe — activity ranking only, no image fetching

```bash
uv run python -m civitai_fetcher.probe
```

Zero-arg default does the full ranking pipeline in one reasonable-length run: discover a
download-ranked candidate pool, probe each for recent post activity (Tier 2), then measure
sustained daily velocity for the top 150 (Tier 3b) to properly rank the models that tie at the
Tier 2 cap. Writes `probe_results_<period>_<ddMMMyy>.csv`, `probe_report_<period>_<ddMMMyy>.html`,
`probe_distribution_<period>_<ddMMMyy>.png`.

Defaults (all tuned from real runs, see **Discoveries** below):
`--candidate-count 1000`, `--period Month`, `--page-limit 100`, `--deep-probe-limit 150`,
`--velocity-top-n 150`, `--velocity-max-pages 400`. `--only-ids` probes specific model IDs instead
of a fresh ranked pool.

`--since-days` and `--velocity-window-days` are **auto-derived from `--period`** when not
explicitly set — `Day`→1, `Week`→7, `Month`→30 — so they stay internally consistent by default
(e.g. `--period Day` alone no longer silently pairs a 1-day activity count with a 3-day velocity
window). Year/AllTime deliberately aren't auto-scaled: blindly extending to 365 days would make
every model probe walk back a full year of pages, a much more expensive run than picking
`--period Year` implies on its own — override `--since-days`/`--velocity-window-days` manually if
you want a longer window there.

Progress markers while it runs: `.` = clean request, `o` = succeeded after a retry (early
rate-limit warning), `X` = gave up after all retries.

The CLI prints a warning (with the specific fix) when a limit is actually constraining your
results — Tier 2 ties unresolved at `--deep-probe-limit`, or Tier 3b velocity hitting
`--velocity-max-pages`. If you don't see a warning, that limit isn't your bottleneck right now.
Note: a candidate pool capped by Civitai's own ranking cursor (see **Discoveries** below) is *not*
currently warned about — check the printed candidate count against `--candidate-count` yourself if
you suspect this.

### Trend analysis (standalone)

```bash
uv run python analyse_trends.py civitai_output.json [max_span_days]
```

Reads a fetcher output JSON and writes `trend_summary.csv`, `images_per_day.png`,
`posts_per_day.png`, `creators_per_day.png` (if a creator field is present), and
`resource_usage_per_day.png` (if `civitaiResources`/`meta.civitaiResources` is present).
Optional `max_span_days` excludes models whose data spans an unusually wide date range
(usually a low-activity model padded out with old images).

## Output shape (per image, from images_cli)

```json
{
  "modelId": 123,
  "modelName": "...",
  "modelVersionId": 456,
  "modelUrl": "https://civitai.red/models/123",
  "imageId": 789,
  "imageUrl": "https://image.civitai.com/...",
  "posterUsername": "...",
  "postId": 1011,
  "postUrl": "https://civitai.red/posts/1011",
  "width": 832,
  "height": 1216,
  "createdAt": "...",
  "nsfwLevel": 1,
  "stats": { "likeCount": 10, "heartCount": 5, "...": "..." },
  "reactionScore": 15,
  "meta": { "prompt": "...", "civitaiResources": [ { "type": "lora", "modelVersionId": 999, "name": "...", "versionName": "...", "creatorUsername": "..." } ], "...": "..." }
}
```

Image-centric, not model-centric: one entry per image, model fields repeated per row. This suits
filtering/aggregation and eventual embedding better than nesting images under models.

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
- `civitaiResources` is only populated for images generated through Civitai's own on-site
  generator. Uploads from A1111/ComfyUI/etc. have empty/missing `civitaiResources` even though a
  LoRA may well have been used — see "bare checkpoint usage" above for why this matters.

## Discoveries (read before re-tuning any of this)

Findings from real runs against the live API, kept here so we don't re-derive them from scratch.

**`total_probe_count` measures posts, not generations.** Tier 2 counts images posted in the
window; it's a proxy for generation activity, not a direct measure — we don't know the
post:generation ratio, and can't from this API.

**Download rank and current activity correlate weakly.** Spearman correlation between
`download_rank` and `total_probe_count` has consistently come out around -0.40 to -0.45 across
Month-period runs — a real signal, but download-popularity is a mediocre proxy for what's active
*right now*. This is the whole reason the probe exists rather than just reading the
downloads-ranked list. Shorter periods (Day/Week) show weaker/noisier correlation still, since the
candidate pool itself is much smaller and more volatile (see next point).

**`--period Day`'s candidate pool is volatile hour-to-hour, not a stable ranking.** Comparing two
Day runs 9 hours apart, over half the 13-15 model candidate pool churned, and the single most
active model in one snapshot (200+ images/day) had vanished from the ranking entirely by the next
run. Day is a genuine point-in-time snapshot, not something to treat as stable across a session.

**Week gives the best signal-to-noise for finding images worth looking at.** Day's pool (~15
models) is too thin and too volatile to be representative. Month's pool (~350 models) is
representative but by the time you've scrolled past a hundred low-signal models you're fatigued
before reaching anything worth a closer look. Week (~100-110 models) sits in the middle — enough
diversity without the fatigue. This is why `images_cli.py` defaults to `--period Week`.

**Activity ranking and reaction quality are genuinely different signals — don't conflate them.**
In one real run, the two highest-velocity models (367 and 363 images/day) topped out at
reactionScore 417 and 184 on their single best image. Two mid-activity models (61 and 118
images/day — ranked #7 and #3 by activity) had best images at reactionScore 1226 and 1652 — 3-9x
higher despite far lower posting volume. High activity tells you what's being posted a lot; high
reactions tell you what's actually landing with people. Use `--top-reactions-per-model` rather
than a flat global top-N if you want to compare across models instead of just seeing whichever
single model has the highest-reaction outlier crowd out everything else.

**Raising the Tier 2 count cap does not resolve ties — the top of the distribution is a
long/power-law tail.** Tested directly: at `--deep-probe-limit 150`, 137/345 models were tied at
the cap. Raising the cap to 1000 only shrank that to 37 — and the pattern (`≥150→137, ≥200→113,
≥300→85, ≥500→61, ≥750→45, ≥1000→37`) doesn't converge, it just keeps peeling off a similar
fraction. **No cap will fully resolve this.** A "Tier 3" that re-probed the top N with a bigger
count cap was built, tested, confirmed to have this diminishing-returns problem, and was then
**removed from the code**. Don't rebuild it — see the next point for what actually works.

**Day-bucketed velocity is the right metric, not a bigger count cap.** Instead of counting posts
up to an arbitrary cap, `probe_recent_velocity` (Tier 3b, `--velocity-top-n`) walks back a fixed
*time* window (`--velocity-window-days`) and buckets posts by calendar day. This is bounded by
time, not count, so it's cheap regardless of how active a model is, and it directly answers "how
many posts/day, sustained" instead of "did you hit my arbitrary ceiling." In one real run this
found a model tied with 18 others at the Tier 2 cap that was actually running 946 images/day —
2.5x the next-highest model — once measured properly. That ranking was completely invisible to
Tier 2 alone.

**A single time-windowed average can hide a viral one-day spike — `burst_ratio` catches it.**
Two models with the same `velocity_per_day` can have totally different shapes: one steady, one a
single viral day followed by silence. `probe_recent_velocity` also reports `max_single_day` and
`burst_ratio` (peak day ÷ window average). `burst_ratio` near 1.0 = sustained; well above 1 =
spiky/viral, treat the average with caution.

**`--candidate-count` has an API-side ceiling, not a code-side one.** Requesting
`--candidate-count 1000` for `--period Month --types Checkpoint` returned only 345 models —
`get_popular_models`'s pagination loop exited because Civitai's own ranking cursor ran out (empty
page or no `nextCursor`), confirmed by `download_rank` topping out exactly at 345 with zero
duplicate `modelId`s. Raising `--candidate-count` further does nothing for that specific
`period`+`types` combo. To get more candidates: widen `--period` (e.g. `AllTime` has a deeper
cursor than `Month`) or drop `--types`. This isn't auto-warned — check the printed candidate count
against what you asked for if you suspect it.

**A `0` in `total_probe_count`/`page1_count` can mean two different things.** Genuinely zero
recent posts vs. the probe silently erroring (deleted model, bad ID, transient network fault) both
produced the same all-zero result dict before `probe_status` (`ok` / `error` / `no_model_id`) was
added — check `probe_status` before treating a `0` as a real signal.

**`503` needs the same retry/backoff treatment as `429` — treating it as a hard failure loses
real data.** Civitai's own `503` error body literally says *"temporarily overloaded — please
retry"*, but an earlier version of `client.py` lumped it in with genuine client errors (400/403)
that fail fast without retrying. During one real API overload, this caused 44 version-fetches to
be abandoned on the very first `503`, at one point silently dropping an entire top-ranked model's
1439 images from the output. Fixed: `500`/`502`/`503`/`504` now retry with backoff like `429`;
only genuine 4xx client errors fail fast.

**A phase timeout that "abandons" slow work still blocks on it anyway if using
`with ThreadPoolExecutor()`, and then discards the result — worst of both.** The context-manager
form of `ThreadPoolExecutor` always calls `shutdown(wait=True)` on exit regardless of what you
pass while running, so a "give up after N seconds" pattern doesn't actually save any time if a
future is already mid-flight — it just blocks until that future finishes anyway, then throws the
completed result away. Caught this exact bug costing an entire model's fetched images (thrown away
after actually completing, for zero time saved). Fixed by using two different wait strategies for
two different needs: `_wait_with_heartbeat` (activity.py) genuinely abandons slow work — correct
there, since a stuck model shouldn't block an otherwise-fast probing run and losing one probe
result is cheap. `_wait_all_with_heartbeat` (images.py) waits for every future no matter how long
it takes — correct there, since losing a completed image fetch is expensive and there was no time
being saved by abandoning it anyway.

**Defaults were tuned up from a run that showed clear headroom.** A `--candidate-count 1000
--velocity-top-n 150` run completed with almost no `o` markers (retry-then-succeed) and zero `X`
markers (exhausted retries) — meaning the API had capacity to spare at those settings. Current
defaults are roughly double what was proven safe, specifically so a **zero-argument run**
(`python -m civitai_fetcher.probe`) does something useful in one shot without hand-tuning. If you
ever see the cap-hit warnings mentioned above during a default run, that's the signal these need
raising again — don't guess, read the warning, it names the exact flag.

## Roadmap

- [ ] Qdrant loader: embed `prompt` (+ optionally `negativePrompt`), rest as filterable payload
- [ ] Optional CLIP pass for character/landscape + gender classification
- [ ] Normalise checkpoint+LoRA combo usage per-checkpoint (currently raw counts, so a
      high-volume checkpoint dominates the list regardless of how *consistently* it uses a LoRA)