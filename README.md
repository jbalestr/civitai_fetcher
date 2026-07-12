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
Tier 2 cap. Writes `probe_results.csv`, `probe_report.html`, and `probe_distribution.png`.

Defaults (all tuned from real runs, see **Discoveries** below):
`--candidate-count 1000`, `--period Month`, `--since-days 30`, `--page-limit 100`,
`--deep-probe-limit 150`, `--velocity-top-n 150`, `--velocity-window-days 3`,
`--velocity-max-pages 400`. `--only-ids` probes specific model IDs instead of a fresh ranked pool.

Progress markers while it runs: `.` = clean request, `o` = succeeded after a retry (early
rate-limit warning), `X` = gave up after 3 retries.

The CLI prints a warning (with the specific fix) when a limit is actually constraining your
results — Tier 2 ties unresolved at `--deep-probe-limit`, or Tier 3b velocity hitting
`--velocity-max-pages`. If you don't see a warning, that limit isn't your bottleneck right now.
Note: