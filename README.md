# civitai-fetcher

Pull recent images + generation metadata for Civitai's most popular models, concurrently,
validated, ready to feed into a downstream vector DB (Qdrant) pipeline.

## Status

Working: model discovery, concurrent image fetch, metadata filtering (`withMeta`), consistency
validation. **Not yet done:** Qdrant loader / embedding step — that's the next milestone, once
this fetcher's output shape is settled.

## Usage

```bash
uv sync
uv run python -m civitai_fetcher.cli --model-count 10 --images-per-model 20
```

To use an API token (optional, raises rate limits), set it in the shell first:

```powershell
# PowerShell
$env:CIVITAI_API_TOKEN = "your_token_here"
uv run python -m civitai_fetcher.cli --model-count 10 --images-per-model 20
```

```bash
# bash/zsh
export CIVITAI_API_TOKEN=your_token_here
uv run python -m civitai_fetcher.cli --model-count 10 --images-per-model 20
```

Writes:
- `civitai_output.json` — one flat record per image
- `civitai_output_issues.json` — only written if validation flags anything

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

## Roadmap

- [ ] Normalise categorical fields (`sampler`, `civitaiResources`) for consistent grouping
- [ ] Qdrant loader: embed `prompt` (+ optionally `negativePrompt`), rest as filterable payload
- [ ] Optional CLIP pass for character/landscape + gender classification
