# VLM demo media

This directory is intentionally empty of footage until a redistributable
aerial source is selected. Do not use VisDrone here if the showcase will be
posted publicly: its terms limit use to research and non-commercial purposes.

For every source, record its stable URL, licence, download date, permitted
use, and SHA-256 in `source_manifest.json`. Store source frames or short clips
beside the manifest, but do not commit them unless they are both tiny and
redistributable. The demo renderer accepts still-image cases; extracting one
representative RGB frame per case keeps the first version simple.

The manifest must contain exactly the four required case kinds: `caption`,
`answer`, `point`, and `abstention`. `abstention` cases must use a question
that can honestly produce a non-`ok`/non-`found` status. The renderer never
modifies the model response: it copies each CLI audit record verbatim into the
run's `predictions.jsonl` before rendering it.

Run the completed demo with:

```shell
uv sync --group vlm-inference --group vlm-demo
uv run python scripts/render_vlm_showcase.py \
  --manifest data/samples/vlm_demo/source_manifest.json \
  --output-dir runs/vlm_demo/<run_id> \
  --device auto --decode-mode schema
```
