# SigLIP tag judge eval

Runs every image in `test-images/` through the app's SigLIP tagger, then asks
Claude Sonnet 5 (vision) to flag any of the 10 tags that don't actually describe
the image. Produces a raw data file and a human-readable HTML report. Manual,
local-only, no CI integration.

## Prerequisites

- `ANTHROPIC_API_KEY` set in `backend/.env` (same file/convention as
  `APIFRAME_API_KEY`).
- `backend/.venv` already has everything else needed (`torch`, `transformers`,
  `pillow`, `anthropic`).

## Run

```
backend/.venv/bin/python eval/judge.py
```

Run from the repo root (or adjust paths accordingly -- `judge.py` resolves
`test-images/` and `backend/` relative to its own location, not the cwd).

This calls `siglip_tags()` directly from `backend/embedding_tagger.py` -- it never
starts the FastAPI app, never calls `/generate`, and never touches
`apiframe_call`/`request_song`/Suno. Uses `ANTHROPIC_API_KEY` (real, billed API
calls -- one Claude Sonnet 5 call per image), no Suno/apiframe involved.

36 images, one Claude call each (all 10 tags judged together per image, not one
call per tag). The SigLIP model itself lazy-loads on the first call (~375MB of
weights) and is cached in memory for the rest of the run.

Output:
- `eval/results.json` -- one entry per image: filename, and each of its 10 tags
  with SigLIP score, judge verdict (`flagged`), and `reason` (empty string when
  not flagged). An image that errors out (judge timeout, malformed response)
  gets a `{"filename": ..., "error": ...}` entry instead of tags, so one bad
  image doesn't kill the run or silently vanish from the output.
- `eval/report.html` -- open directly in a browser, no server needed. Every
  image's thumbnail is embedded as base64 JPEG (self-contained, works offline).
  Flagged tags are highlighted red; hover one to see the judge's reason.

## Regenerating just the report

If you already have `eval/results.json` and only want to rebuild the HTML (e.g.
after tweaking `report.py`'s styling), run from inside `eval/`:

```
cd eval
../backend/.venv/bin/python report.py
```

## Design notes

- **Judge prompt** (`JUDGE_PROMPT_TEMPLATE` in `judge.py`): tells the judge the
  10 words were chosen by embedding similarity against a fixed vocabulary, not
  hand-picked for the image, and explicitly asks it to give tags the benefit of
  the doubt -- these are meant to be evocative/moodboard-style, not literal
  captions. It flags a word only when a reasonable viewer would call it
  outright wrong (e.g. "aquatic" for a desert photo), not merely a non-obvious
  choice. This is a deliberate design choice to keep the judge from flagging
  half of every image's tags on the grounds that a mood word is "not the most
  obvious" description -- see PLAN-eval.md's risk note that this signal is
  soft, not ground truth.
- **Structured output**: uses the Claude API's `output_config.format`
  (`json_schema`) so the judge returns one verdict per tag in a fixed shape,
  rather than depending on prompt engineering for valid JSON.
- **Determinism**: not pinned. Claude Sonnet 5 doesn't accept `temperature`
  (non-default sampling parameters are rejected on this model), so run-to-run
  identical flags aren't guaranteed. If you need to check run-to-run stability,
  re-run the script and diff `results.json`'s `flagged` values.
- **Image resizing**: the image sent to the judge is resized to max 1024px on
  the long side (vision models don't gain anything from full-resolution
  originals and gemma3 downsamples internally regardless); report thumbnails
  are separately resized to 320px JPEG to keep `report.html` small.
- **Out of scope** (per `PLAN-eval.md`): no changes to `vocabulary.py`, no
  ground-truth manifest, no CI integration, no comparison against any model
  other than `gemma3:12b`. `embedding_tagger.py` is no longer entirely out of
  scope: the live app's `/tag` now calls the judge (`backend/judge.py`,
  the same module this script imports) and drops any flagged tag from the
  list outright via `embedding_tagger.drop_flagged_tags()` -- no replacement
  is picked. This eval script itself still only reports what the judge flags
  -- it does not exercise the drop path.
