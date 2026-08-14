# PLAN.md — Image-to-Song, rebuilt around the core pipeline

## Objective

Rebuild this project from scratch in a new repo, keeping only the core loop:
a user uploads an image and picks a genre on a knob; the image is embedded into
SigLIP space and scored against a fixed adjective vocabulary to pull its top 10
tags; genre + tags are composed into a prompt and sent to Suno via apiframe;
the rendered song (both takes) is saved to local history, viewable and
re-listenable in the same page. Same stack (FastAPI + Next.js), same frontend
shell (upload, genre knob, generate button, history list with tag scores,
double-click-armed clear-history button). Everything after song generation —
cover art, the 2.5D visualizer, MP4 export, and publishing to
Instagram/TikTok/YouTube — is cut, not stubbed. "Upload an MP3" is cut too,
since nothing downstream would ever act on it. Done means: a fresh repo where
`uvicorn main:app` and `npm run dev` produce a working local app that
reproduces image → tags → Suno song → local history, with no dead code paths
left over from the parts that were removed. The result is meant to be a clean
testbed for iterating on the tagging/prompting pipeline itself — that
iteration work is out of scope for this plan, which only builds the testbed.

## Context

**This plan was written from, and refers throughout to, an existing repo on
this same machine, at the absolute path:**
`/Users/tomragus/Desktop/local/GitHub Projects/image-to-song-v1` (referred to
below as `SOURCE_REPO`). This document is being placed in a different,
separate, likely-empty repo — it has no relationship to `SOURCE_REPO` other
than what's written here. Every file reference below is a full path into
`SOURCE_REPO`, readable from anywhere on this machine regardless of which
directory the executing session's shell is in. If `SOURCE_REPO` has moved,
been deleted, or this plan is being run on a different machine, steps 2 and 3
(verbatim ports) cannot be completed as written — the source files must be
supplied some other way (e.g. pasted inline) before proceeding.

Files read from `SOURCE_REPO` before writing this plan:

- `SOURCE_REPO/backend/siglip_tagger.py` — embeds every word in `vocabulary.ADJECTIVES`
  once (averaged over 4 templates in `TEMPLATES`), then per-facet top-k
  (`QUOTAS`) against the image embedding, calibrated through SigLIP's own
  `logit_scale`/`logit_bias`. Lazy-loads torch/transformers inside `_load()`,
  guarded by a threading lock since FastAPI runs sync handlers on a
  threadpool. Model: `google/siglip2-base-patch16-224`. This file ports
  essentially unchanged.
- `SOURCE_REPO/backend/vocabulary.py` — the 6 facets (`colour`, `light`, `mood`,
  `texture`, `energy`, `style`), their adjective lists, `QUOTAS` (sums to 10,
  weighted toward mood), and `TEMPLATES`. Copies verbatim — this is the
  vocabulary the task explicitly says to keep.
- `SOURCE_REPO/backend/genres.py` — `GENRES` (ordered by family, not alphabetically, for
  a knob you spin) and `PROMPT_TERMS` (label-on-knob vs. text-sent-to-Suno,
  e.g. "house" → "deep house" via `prompt_term()`). Copies verbatim.
- `SOURCE_REPO/backend/main.py` (1856 lines currently) — only a slice is relevant:
  `apiframe_call`/`wait_for_job`/`request_song` (job-submit-and-poll against
  apiframe's v2 API, `SUNO_MODEL_VERSION = "V5_5"`, instrumental only),
  `read_image` (content-type allowlist), `save_to_history` (writes the image
  + both rendered takes to `backend/history/`, appends one row to
  `history.jsonl`), `iter_history`/`read_history`, `wipe_history` (the
  double-click-armed clear), and the `/generate`, `/genres`, `/history` routes
  themselves. Conventions to preserve: heavy imports (torch, PIL) stay lazy
  inside the functions that need them so `import main` stays fast; external
  API wrappers raise `HTTPException` with the upstream error text passed
  through, never swallowed. Everything about covers, visualizer, export,
  Buffer, R2, and YouTube (artwork/motion/ffmpeg/publish routes, ~1200 of the
  1856 lines) is cut, along with `/upload` and its `read_audio`.
- `SOURCE_REPO/frontend/app/page.tsx` (~1500 lines) and
  `SOURCE_REPO/frontend/app/GenreKnob.tsx` (155
  lines, unchanged design — drag-to-rotate genre picker, 12°/genre). The
  slice to keep: the `Generation`/`Track` types, the drop-zone file picker,
  `GenreKnob` wiring (`query` state → `genre` derived from it →
  `GenreKnob`'s `value`/`onChange`), `handleGenerate` (posts to
  `/generate?genre=...`), the history list rendering `entry.tags` with
  `entry.tag_scores[tag]` next to `entry.prompt` and playable `<audio>` per
  track, and the `armed`/`setArmed` double-click clear-history flow
  (`DELETE /history`). Cut: `Cover`/`Video`/`Export`/`Publish`/
  `YoutubeUpload` types and all UI/state that references them, the MP3
  upload control, everything downstream of "song generated."
- `SOURCE_REPO/frontend/AGENTS.md` / `CLAUDE.md` — this Next.js version has
  training-data-breaking changes; the bundled docs at
  `node_modules/next/dist/docs/` (in whatever repo ends up hosting the new
  frontend's `node_modules`, not `SOURCE_REPO`'s) are the source of truth.
  Carries over as a standing note in the new repo too.
- No test suite exists in `SOURCE_REPO`. `frontend/lint` is the only
  automated check there (`npm run lint`).

## Decisions

- **`/upload` (external MP3 import) is removed entirely** — no endpoint, no
  UI. Nothing downstream exists for it to feed.
- **Prompt/tag optimization work is explicitly out of scope for this plan's
  steps** — the user already has a plan for that and wants it left as an open
  door on top of a clean pipeline, not built here.
- **Delegation**: `ai-engineer` owns backend/frontend implementation;
  `qa-eval-specialist` is not used in this plan (no eval-harness step was
  requested) but remains the natural owner if/when the optimization work
  gets its own plan.
- **New repo location**: no new folder or `git init` — PLAN.md is written
  into `SOURCE_REPO`'s root (`SOURCE_REPO/PLAN.md`) for the user to carry
  into a fresh, separate repo themselves. All steps below assume they're
  being executed from *inside that new repo*, with `SOURCE_REPO` reachable
  only via the absolute path above, not as an ancestor/sibling directory or
  anything else implicit.
- **Frontend scope**: upload, genre knob, generate button, history list
  (including per-tag SigLIP scores, already present in the current list
  rendering), and the armed clear-history button. Nothing else.
- **File path for this document**: `PLAN.md` at repo root, restoring the
  convention visible in `git status` (a prior `PLAN.md` was deleted).

## Steps

### 1. Scaffold the fresh repo skeleton
Files: repo root, `backend/`, `frontend/`, `.gitignore`, `backend/.env.example`, `backend/requirements.txt`, `frontend/package.json` (via `create-next-app` matching the current Next.js version)
Execution: [main]
Done when: empty `backend/` and `frontend/` directories exist with baseline config files in place, `.env.example` lists `APIFRAME_API_KEY` only (no OpenAI/Immersity/Buffer/R2/YouTube keys), and `.gitignore` excludes `.env`, `backend/history/`, and standard Next.js/Python artifacts.

### 2. Port `vocabulary.py` and `genres.py` verbatim
Files: copy `SOURCE_REPO/backend/vocabulary.py` → `backend/vocabulary.py`, `SOURCE_REPO/backend/genres.py` → `backend/genres.py` (source paths are absolute, outside the new repo; destination paths are relative to the new repo root)
Execution: [main]
Done when: both files exist in the new repo's `backend/`, byte-identical to their `SOURCE_REPO` originals.

### 3. Port `siglip_tagger.py` verbatim
Files: copy `SOURCE_REPO/backend/siglip_tagger.py` → `backend/siglip_tagger.py`
Execution: [main]
Done when: the file exists in the new repo, byte-identical to the `SOURCE_REPO` original; only its import of `vocabulary` needs to resolve in the new layout.

### 4. Build the minimal FastAPI backend
Files: `backend/main.py`, `backend/requirements.txt`
Execution: [delegate to ai-engineer]
Done when: `uvicorn main:app --reload` starts cleanly; `GET /` reports whether `APIFRAME_API_KEY` loaded; `GET /genres` returns `GENRES`; `POST /generate` (image + optional genre) runs SigLIP tagging, builds the prompt via `prompt_term`, calls apiframe's Suno endpoint through `apiframe_call`/`wait_for_job`/`request_song`, downloads both takes, and writes them plus a `history.jsonl` row via a ported `save_to_history`; `GET /history` returns rows newest-first; `DELETE /history` wipes `history.jsonl` and the saved images/audio; static files under `backend/history/` are served for playback. No cover/visualizer/export/publish/YouTube/upload code paths exist anywhere in the file.

### 5. Build the minimal frontend
Files: `frontend/app/page.tsx`, `frontend/app/GenreKnob.tsx` (ported unchanged)
Execution: [delegate to ai-engineer]
Done when: the page renders a drop-zone image picker, the genre knob wired to a search box exactly as today, a generate button that posts to `/generate` and shows a loading state, a history list showing each run's image, genre, tags with their SigLIP scores, prompt, and playable tracks, and a clear-history button that requires a second click within the existing arm/disarm window before it calls `DELETE /history`. No type, state, or UI element referencing covers, videos, exports, publishes, or YouTube remains.

### 6. End-to-end smoke test
Files: none (manual verification)
Execution: [main]
Done when: with both servers running locally and a real `APIFRAME_API_KEY` in `backend/.env`, uploading an image and picking a genre in the browser produces a saved history row with two playable tracks, tags with scores are visible, and clearing history (double-click) empties the list and the `backend/history/` folder.

### 7. Write the new repo's README
Files: `README.md` (new repo root)
Execution: [main]
Done when: the README documents what the app does (one paragraph), how to run it (`backend/.env` setup, `uvicorn`/`npm run dev` commands, matching the current repo's launcher-free dev instructions), and states explicitly that this repo stops at song generation by design.

## Out of scope

- Cover art generation, the 2.5D visualizer, MP4 export, and publishing to
  Instagram/TikTok/YouTube — cut entirely, not stubbed or feature-flagged.
- `/upload` (bringing in an externally-made MP3) — removed, since nothing
  downstream would ever consume it.
- Any eval harness, prompt-variant comparison tooling, or systematic
  optimization of the vocabulary/quotas/templates/prompt construction — the
  user has a separate plan for this; this plan only builds the clean
  pipeline it would run against.
- A double-clickable `.command` launcher script (`start-ideagram.command`
  in the current repo) — not mentioned in the request; add only if asked.
- Migrating or importing history from `SOURCE_REPO/backend/history/` — the
  new repo starts with empty history.

## Risks

- **Suno prompt behavior may not transfer 1:1.** The current prompt
  construction (`genre, tag1, tag2, ...` via `prompt_term` then plain join)
  was tuned against `SOURCE_REPO`'s exact flow; copying it verbatim is the
  right starting point, but it's still the piece the user intends to
  optimize next, so treat step 4's prompt-building code as a known
  first-draft, not a settled design.
- **`create-next-app`'s current default may not match the pinned Next.js
  version `SOURCE_REPO` relies on.** `SOURCE_REPO/frontend/AGENTS.md` warns
  this version has training-data-breaking API changes; scaffolding the new
  frontend needs to check `node_modules/next/dist/docs/` in the *new*
  install, not assume `SOURCE_REPO`'s docs still apply verbatim if the
  resolved version differs.
- **`save_to_history`'s file-naming and dedupe logic (timestamp+uuid stems,
  same-second collision handling) is easy to under-port** since it's
  entangled with genre/title formatting helpers (`audio_filename`,
  `safe_for_filename`) that live outside the slice read for this plan;
  step 4 should re-read those helpers directly from
  `SOURCE_REPO/backend/main.py` before writing the new version rather than
  reconstructing them from memory.
- **This entire plan is only executable while `SOURCE_REPO` still exists at
  the path above.** If it's moved, renamed, or deleted before a fresh
  session picks this plan up, steps 2–5 lose their source of truth and
  someone will need to re-supply the relevant file contents by hand.

---

**Delegation tags in this plan**: steps 4 and 5 are tagged `[delegate to
ai-engineer]`. Every other step (1, 2, 3, 6, 7) is `[main]`. If every tagged
step were executed, that's **2 ai-engineer delegations** total. Trigger each
one yourself, per the delegation gate in CLAUDE.md, when you're ready.
