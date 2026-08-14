# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repository is pre-implementation: it contains only `PLAN.md` and no
application code, no commits, and no build/lint/test tooling yet. Before
doing any work here, read `PLAN.md` in full — it is the authoritative spec
for what this repo is meant to become and supersedes any assumption below
about "the codebase" until the plan's steps have actually been executed.

## What this repo is for

A from-scratch rebuild of an image-to-song app (FastAPI backend + Next.js
frontend), keeping only the core loop: upload an image, pick a genre on a
knob, embed the image with SigLIP and score it against a fixed adjective
vocabulary for top-10 tags, compose genre + tags into a prompt sent to Suno
via apiframe, save both rendered takes to local history. Cover art, the
2.5D visualizer, MP4 export, and social publishing are deliberately cut,
not stubbed.

The plan is executed by porting specific files from a sibling repo,
`SOURCE_REPO` = `/Users/tomragus/Desktop/local/GitHub Projects/image-to-song-v1`
(absolute path, not an ancestor/sibling reachable by relative path). Steps
2–5 of `PLAN.md` depend on that repo still existing at that path — if it has
moved or been deleted, re-check with the user before improvising
replacements for the "verbatim port" steps.

## Working from PLAN.md

- `PLAN.md` encodes step-by-step execution tags: `[main]` steps are done
  directly; `[delegate to ai-engineer]` steps are only triggered by the user
  explicitly invoking delegation (see the global CLAUDE.md delegation gate —
  do not self-route into the agent framework here).
- The "Decisions", "Out of scope", and "Risks" sections of `PLAN.md` are load
  -bearing — e.g. `/upload` (MP3 import) is permanently cut, prompt/tag
  optimization is explicitly out of scope for this plan's steps, and
  `save_to_history`'s filename/dedupe helpers live outside the file slice
  already read into the plan and need to be re-read from `SOURCE_REPO`
  before step 4 reconstructs them.
- Once steps 1–5 are executed, the expected dev loop (per the plan) is:
  backend via `uvicorn main:app --reload` in `backend/`, frontend via
  `npm run dev` in `frontend/`. `backend/.env` needs `APIFRAME_API_KEY` only.
  There is no test suite planned; `npm run lint` is the only automated
  check called for.

## Conventions to carry over from the source repo (per PLAN.md)

- Heavy imports (`torch`, `PIL`, transformers) stay lazy, imported inside
  the functions that need them, so `import main` stays fast.
- External API wrappers (apiframe calls) raise `HTTPException` with the
  upstream error text passed through — never swallowed.
- `vocabulary.py`, `genres.py`, and `siglip_tagger.py` are ported verbatim
  from `SOURCE_REPO`, byte-identical except for import paths.
