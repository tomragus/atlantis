# Tag analysis card refinements

## Objective

Simplify the tag-analysis presentation across both run modes and streamline the
SigLIP-only flow. Concretely: drop the score-meaning note and the "Top 20 candidates"
section from every tag analysis card (live UI and the downloadable report), restrict
the embedding plot to only the final (Suno-bound) tags with a per-algorithm title,
keep that plot for the 4 Algorithm Suite but drop it entirely for the SigLIP-only
"Generate Song" path, remove the non-functional per-track Download buttons, alternate
image/tag-column placement across Suite-mode rows, and auto-chain SigLIP tagging
straight into song generation without a manual second click. Done when all of the
above render correctly in a live browser check, `npm run lint` is clean, and the
backend's `pytest` suite passes with its `top20`-dependent tests updated to match the
removed field.

## Context

Read `frontend/app/page.tsx` (`EmbeddingPlot`, `TagBreakdown`, `RunRow`,
`algorithmSectionHtml`/`buildReportHtml`, `tagOneAlgorithm`/`startRun`/`handleGenerate`,
`downloadTrack`) and `backend/main.py` (`build_tagging_summary`, the `/tag` route's
`embedding_projection` call) before planning. Key findings:

- **Score-meaning note**: `TagBreakdown` (`page.tsx:687-781`), the calibrated/
  uncalibrated branches at `~713-731` ("The number beside each tag is...").
  UI-only — the downloadable report's `algorithmSectionHtml` never included this copy,
  so no change needed there.
- **Top 20 candidates**: rendered twice — live UI at `TagBreakdown`'s
  `~755-771` (`tagging.top20.length > 0 && ...`), and in the downloadable report at
  `algorithmSectionHtml`'s `~309-315`. Backing data comes from
  `build_tagging_summary` in `backend/main.py` (`~298-345`): `top20` is assembled
  (guaranteeing every final tag appears, then filling to 20 with runners-up) purely to
  feed this UI section and the embedding plot's point set (the `/tag` route passes
  `tagging["top20"]`'s `(tag, facet)` pairs into `embedding_projection`,
  `main.py:~477-483`).
- **Embedding plot**: `EmbeddingPlot` (`page.tsx:434-621`), rendered from `RunRow`
  (`~811-820`) with the label `"embedding space (illustrative)"`. Its `points` come
  straight from `embedding.points` (currently the top-20 set); its SVG `aria-label`
  (`~517`) also says "top 20 tag candidates," which will be stale once the point set
  changes and needs updating alongside it. `algorithmLabel` is already threaded down
  to `RunRow`/`EmbeddingPlot`, so a per-algorithm title needs no new plumbing.
- **RunRow structure** (`page.tsx:787-893`): `.run-row-media` (image + embedding
  section) then `.run-row-body` (tag breakdown + generate/audio) — a fixed
  image-left, tags-right order, no per-row variation today. Rendered via
  `runAlgorithms.map((algorithm) => <RunRow ... />)` at `~1313`, no index currently
  passed in. `runAlgorithms.length > 1` is the existing signal Suite mode already uses
  elsewhere (the "Generate all" actions block) to distinguish it from the SigLIP-only
  single-row case — reusable here instead of checking the algorithm id directly.
- **Per-track Download button**: `RunRow`'s track-mapping block (`~867-887`), calls
  module-level `downloadTrack` (`page.tsx:198-203`). That function has exactly one call
  site — removing the button leaves it dead code to delete too. `downloadBlob`
  (`~181-190`) is shared with `handleDownloadReport` and stays.
- **SigLIP-only flow**: `startRun(["siglip"])` (button in the `upload-card`) kicks off
  `tagOneAlgorithm` (`~1058-1085`) via `Promise.all` in `startRun` (`~1090-1111`), then
  commits `results` once settled. `handleGenerate` (`~1113-...`) reads
  `results[algorithm]` from component state (not a passed-in value), so it must be
  triggered *after* `results` has actually re-rendered with the ready analysis, not
  synchronously inside `startRun`'s `.then()` — a `useEffect` watching `results` for
  the single-algorithm, ready-and-not-yet-generated case is the safe way to auto-chain
  it without relying on a stale closure.
- **Backend `top20` removal** (per interview decision): `build_tagging_summary`'s
  `top20` computation/key (`main.py:~314-341`) is going away entirely, and the `/tag`
  route's `embedding_projection` call switches from `tagging["top20"]` to
  `tagging["final"]`'s `(tag, facet)` pairs. Tests referencing `top20`:
  `backend/tests/test_main_helpers.py` (three dedicated tests —
  `test_build_tagging_summary_...` around lines 47, 52-58, 65-73) and one stub value in
  `backend/tests/test_generate_endpoint.py:40` (an arbitrary JSON blob for a `/generate`
  test, not schema-checked — trimming the key there is optional cleanup, not required).

## Decisions

- **Embedding plot scope**: backend recomputes PCA over final tags only (`main.py`'s
  `/tag` route passes `tagging["final"]` into `embedding_projection`), not a
  frontend-only filter of the existing top-20-fit geometry.
- **Backend `top20` field**: deleted entirely — `build_tagging_summary`'s top20 logic
  and key are removed, with `test_main_helpers.py`'s three top20-specific tests updated
  to match (removed or rewritten against `final`, whichever the code change leaves
  testable).
- **Downloadable report**: Top 20 candidates section removed there too, matching the
  live UI exactly.
- **SigLIP-only flow**: auto-chain — tagging completes, generation fires automatically
  with no second click, but the Tag Analysis card (tags, judge results, prompt) still
  renders once ready, above the "Composing..."/audio result. Not a fully hidden
  loading-only flow.
- **Scope boundary**: only the changes explicitly listed. No incidental prop/type
  cleanup beyond what's naturally dead after these edits (e.g. `downloadTrack`), no
  broader `TagBreakdown` signature rework.
- **Delegation**: main thread only, no subagents.
- **Plan file path**: `plans/tag-analysis-refinements.md`.

## Steps

### 1. Remove the score-meaning note from TagBreakdown
Files: `frontend/app/page.tsx` (`TagBreakdown`, ~lines 713-731)
Execution: [main]
Done when: the `<p className="note">` block with the calibrated/uncalibrated
"The number beside each tag is..." copy is deleted; the tag chips row above it is
unchanged, and the `calibrated` variable/branching is removed if it has no other use
in the function.

### 2. Remove the "Top 20 candidates" section from the live UI
Files: `frontend/app/page.tsx` (`TagBreakdown`, ~lines 755-771)
Execution: [main]
Done when: the `tagging.top20.length > 0 && <div className="top20">...</div>` block is
deleted from `TagBreakdown`. (The `Tagging` type's `top20` field itself is addressed in
Step 6 once the backend stops sending it.)

### 3. Restrict the embedding plot to final tags and retitle it, backend-side
Files: `backend/main.py` (`/tag` route, ~lines 477-483)
Execution: [main]
Done when: the `embedding_projection` call is passed `tagging["final"]`'s
`(tag, facet)` pairs instead of `tagging["top20"]`'s; a fresh `/tag` response's
`tagging.embedding.points` contains only the final tag set (as few as 0, as many as
the quota allows — not always exactly 10).

### 4. Update the embedding plot's label, aria-label, and rendering to match
Files: `frontend/app/page.tsx` (`EmbeddingPlot` ~line 517, `RunRow` ~line 813)
Execution: [main]
Done when: the `meta-label` above the plot reads `{algorithmLabel} Joint Embedding
Space` (e.g. "SigLIP Joint Embedding Space", "CLIP Joint Embedding Space") instead of
"embedding space (illustrative)"; the SVG's `aria-label` no longer says "top 20 tag
candidates" and instead describes the final tag set; since every point now belongs to
`finalTags` by construction, the dimmed/non-final circle styling either naturally
stops triggering or is simplified away if it's confirmed dead.

### 5. Drop the embedding plot entirely for the SigLIP-only path
Files: `frontend/app/page.tsx` (`RunRow`, ~lines 802-820)
Execution: [main]
Done when: `RunRow`'s embedding-section render is gated on `runAlgorithms.length > 1`
(Suite mode) in addition to the existing `embedding && embedding.points.length > 0`
check — a single-row SigLIP-only run never shows the plot, a multi-row Suite run still
does.

### 6. Remove the "Top 20 candidates" table from the downloadable report
Files: `frontend/app/page.tsx` (`algorithmSectionHtml`, ~lines 309-315)
Execution: [main]
Done when: the `tagging.top20.length > 0 ? <h3>Top 20 candidates</h3>...` block is
removed from the generated HTML string; `tagTableRows`'s `selected` parameter (only
used by the top20 table) is removed if it has no other caller, and the `Tagging`
TypeScript type's `top20` field is removed once Step 3's backend change means it's no
longer sent.

### 7. Remove the backend's `top20` computation and field
Files: `backend/main.py` (`build_tagging_summary`, ~lines 298-345)
Execution: [main]
Done when: the `top20`-building block (final-tag-set/runners-up/sort) is deleted from
`build_tagging_summary`, and its return dict no longer has a `top20` key.

### 8. Update backend tests for the removed `top20` field
Files: `backend/tests/test_main_helpers.py`, `backend/tests/test_generate_endpoint.py`
Execution: [main]
Done when: `test_main_helpers.py`'s three `top20`-specific tests
(`test_build_tagging_summary_...`) are removed or rewritten to assert against
`final`/whatever `build_tagging_summary` still returns; `test_generate_endpoint.py:40`'s
stub JSON either drops the now-nonexistent `top20` key or is left as harmless unused
data (caller's judgment during implementation) — `pytest` passes either way.

### 9. Remove the per-track Download button
Files: `frontend/app/page.tsx` (`RunRow`'s track block ~lines 867-887, `downloadTrack`
~lines 198-203)
Execution: [main]
Done when: the `<div className="actions"><button onClick={() => downloadTrack(track)}>
⬇ Download</button></div>` block is removed from each rendered take; the now-unused
`downloadTrack` function is deleted (its only call site is gone); `<audio controls>`
itself is unchanged (the browser's built-in menu remains the download path).

### 10. Alternate image/tag-column placement across Suite-mode rows
Files: `frontend/app/page.tsx` (`RunRow` ~lines 787-893, its call site ~line 1313),
`frontend/app/globals.css` (`.run-row`, `.run-row-media`, `.run-row-body`)
Execution: [main]
Done when: `runAlgorithms.map` passes each row's index into `RunRow` (e.g.
`(algorithm, index) => <RunRow ... reversed={index % 2 === 1} />`); `RunRow` accepts a
`reversed` prop and swaps which side `.run-row-media` vs `.run-row-body` renders on
(image left/tags right for even indices, tags left/image right for odd indices) —
confirmed only in Suite mode, since the SigLIP-only path always renders exactly one row
(index 0, unreversed) and this is visually inert there regardless.

### 11. Auto-chain SigLIP-only tagging into generation
Files: `frontend/app/page.tsx` (near `tagOneAlgorithm`/`startRun`/`handleGenerate`,
~lines 1058-1170)
Execution: [main]
Done when: after a single-algorithm run (`runAlgorithms.length === 1`, currently always
`["siglip"]`) finishes tagging successfully, `handleGenerate` fires automatically with
no user click required — implemented via a `useEffect` watching `results` for the
ready-and-not-yet-generating-or-generated single-row case (avoiding the stale-closure
problem of calling it directly inside `startRun`'s `.then()`); the manual "Generate
song" button still renders and still works normally in Suite mode (untouched there);
the Tag Analysis card still appears once tagging completes, per the interview decision.

### 12. Lint, test, and manual verification
Files: none (verification only)
Execution: [main]
Done when: `npm run lint` (from `frontend/`) is clean (pre-existing `<img>` warning
aside); `pytest` (from `backend/`, venv active) passes; a manual browser pass with both
dev servers running confirms: SigLIP-only "Generate Song" goes straight from click to a
finished song with no manual second click, no embedding plot, and no Top 20/score-note
text, with the Tag Analysis card visible once ready; a 4 Algorithm Suite run shows all
four rows alternating image-left/image-right, each with its embedding plot titled
"`{Algorithm}` Joint Embedding Space" showing only final tags, no Top 20/score-note
text; no per-track Download button appears under any `<audio>` element in either mode;
downloading the tag-analysis report produces HTML with no Top 20 table.

## Out of scope

- Any further `TagBreakdown` prop/type simplification beyond what's naturally dead
  after these edits (e.g. `downloadTrack`, `top20`-related types).
- Changing the dimmed/non-final circle styling in `EmbeddingPlot` into a whole new
  visual treatment — if it becomes simplifiable because every point is now final, only
  simplify what's confirmed dead, don't redesign the plot's look.
- Anything about the Genre Knob, hero, or button-ratio work from the prior
  `ideagream-rebrand` plan — already implemented, untouched here.
- `eval/`, `musicgen-experiment/`, `check_facets.py`/`eval_tags.py`.

## Risks

- **Auto-chain UX**: firing generation immediately and automatically means a user can
  no longer review the SigLIP tag set and *choose* not to spend apiframe credits before
  a song renders — "Generate Song" now really does generate a song on one click, credits
  committed automatically. This is exactly what was asked, but it's a real behavior
  change worth a last look once it's live (previously a deliberate two-step gate).
- **PCA geometry shift**: fitting the embedding plot's PCA basis on ~10 final tags
  instead of 20 candidates changes the actual plotted layout, not just what's filtered
  from view — points may look more spread out or cluster differently than before. This
  is the intended, literal interpretation from the interview, but is a bigger visual
  change than a pure client-side filter would have been.
- **Backend `top20` removal touches tests**: deleting `build_tagging_summary`'s top20
  logic requires rewriting three existing backend tests; if any other backend code
  (eval/, dev tools) turns out to read the `top20` key beyond what grep found, that
  usage wasn't surfaced during research and could break silently until `pytest`/`eval`
  is run.
