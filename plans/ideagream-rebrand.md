# Ideagream Studio rebrand + hero/layout rework

## Objective

Rename the app from "Atlantis"/"Image to Song" to "Ideagream Studio" everywhere it
identifies itself (frontend tab title, on-page heading, backend FastAPI title, README
heading), add a one-line subtitle under the on-page title, default the LLM-judges
toggle off, relabel and resize the two primary run buttons (Generate Song 3/4, N
Algorithm Suite 1/4), strip the explanatory note under the Clear button, and relocate
the Genre card from the header to the very bottom of the page as a full-width card
renamed "Genre Knob." Done when every listed string/layout change is live in the
frontend (and the two backend/README strings are renamed), `npm run lint` is clean, and
a manual pass in the browser confirms the new button ratio, judges-off default, and
Genre card's new position/width all render correctly.

## Context

Read `frontend/app/layout.tsx`, `frontend/app/page.tsx`, `frontend/app/globals.css`,
`backend/main.py`, and `README.md` before planning. Key findings:

- **Identity strings** live in three places: `frontend/app/layout.tsx:22`
  (`metadata.title = "Image to Song"`), `backend/main.py:24`
  (`FastAPI(title="Image to Song")`, surfaced at `/docs`), and `README.md:1`
  (`# Image to Song`). The on-page `<h1>` is separate: `frontend/app/page.tsx:1189`
  (`<h1 className="title">Atlantis</h1>`).
- **Genre card** is currently `frontend/app/page.tsx:1191-1245`, nested inside
  `<header className="hero">` next to the title. Its CSS (`globals.css:396-404`)
  fixes it at `width: 17rem; flex-shrink: 0` — a small sidebar card, not a full-width
  one like `.info-card`/`.upload-card` (which have no explicit width and stretch to
  fill `.page`'s flex column, `globals.css:68-76`).
- **LLM judges default**: `frontend/app/page.tsx:911` —
  `const [judgesEnabled, setJudgesEnabled] = useState(true)`.
- **Run buttons**: `frontend/app/page.tsx:1308-1333`, two `<button className="btn-lg">`
  inside `.run-buttons` (`globals.css:165-175`), which currently gives both buttons
  `flex: 1` (equal width) via `.run-buttons .btn-lg`. First button calls
  `startRun(["siglip"])` and is labeled "Use SigLIP"; it only runs SigLIP tagging, not
  a full Suno generation (confirmed by reading `startRun`, `page.tsx:1090-1111` — the
  actual Suno call is a separate per-row "Generate song" button reached later, via
  `handleGenerate`). Second button is `{algorithms.length} Algorithm Suite`
  (`startRun(algorithms.map(a => a.id))`) — already dynamic, not a literal "4" (fixed
  in a prior session so it can't drift if the algorithm count changes again).
- **Clear button / danger zone**: `frontend/app/page.tsx:1379-1391`. The button and its
  explanatory `<p className="note">` live in `.danger-zone` (`globals.css:727-731`),
  currently the last element on the page.
- **Hero layout**: `.hero` (`globals.css:78-86`) is `display: flex; justify-content:
  space-between` — built specifically to hold the title on the left and the
  17rem genre card on the right. Once the genre card moves out, `.hero` will contain
  only the title (and new subtitle), so `justify-content: space-between` no longer has
  a second flex item to space against — needs a look after the move (Step 3/9).

## Decisions

- **Rename scope**: rename all three identity strings (frontend tab title, backend
  FastAPI title, README heading) to "Ideagream Studio", not just the frontend tab.
- **"Generate Song" label**: label-only change. The button keeps calling
  `startRun(["siglip"])` (SigLIP tagging only) — no behavior change, even though the
  new label doesn't literally match what one click does.
- **Genre card placement**: moves to the literal bottom of the page — after the
  danger-zone (Clear button) and after the `runs` section when a run is active — exactly
  as described, despite genre needing to be picked before a run starts to affect the
  prompt.
- **Scope boundary**: only the changes explicitly listed in the request. No incidental
  cleanup, no rewriting CLAUDE.md's or README's descriptive prose beyond the literal
  string/heading renames, no touching `eval/`, `musicgen-experiment/`, or anything not
  named.
- **Delegation**: main thread only. No subagents invoked for this plan.
- **Plan file path**: `plans/ideagream-rebrand.md`.

## Steps

### 1. Rename the app's identity strings
Files: `frontend/app/layout.tsx`, `backend/main.py`, `README.md`
Execution: [main]
Done when: `layout.tsx`'s `metadata.title` is `"Ideagream Studio"`,
`main.py`'s `FastAPI(title=...)` is `"Ideagream Studio"`, and `README.md`'s top `#`
heading reads `# Ideagream Studio`. (`layout.tsx`'s `description` field and
`README.md`'s body prose are untouched — out of scope.)

### 2. Rewrite the on-page hero: title + new subtitle
Files: `frontend/app/page.tsx` (~line 1189), `frontend/app/globals.css`
Execution: [main]
Done when: `<h1 className="title">` reads "Ideagream Studio", and a new element
directly beneath it renders "A better way to compose mood-accurate songs from images
using joint embedding spaces" with a new CSS class (e.g. `.subtitle`) sized/colored
distinctly from both the h1 and the existing `.meta`/`.note` styles.

### 3. Default LLM judges to off
Files: `frontend/app/page.tsx` (line 911)
Execution: [main]
Done when: `useState(true)` for `judgesEnabled` is changed to `useState(false)`; the
checkbox renders unchecked on load.

### 4. Relabel and resize the run buttons
Files: `frontend/app/page.tsx` (~lines 1308-1322), `frontend/app/globals.css`
(~lines 165-175)
Execution: [main]
Done when: the first button's visible text is "Generate Song" (handler unchanged,
still `startRun(["siglip"])`); the two `.btn-lg` buttons in `.run-buttons` are
styled so "Generate Song" occupies 3/4 of the row's width and "{N} Algorithm Suite"
occupies 1/4 (e.g. distinct classes with `flex: 3` / `flex: 1` replacing the shared
`flex: 1` rule), at viewport widths where they sit side by side; the judges-toggle
checkbox still renders in the same row without visually breaking the ratio.

### 5. Remove the Clear button's explanatory note
Files: `frontend/app/page.tsx` (~lines 1379-1391)
Execution: [main]
Done when: the `<p className="note">Deletes the current image...</p>` block under the
Clear button is deleted; the Clear button itself and its arm/confirm behavior are
unchanged. If `.note` becomes unused elsewhere in `danger-zone`, no CSS cleanup is
needed unless it's now fully dead in the file.

### 6. Relocate the Genre card to the bottom of the page and rename it
Files: `frontend/app/page.tsx` (move block from ~1191-1245 to after the danger-zone
close, ~line 1391), `frontend/app/globals.css` (~lines 396-404)
Execution: [main]
Done when: the entire `<div className="genre-card">...</div>` JSX block renders as the
last element on the page (after `.danger-zone`, after the conditional `.runs` section);
its `<h2 className="section-title">` reads "Genre Knob" (dropping the bare "Genre");
the `<span className="meta-label">` next to it reads "Optional" (dropping "just for
fun"); the card's width is no longer fixed at `17rem` — `.genre-card`'s
`width: 17rem; flex-shrink: 0` are removed (or overridden) so it stretches to the same
width as `.info-card`/`.upload-card` above it.

### 7. Clean up the now-genre-card-less hero layout
Files: `frontend/app/globals.css` (`.hero`, ~lines 78-86)
Execution: [main]
Done when: `.hero` no longer has unused `justify-content: space-between`/flex-wrap
rules that made sense only when a second flex item (the genre card) sat beside the
title — replaced with whatever simpler block/flex-column layout suits a title +
subtitle pair, visually confirmed in Step 8.

### 8. Lint and manual verification
Files: none (verification only)
Execution: [main]
Done when: `npm run lint` (from `frontend/`) is clean (no new errors — the pre-existing
`<img>` warning is fine), both dev servers are started, and a manual pass in the
browser confirms: page title/tab reads "Ideagream Studio"; subtitle renders under the
h1; LLM judges checkbox is unchecked by default; "Generate Song" and "{N} Algorithm
Suite" buttons render at roughly a 3:1 width ratio; Clear button has no note text
beneath it; Genre Knob card renders full-width at the bottom of the page, after the
Clear button (and after any active run's rows), with "Optional" instead of "(optional,
just for fun)"; genre selection still works (spin/search/roll/clear) and still affects
the prompt sent on the next run.

## Out of scope

- Rewriting CLAUDE.md's or README's descriptive prose (architecture notes, pitch
  paragraph) beyond the literal heading rename in README.md.
- Changing "Generate Song"'s actual behavior to trigger a real Suno generation —
  label-only per the interview decision.
- `eval/`, `musicgen-experiment/`, or any file not touched by the listed changes.
- Any backend logic change — this is a frontend copy/layout pass plus two one-line
  backend/README string renames.

## Risks

- **Genre-card-at-the-bottom is a real workflow inversion**: genre must be picked
  before a run starts to affect the generated prompt, but it will now render below the
  buttons that start a run (and below the entire runs/Clear section). Confirmed
  intentional in the interview, but worth a final look once it's live — it may read as
  broken/misplaced to a first-time user even though it functions correctly.
- **"Generate Song" mislabels current behavior**: the button will say "Generate Song"
  but only performs SigLIP tagging, not an actual Suno song render. Accepted as a
  literal, label-only change per the interview, but it's a real UX honesty gap that may
  need a follow-up decision later (e.g. two-step copy, or wiring real generation).
- **Hero CSS after the genre card leaves**: `.hero`'s `justify-content: space-between`
  was built for two flex children; removing one without adjusting the rule could leave
  odd empty space or misalignment that's only obvious once rendered — Step 7's fix is
  a judgment call to be confirmed visually in Step 8, not something fully specifiable
  in the plan text alone.
