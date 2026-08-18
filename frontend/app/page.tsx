"use client";

import { useEffect, useMemo, useState } from "react";

import SpinKnob from "./SpinKnob";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Track = {
  title: string;
  duration: number;
  audioUrl: string;
  audioFile: string;
};

type FacetTag = {
  tag: string;
  facet: string;
  score: number;
};

type JudgeVerdict = {
  tag: string;
  flagged: boolean;
  reason: string;
};

// A cross-facet mood contradiction the second judge found (e.g. "joyful" from
// valence next to "haunting" from mood) -- `drop` names which of the pair got
// dropped from the tag list entirely (no replacement is picked).
type Contradiction = {
  tag_a: string;
  tag_b: string;
  drop: string;
  reason: string;
};

// What GET /algorithms returns: the embedding algorithms the picker offers, each
// with a display label and a one-line blurb (surfaced in the algorithm-selector
// card and used to keep score-meaning copy algorithm-aware).
type AlgorithmInfo = {
  id: string;
  label: string;
  blurb: string;
};

type EmbeddingPoint = {
  tag: string;
  facet: string;
  x: number;
  y: number;
};

// A 2D PCA projection of the top-20 candidates' vectors in the chosen
// algorithm's embedding space, with the image vector projected onto that
// same basis -- illustrative only, see `embedding_projection` in
// backend/embedding_tagger.py for the caveats.
type Embedding = {
  model: string;
  source_dimensions: number;
  image: { x: number; y: number };
  points: EmbeddingPoint[];
};

// How the final tags were arrived at: the full top-20 candidates, the first
// judge's per-tag verdicts, the second judge's cross-facet contradiction
// pairs, and the final tags -- each labeled with its facet. A tag either
// judge flags is dropped outright, not replaced, so `final` can be shorter
// than `top20`'s selected subset.
type Tagging = {
  top20: FacetTag[];
  judge: JudgeVerdict[];
  contradictions: Contradiction[];
  final: FacetTag[];
  // Whether this run actually invoked the Claude judges (the "LLM judges"
  // toggle's value at request time) -- an empty `judge`/`contradictions`
  // list looks the same whether the judges ran and flagged nothing, or
  // never ran at all, so this is what tells those two cases apart.
  judges_enabled: boolean;
  embedding?: Embedding;
  // CoCa-only: a free-text caption from its captioning head. Display-only --
  // never fed into tag selection or the Suno prompt.
  caption?: string;
  // The tagged image itself, as a data: URI -- sent back by the backend
  // (which already had the bytes in memory either way) so the downloadable
  // tag-analysis report can embed it directly, without the frontend fetching
  // it a second time. See main.py's `/tag` docstring for why that second
  // fetch is deliberately avoided for a sample-picker image.
  image_data_url: string;
};

// Fixed hue order for the 8 facets, never reassigned/cycled per-render --
// matches backend/vocabulary.py's ADJECTIVES order. Dark-surface steps, since
// the embedding plot sits on a dark panel; identity is confirmed on hover
// (the tooltip names the exact tag and facet), so color here is a grouping
// cue rather than the only way to tell two points apart.
const FACET_ORDER = [
  "color",
  "light",
  "mood",
  "valence",
  "arousal",
  "texture",
  "energy",
  "style",
];
const FACET_COLORS: Record<string, string> = {
  color: "#3987e5",
  light: "#d95926",
  mood: "#199e70",
  valence: "#c98500",
  arousal: "#d55181",
  texture: "#008300",
  energy: "#9085e9",
  style: "#e66767",
};

// What `POST /tag` returns: the tagging half of the pipeline only, no Suno
// call yet. Held in state until the user commits with "Generate song", at
// which point it's sent straight back to `POST /generate` so the judge
// never has to run a second time for the same image.
type Analysis = {
  tags: string[];
  tag_scores: Record<string, number>;
  tagging: Tagging;
  prompt: string;
  algorithm: string;
};

// What `POST /generate` returns: Suno renders two takes per job and both are
// kept -- they cost nothing extra, and one is usually the better listen.
type GeneratedSong = {
  job_id: string;
  tracks: Track[];
};

// One algorithm's slot within the current run. SigLIP mode has exactly one
// of these; Suite mode has one per algorithm, tracked independently so a
// single algorithm failing (a model that errors, a judge call that times
// out) doesn't blank the other rows.
type RunResult = {
  status: "loading" | "ready" | "error";
  analysis?: Analysis;
  error?: string;
  generating?: boolean;
  generated?: GeneratedSong;
  generateError?: string;
};

// Downloaded MP3s are served by the backend out of its (non-history) audio
// folder. Filenames contain spaces, `&` and apostrophes, so each path
// segment is escaped separately -- encoding the whole path would eat the
// `/` too.
const fileUrl = (path: string) =>
  `${API_URL}/files/${path.split("/").map(encodeURIComponent).join("/")}`;

// Shared by every mutating call (wipe run, extract tags, generate song):
// fetch, parse JSON, and normalize both an API error (the response body's
// `detail`) and a network failure into one `Error` with a message fit to
// show the user -- so each call site's catch block is just `setError`.
async function apiRequest(
  url: string,
  options: RequestInit,
  fallbackMessage: string,
) {
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail ?? fallbackMessage);
    }
    return data;
  } catch (err) {
    throw new Error(
      err instanceof Error ? err.message : "Could not reach the server.",
    );
  }
}

// Shared by every local-save action (a downloaded track, the tag-analysis
// report): build a temporary object URL for the blob and click a hidden
// link, since a blob's URL always belongs to this page regardless of where
// the bytes came from.
function downloadBlob(blob: Blob, filename: string) {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

// The frontend (localhost:3000) and backend (127.0.0.1:8000) are different
// origins, so a plain `<a download href={fileUrl(...)}>` won't reliably
// force a save -- browsers ignore the `download` attribute cross-origin and
// just navigate instead. Fetching the audio as a blob first and saving
// *that* via downloadBlob works regardless of origin, since the blob URL
// belongs to this page once created.
async function downloadTrack(track: Track) {
  const response = await fetch(fileUrl(track.audioFile));
  if (!response.ok) throw new Error("Could not download the track.");
  const blob = await response.blob();
  downloadBlob(blob, track.audioFile.split("/").pop() ?? "track.mp3");
}

// Scores sit in a narrow band near zero (SigLIP's probabilities, and the other
// algorithms' raw cosine similarities alike), so a flat 3 decimals reads as
// "0.000" for the smallest ones -- indistinguishable from an actual zero. Below
// 0.001 this switches to scientific notation instead of adding more decimals, so
// a real value is always visible no matter how small.
function formatScore(score: number): string {
  if (!Number.isFinite(score) || score === 0) return "0.000";
  if (Math.abs(score) < 0.001) return score.toExponential(2);
  return score.toFixed(3);
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Which of a judge pass's verdicts actually get shown as a dropped tag, and
// what to say about each -- the data-shaping both JudgeReviewSection (JSX,
// in TagBreakdown below) and judgeListHtml (the HTML report, in
// algorithmSectionHtml below) render. Kept in one place so a future change
// to what counts as "flagged" or how a contradiction's reason reads can't
// update one presentation and silently miss the other.
type JudgeRow = { key: string; flaggedTag: string; reason: string };

function flaggedJudgeRows(judge: JudgeVerdict[]): JudgeRow[] {
  return judge
    .filter((v) => v.flagged)
    .map((v) => ({ key: v.tag, flaggedTag: v.tag, reason: v.reason }));
}

function contradictionRows(contradictions: Contradiction[]): JudgeRow[] {
  return contradictions.map((c, index) => {
    const kept = c.drop === c.tag_a ? c.tag_b : c.tag_a;
    return {
      key: `${c.tag_a}-${c.tag_b}-${index}`,
      flaggedTag: c.drop,
      reason: `contradicted “${kept}” -- ${c.reason}`,
    };
  });
}

function tagTableRows(tags: FacetTag[], selected?: Set<string>): string {
  return tags
    .map(
      ({ tag, facet, score }) => `
        <tr${selected && !selected.has(tag) ? ' class="muted"' : ""}>
          <td>${escapeHtml(tag)}</td>
          <td>${escapeHtml(facet)}</td>
          <td class="mono">${formatScore(score)}</td>
        </tr>`,
    )
    .join("");
}

function judgeListHtml(
  rows: { flaggedTag: string; reason: string }[],
  emptyMessage: string,
): string {
  if (rows.length === 0) return `<p class="empty">${escapeHtml(emptyMessage)}</p>`;
  return `<ul class="judge-list">${rows
    .map(
      (row) => `
        <li><span class="flagged">${escapeHtml(row.flaggedTag)}</span> dropped
          ${row.reason ? `<div class="reason">${escapeHtml(row.reason)}</div>` : ""}
        </li>`,
    )
    .join("")}</ul>`;
}

// One algorithm's full tag-analysis section: final tags, CoCa's caption (if
// any), both judge passes' verdicts, the top-20 candidate pool, and the
// prompt actually sent to Suno -- the same fields TagBreakdown renders in
// the UI, just as static HTML instead of React.
function algorithmSectionHtml(algorithmLabel: string, analysis: Analysis): string {
  const { tagging, prompt } = analysis;
  const finalTagNames = new Set(tagging.final.map((f) => f.tag));

  return `
    <section class="algo">
      <h2>${escapeHtml(algorithmLabel)}</h2>

      ${
        tagging.final.length > 0
          ? `<table class="tags"><thead><tr><th>tag</th><th>facet</th><th>score</th></tr></thead>
             <tbody>${tagTableRows(tagging.final)}</tbody></table>`
          : `<p class="empty">No tags survived judging.</p>`
      }

      ${
        tagging.caption
          ? `<p class="caption"><span class="label">caption</span> &ldquo;${escapeHtml(tagging.caption)}&rdquo;</p>`
          : ""
      }

      <h3>Judge review 1 &mdash; poor-quality tags</h3>
      ${judgeListHtml(
        flaggedJudgeRows(tagging.judge),
        tagging.judges_enabled ? "Judge #1 found no tags that didn't fit the image's mood this run." : "LLM judges skipped.",
      )}

      <h3>Judge review 2 &mdash; contradictions</h3>
      ${judgeListHtml(
        contradictionRows(tagging.contradictions),
        tagging.judges_enabled ? "Judge #2 identified no contradictory tags this run." : "LLM judges skipped.",
      )}

      ${
        tagging.top20.length > 0
          ? `<h3>Top 20 candidates</h3>
             <table class="tags"><thead><tr><th>tag</th><th>facet</th><th>score</th></tr></thead>
             <tbody>${tagTableRows(tagging.top20, finalTagNames)}</tbody></table>`
          : ""
      }

      ${prompt ? `<p class="prompt"><span class="label">prompt sent to Suno</span> <span class="mono">${escapeHtml(prompt)}</span></p>` : ""}
    </section>`;
}

// Builds a single, self-contained HTML document of the current run's full
// tag analysis across every algorithm -- everything TagBreakdown shows in
// the UI, minus the embedding plot (a live canvas, not something worth
// statically rendering here) and any generated audio (out of scope for a
// *tag* analysis report). The image comes from an algorithm's own
// `tagging.image_data_url` (a data: URI baked in by the backend -- see the
// `Tagging` type above) rather than the frontend's own `previewUrl`, so
// this file is genuinely self-contained: it opens correctly later even
// with the backend closed or an uploaded file's blob: URL long gone.
function buildReportHtml({
  results,
  runAlgorithms,
  algorithmLabels,
  genre,
}: {
  results: Record<string, RunResult>;
  runAlgorithms: string[];
  algorithmLabels: Record<string, string>;
  genre: string;
}): string {
  const imageSrc = runAlgorithms
    .map((id) => results[id]?.analysis?.tagging.image_data_url)
    .find((url): url is string => Boolean(url));

  const sections = runAlgorithms
    .map((id) => {
      const result = results[id];
      const label = algorithmLabels[id] ?? id;
      if (result?.status === "ready" && result.analysis) {
        return algorithmSectionHtml(label, result.analysis);
      }
      if (result?.status === "error") {
        return `<section class="algo"><h2>${escapeHtml(label)}</h2><p class="empty">Tagging failed: ${escapeHtml(result.error ?? "unknown error")}</p></section>`;
      }
      return "";
    })
    .join("");

  const generated = new Date().toLocaleString();

  return `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Atlantis tag analysis report</title>
<style>
  @page { margin: 1.5cm; }
  body { font-family: Georgia, serif; color: #201d19; background: #faf8f3; }
  h1 { font-size: 1.6rem; }
  h2 { font-size: 1.15rem; border-top: 2px solid #201d19; padding-top: 0.75rem; margin-top: 2rem;
       break-before: auto; break-after: avoid; }
  h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: #5a544c;
       margin-top: 1.25rem; margin-bottom: 0.4rem; break-after: avoid; }
  .meta { color: #5a544c; font-size: 0.85rem; }
  .cover-image { max-width: 100%; max-height: 24rem; display: block; margin: 1rem 0;
                 border: 1px solid #ddd4c4; }
  table.tags { border-collapse: collapse; width: 100%; font-size: 0.85rem; break-inside: avoid; }
  table.tags th, table.tags td { text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid #ddd4c4; }
  table.tags tr.muted { color: #8c8579; }
  .label { font-weight: 700; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.06em;
           color: #8c8579; margin-right: 0.4rem; }
  .mono { font-family: ui-monospace, Menlo, monospace; }
  .empty { color: #8c8579; font-size: 0.85rem; font-style: italic; }
  .judge-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.5rem; }
  .judge-list li { background: #f2e3dd; padding: 0.5rem 0.75rem; border-left: 3px solid #7a3226; font-size: 0.85rem;
                   break-inside: avoid; }
  .flagged { font-weight: 700; }
  .reason { color: #5a544c; margin-top: 0.2rem; }
  .caption, .prompt { font-size: 0.85rem; margin-top: 0.75rem; }
</style>
</head>
<body>
  <h1>Atlantis &mdash; tag analysis report</h1>
  <p class="meta">Generated ${escapeHtml(generated)}${genre ? ` &middot; genre: ${escapeHtml(genre)}` : ""}</p>
  ${imageSrc ? `<img class="cover-image" src="${escapeHtml(imageSrc)}" alt="">` : ""}
  ${sections}
</body>
</html>`;
}

// downloadBlob, not window.print()/a native dialog -- an earlier version of
// this feature drove the browser's print-to-PDF dialog instead, which
// crashed inside a Cursor-embedded webview. A downloaded .html file the
// user opens in their own browser sidesteps that entirely, and can still be
// printed-to-PDF from there if wanted.
function downloadReportHtml(html: string) {
  const blob = new Blob([html], { type: "text/html" });
  downloadBlob(blob, `atlantis-tag-report-${new Date().toISOString().slice(0, 10)}.html`);
}

const PLOT_WIDTH = 520;
const PLOT_HEIGHT = 300;
const PLOT_MARGIN = 30;

function clamp(value: number, low: number, high: number) {
  return Math.min(Math.max(value, low), high);
}

// Rough monospace-ish width estimate so the tooltip box fits its text without
// measuring the DOM -- fine for a small "tag · facet" string.
function estimateTextWidth(text: string) {
  return text.length * 5.4 + 16;
}

// A 2D scatter of the top-20 tag candidates, color-coded by facet, with the
// image's projected position marked separately, on its own dark panel.
// SigLIP/CLIP-family models' image/text "modality gap" means the image often
// lands well outside the tag cluster (see embedding_projection's docstring in
// backend/embedding_tagger.py) -- rather than stretch the shared scale to fit
// it and crush all 20 tags into a corner, the image marker is clamped to the
// plot's edge and flagged as approximate. Points aren't directly labeled (they
// overlap too easily when several near-synonyms land close together) --
// hover or focus a point for its exact tag and facet instead.
function EmbeddingPlot({
  embedding,
  finalTags,
  algorithmLabel,
}: {
  embedding: Embedding;
  finalTags: Set<string>;
  algorithmLabel: string;
}) {
  const { points } = embedding;
  const [hovered, setHovered] = useState<string | null>(null);

  // Only `hovered` changes on every interaction; the scatter's geometry
  // (bounds, coordinate mapping, which facets are present) only depends on
  // `embedding` itself, so it's wasted work to redo the array scans and
  // rebuild both mapping functions on every hover. `embedding` is a stable
  // reference across those re-renders (it lives in the parent's state, not
  // here), so this only recomputes when the tag analysis actually changes.
  const geometry = useMemo(() => {
    const { points, image } = embedding;
    if (points.length === 0) return null;

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const spanX = Math.max(...xs) - Math.min(...xs) || 1;
    const spanY = Math.max(...ys) - Math.min(...ys) || 1;
    const loX = Math.min(...xs) - spanX * 0.15;
    const hiX = Math.max(...xs) + spanX * 0.15;
    const loY = Math.min(...ys) - spanY * 0.15;
    const hiY = Math.max(...ys) + spanY * 0.15;

    const toSvgX = (x: number) =>
      PLOT_MARGIN + ((x - loX) / (hiX - loX)) * (PLOT_WIDTH - 2 * PLOT_MARGIN);
    // Flipped: data "up" should render toward the SVG top, where y is smaller.
    const toSvgY = (y: number) =>
      PLOT_HEIGHT -
      PLOT_MARGIN -
      ((y - loY) / (hiY - loY)) * (PLOT_HEIGHT - 2 * PLOT_MARGIN);

    const imageOffscreen =
      image.x < loX || image.x > hiX || image.y < loY || image.y > hiY;

    return {
      toSvgX,
      toSvgY,
      imageOffscreen,
      imageX: clamp(image.x, loX, hiX),
      imageY: clamp(image.y, loY, hiY),
      facets: FACET_ORDER.filter((facet) =>
        points.some((p) => p.facet === facet),
      ),
    };
  }, [embedding]);

  if (!geometry) return null;
  const { toSvgX, toSvgY, imageOffscreen, imageX, imageY, facets } = geometry;

  const hoveredPoint =
    hovered === "image" ? null : points.find((p) => p.tag === hovered);
  const tooltip = hoveredPoint
    ? {
        cx: toSvgX(hoveredPoint.x),
        cy: toSvgY(hoveredPoint.y),
        text: `${hoveredPoint.tag} · ${hoveredPoint.facet}`,
      }
    : hovered === "image"
      ? { cx: toSvgX(imageX), cy: toSvgY(imageY), text: "image" }
      : null;
  const tooltipWidth = tooltip ? estimateTextWidth(tooltip.text) : 0;
  // Keep the box on-canvas: flip to the left of the point once it would
  // overflow the right edge, and clamp vertically within the plot margins.
  const tooltipX = tooltip
    ? tooltip.cx + 10 + tooltipWidth > PLOT_WIDTH
      ? tooltip.cx - 10 - tooltipWidth
      : tooltip.cx + 10
    : 0;
  const tooltipY = tooltip ? clamp(tooltip.cy - 10, 12, PLOT_HEIGHT - 20) : 0;

  return (
    <div className="embedding-plot">
      <svg
        viewBox={`0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`}
        role="img"
        aria-label={`2D projection of the top 20 tag candidates and the image in ${algorithmLabel}'s embedding space. Hover a point for its tag and facet.`}
      >
        {points.map((p) => {
          const isFinal = finalTags.has(p.tag);
          return (
            <circle
              key={p.tag}
              cx={toSvgX(p.x)}
              cy={toSvgY(p.y)}
              r={isFinal ? 6 : 4}
              fill={FACET_COLORS[p.facet]}
              opacity={isFinal ? 1 : 0.4}
              stroke="var(--paper)"
              strokeWidth={1}
              tabIndex={0}
              onMouseEnter={() => setHovered(p.tag)}
              onMouseLeave={() => setHovered(null)}
              onFocus={() => setHovered(p.tag)}
              onBlur={() => setHovered(null)}
            />
          );
        })}

        <circle
          cx={toSvgX(imageX)}
          cy={toSvgY(imageY)}
          r={7}
          fill="var(--paper-dim)"
          stroke="var(--paper)"
          strokeWidth={2}
          tabIndex={0}
          onMouseEnter={() => setHovered("image")}
          onMouseLeave={() => setHovered(null)}
          onFocus={() => setHovered("image")}
          onBlur={() => setHovered(null)}
        />
        {imageOffscreen && (
          <text
            x={toSvgX(imageX)}
            y={toSvgY(imageY) - 12}
            className="embedding-image-flag"
            textAnchor="middle"
          >
            image →
          </text>
        )}

        {tooltip && (
          <g pointerEvents="none">
            <rect
              x={tooltipX}
              y={tooltipY}
              width={tooltipWidth}
              height={20}
              rx={4}
              className="embedding-tooltip-box"
            />
            <text
              x={tooltipX + tooltipWidth / 2}
              y={tooltipY + 14}
              textAnchor="middle"
              className="embedding-tooltip-text"
            >
              {tooltip.text}
            </text>
          </g>
        )}
      </svg>

      <div className="embedding-legend">
        {facets.map((facet) => (
          <span key={facet} className="embedding-legend-item">
            <span
              className="embedding-swatch"
              style={{ background: FACET_COLORS[facet] }}
            />
            {facet}
          </span>
        ))}
      </div>

      <p className="note">
        {algorithmLabel} embeds the image and each candidate word into a{" "}
        {embedding.source_dimensions}-dimensional vector; PCA projects those
        vectors onto the two directions of greatest variance so they can be
        plotted here in 2D. The selected tags are bright, the other
        candidates dimmed -- hover or tab to a point for its tag and facet.
        {imageOffscreen &&
          " The image's true position sits well outside this tag cluster -- pinned here to the edge nearest it."}
      </p>
      <p className="note">
        Circle size/brightness marks selection, not rank -- for a word&rsquo;s
        actual match strength, read its score above. Distance between two{" "}
        <em>tag</em> points is the meaningful reading here: tags sitting close
        together really are similar to each other in {algorithmLabel}&rsquo;s
        full space. Distance from a tag to the image marker isn&rsquo;t
        reliable the same way -- SigLIP/CLIP-family models&rsquo; image and
        text embeddings occupy separate regions of the space almost
        regardless of how well a word actually fits (a &ldquo;modality
        gap&rdquo;), so the image&rsquo;s position here says little about any
        one tag&rsquo;s fit.
      </p>
    </div>
  );
}

// One tag pill -- used both for the final tag row (bright) and the top-20
// candidates row (muted, with the selected ones highlighted), which used to
// hand-build the same tag-group/tag-facet/tag/tag-score markup separately.
function TagChip({
  tag,
  facet,
  score,
  muted,
  selected,
}: {
  tag: string;
  facet: string;
  score: number;
  muted?: boolean;
  selected?: boolean;
}) {
  return (
    <span className={`tag-group${selected ? " is-selected" : ""}`}>
      <span className="tag-facet">{facet}</span>
      <span className={`tag${muted ? " tag-muted" : ""}`}>
        {tag}
        <span className="tag-score">{formatScore(score)}</span>
      </span>
    </span>
  );
}

// One judge's findings -- the mood-fit judge and the contradiction judge
// used to each hand-write the same heading/list/empty-state JSX shape with
// only their row data differing; this is that shape, parameterized once.
function JudgeReviewSection({
  title,
  emptyMessage,
  rows,
}: {
  title: string;
  emptyMessage: string;
  rows: { key: string; flaggedTag: string; reason: string }[];
}) {
  return (
    <div className="judge-review">
      <span className="meta-label">{title}</span>
      {rows.length > 0 ? (
        <ul className="judge-list">
          {rows.map((row) => (
            <li key={row.key} className="judge-item">
              <span className="judge-flagged">{row.flaggedTag}</span>
              <span className="judge-dropped">dropped</span>
              {row.reason && <div className="judge-reason">{row.reason}</div>}
            </li>
          ))}
        </ul>
      ) : (
        <p className="judge-empty">{emptyMessage}</p>
      )}
    </div>
  );
}

// The tag row, judge review, top-20 candidates and prompt line -- shared by
// every algorithm row in the current run, so SigLIP mode and Suite mode
// never drift apart in how they present a tag set. Its one call site
// (RunRow) only renders this once tagging has actually succeeded, so every
// prop here is guaranteed to exist by then -- no defaulting needed.
function TagBreakdown({
  tagging,
  prompt,
  algorithm,
  algorithmLabel,
}: {
  tagging: Tagging;
  prompt: string;
  algorithm: string;
  algorithmLabel: string;
}) {
  // Only SigLIP has a trained calibration head turning cosine similarity into a
  // 0-1 probability (see backend/embedding_tagger.py's module docstring) -- every
  // other algorithm here scores with raw cosine similarity on its own scale, so
  // the score-meaning copy has to say something different for it.
  const calibrated = algorithm === "siglip";

  return (
    <>
      {tagging.final.length > 0 && (
        <>
          <div className="tags">
            {tagging.final.map(({ tag, facet, score }) => (
              <TagChip key={tag} tag={tag} facet={facet} score={score} />
            ))}
          </div>
          <p className="note">
            {calibrated ? (
              <>
                The number beside each tag is {algorithmLabel}&rsquo;s match
                probability for that word against this image (0–1) -- not a
                rank. Scores across the whole vocabulary sit close to zero, so
                small differences still mean one word fit noticeably better
                than another.
              </>
            ) : (
              <>
                The number beside each tag is raw cosine similarity between
                the image and that word in {algorithmLabel}&rsquo;s own
                embedding space -- not a rank, and not on the same 0–1 scale
                as SigLIP&rsquo;s calibrated probability, so it isn&rsquo;t
                meaningful to compare these numbers across algorithms.
              </>
            )}
          </p>
        </>
      )}

      {tagging.caption && (
        <div className="caption-block">
          <span className="meta-label">caption</span>
          <p className="caption-text">&ldquo;{tagging.caption}&rdquo;</p>
          <p className="note">
            {algorithmLabel}&rsquo;s captioning head describing the image in
            free text -- shown for comparison only, never sent to Suno or
            used to pick tags.
          </p>
        </div>
      )}

      <JudgeReviewSection
        title="judge review 1 -- poor-quality tags"
        emptyMessage={
          tagging.judges_enabled
            ? "Judge #1 found no tags that didn't fit the image's mood this run."
            : "LLM judges skipped."
        }
        rows={flaggedJudgeRows(tagging.judge)}
      />

      <JudgeReviewSection
        title="judge review 2 -- contradictions"
        emptyMessage={
          tagging.judges_enabled
            ? "Judge #2 identified no contradictory tags this run."
            : "LLM judges skipped."
        }
        rows={contradictionRows(tagging.contradictions)}
      />

      {tagging.top20.length > 0 && (
        <div className="top20">
          <span className="meta-label">top 20 candidates</span>
          <div className="tags tags-muted">
            {tagging.top20.map(({ tag, facet, score }) => (
              <TagChip
                key={tag}
                tag={tag}
                facet={facet}
                score={score}
                muted
                selected={tagging.final.some((f) => f.tag === tag)}
              />
            ))}
          </div>
        </div>
      )}

      {prompt && (
        <div className="meta prompt-line">
          <span className="meta-label">prompt sent to Suno: </span>
          <span className="mono">{prompt}</span>
        </div>
      )}
    </>
  );
}

// One row of the run: the image on the left, that algorithm's tag breakdown
// on the right, and (once tagging succeeded) its own independent Generate
// song action and result. Used for both SigLIP mode (one row) and Suite
// mode (five rows, back to back, so tag sets are easy to compare).
function RunRow({
  algorithmLabel,
  previewUrl,
  result,
  onGenerate,
}: {
  algorithmLabel: string;
  previewUrl: string | null;
  result: RunResult;
  onGenerate: () => void;
}) {
  const embedding = result.analysis?.tagging.embedding;

  return (
    <div className="run-row">
      <div className="run-row-media">
        {previewUrl && (
          // eslint-disable-next-line @next/next/no-img-element -- local blob: URL, not a remote image
          <img src={previewUrl} alt="" className="run-row-image" />
        )}

        {/* Fills the space below the image rather than leaving it blank --
            same data the old inline embedding section showed, just relocated
            so the image column isn't dead space once tagging finishes. */}
        {embedding && embedding.points.length > 0 && result.analysis && (
          <div className="embedding-section">
            <span className="meta-label">embedding space (illustrative)</span>
            <EmbeddingPlot
              embedding={embedding}
              finalTags={new Set(result.analysis.tagging.final.map((f) => f.tag))}
              algorithmLabel={algorithmLabel}
            />
          </div>
        )}
      </div>

      <div className="run-row-body">
        <div className="run-row-head">
          <span className="run-name">{algorithmLabel}: Tag Analysis</span>
        </div>

        {result.status === "loading" && (
          <p className="status-notice">Tagging with {algorithmLabel}…</p>
        )}

        {result.status === "error" && (
          <p className="status-error">{result.error}</p>
        )}

        {result.status === "ready" && result.analysis && (
          <>
            <TagBreakdown
              tagging={result.analysis.tagging}
              prompt={result.analysis.prompt}
              algorithm={result.analysis.algorithm}
              algorithmLabel={algorithmLabel}
            />

            {!result.generated && (
              <div className="actions">
                <button
                  onClick={onGenerate}
                  disabled={result.generating}
                  className="btn-primary"
                >
                  {result.generating ? "Composing…" : "Generate song"}
                </button>
              </div>
            )}

            {result.generating && (
              <p className="status-notice">
                This takes about two minutes. Keep this tab open.
              </p>
            )}

            {result.generateError && (
              <p className="status-error">{result.generateError}</p>
            )}

            {result.generated?.tracks.map((track, index) => (
              <div key={track.audioFile} className="take">
                <div className="take-title">
                  Take {index + 1}
                  <span className="take-name">
                    {" — "}
                    {track.title}
                  </span>
                </div>
                <audio controls src={fileUrl(track.audioFile)} />
                <div className="actions">
                  <button
                    type="button"
                    onClick={() => downloadTrack(track)}
                    className="btn-sm"
                  >
                    ⬇ Download
                  </button>
                </div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [genres, setGenres] = useState<string[]>([]);
  // The text in the search box, and the single source of truth for the genre.
  // A half-typed word simply isn't a genre yet, which is what `genre` derives.
  const [query, setQuery] = useState("");
  const genre = genres.includes(query) ? query : "";

  // Whether /tag runs its two Claude judge passes for the run about to
  // start. On by default; off skips both calls entirely on the backend
  // (see main.py's `judges` param), not just their result. Only meaningful
  // before a run starts -- the toggle disables once one is active, same as
  // the genre knob, since it can't retroactively affect tagging that
  // already happened.
  const [judgesEnabled, setJudgesEnabled] = useState(true);

  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const algorithmLabels = useMemo(
    () => Object.fromEntries(algorithms.map((a) => [a.id, a.label])),
    [algorithms],
  );

  // Filenames offered by the "select an image" grid, and which one (if any)
  // is currently picked -- `null` when the active image instead came from
  // the "upload your own" file input. Both write into the same `file` state
  // below, so picking one always clears the other.
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [sampleGridExpanded, setSampleGridExpanded] = useState(false);

  // The run in progress (or just completed): each algorithm's independent
  // tagging/generation state, keyed by algorithm id. Empty means no run is
  // active, which is also what gates the file input -- one run at a time.
  // `runAlgorithms` is just this object's keys (insertion order is
  // preserved for the non-numeric string keys used here), kept as its own
  // derived value rather than a second piece of state that could drift out
  // of sync with `results`.
  const [results, setResults] = useState<Record<string, RunResult>>({});
  const runAlgorithms = Object.keys(results);
  const runActive = runAlgorithms.length > 0;
  const busy =
    runActive &&
    Object.values(results).some((r) => r.status === "loading" || r.generating);

  // Suite-mode-only: algorithms that finished tagging but haven't had a song
  // generated (or started generating) yet -- what "Generate all" below would
  // actually kick off if clicked right now. Recomputed on every render
  // rather than tracked separately, so it stays correct as rows finish
  // tagging or generating one at a time.
  const pendingGenerateAlgorithms = runAlgorithms.filter((id) => {
    const result = results[id];
    return result?.status === "ready" && result.analysis && !result.generated && !result.generating;
  });

  // Deleting is irreversible and nothing is backed up, so the button has to be
  // clicked twice. This holds whether it is currently armed.
  const [armed, setArmed] = useState(false);
  // Set when the first loads can't reach the backend, so the page can say so
  // instead of looking like an app with every button broken.
  const [apiDown, setApiDown] = useState(false);

  useEffect(() => {
    const offline = (message: string) => () => {
      setError(message);
      setApiDown(true);
    };

    fetch(`${API_URL}/genres`)
      .then((response) => response.json())
      .then(setGenres)
      .catch(offline("Could not load the genre list."));

    fetch(`${API_URL}/algorithms`)
      .then((response) => response.json())
      .then(setAlgorithms)
      .catch(offline("Could not load the algorithm list."));

    fetch(`${API_URL}/test-images`)
      .then((response) => response.json())
      .then(setSampleImages)
      .catch(offline("Could not load the sample image list."));
  }, []);

  // A button that stays armed is its own trap, so it disarms itself shortly
  // after being primed.
  useEffect(() => {
    if (!armed) return;
    const timer = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(timer);
  }, [armed]);

  // Whether an image is picked at all -- either an uploaded `file` or a
  // `selectedSample` filename, mutually exclusive with each other. Gates
  // every run/clear button below, same as a plain `!file` check used to.
  const hasImage = Boolean(file || selectedSample);

  // A live preview of whatever image is picked. An uploaded file gets an
  // object URL (memoized rather than stored in state so creating it can't
  // itself trigger a render; the effect below only handles the revoke). A
  // sample image just points straight at the backend's static route --
  // plain <img> loads don't need CORS, unlike a `fetch()` of the same URL
  // would, so there's no bytes to fetch or object URL to manage here.
  const previewUrl = useMemo(() => {
    if (file) return URL.createObjectURL(file);
    if (selectedSample) return `${API_URL}/test-image-files/${selectedSample}`;
    return null;
  }, [file, selectedSample]);
  useEffect(() => {
    return () => {
      if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // Just records which sample was picked -- no bytes fetched client-side.
  // /tag re-reads them from disk server-side (see main.py's `sample_image`
  // param), so this never needs the browser to GET the image cross-origin.
  function selectSampleImage(name: string) {
    if (runActive || busy) return;
    setFile(null);
    setSelectedSample(name);
    setError(null);
  }

  async function handleClear() {
    if (!armed) {
      setArmed(true);
      return;
    }

    setArmed(false);
    setError(null);

    try {
      const data = await apiRequest(
        `${API_URL}/history`,
        { method: "DELETE" },
        "Could not clear the run.",
      );
      setFile(null);
      setSelectedSample(null);
      setResults({});
      setQuery("");
      setNotice(
        `Deleted ${data.files_removed} files, ${data.megabytes_freed} MB freed.`,
      );
    } catch (err) {
      // A failed wipe leaves the run in place rather than silently resetting
      // the UI while stale files remain on the backend.
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function rollRandomGenre() {
    if (genres.length > 0) {
      setQuery(genres[Math.floor(Math.random() * genres.length)]);
    }
  }

  // Returns rather than writes to `results` directly -- startRun below
  // waits for every algorithm's call to settle before committing any of
  // them, so the cards reveal together instead of one at a time.
  async function tagOneAlgorithm(algorithm: string): Promise<[string, RunResult]> {
    if (!hasImage) return [algorithm, { status: "error", error: "No image selected." }];

    const params = new URLSearchParams({ algorithm, judges: String(judgesEnabled) });
    if (genre) params.set("genre", genre);

    let body: FormData | undefined;
    if (file) {
      body = new FormData();
      body.append("image", file);
    } else if (selectedSample) {
      params.set("sample_image", selectedSample);
    }

    try {
      const data: Analysis = await apiRequest(
        `${API_URL}/tag?${params}`,
        { method: "POST", body },
        `Could not analyze the image with ${algorithm}.`,
      );
      return [algorithm, { status: "ready", analysis: data }];
    } catch (err) {
      return [
        algorithm,
        { status: "error", error: err instanceof Error ? err.message : String(err) },
      ];
    }
  }

  // FormData is what makes these file uploads. Note neither /tag nor
  // /generate ever sets a Content-Type header: the browser has to set that
  // itself so it can include the multipart boundary marker.
  function startRun(algorithmIds: string[]) {
    if (!hasImage || runActive) return;

    setError(null);
    setNotice(null);
    setResults(
      Object.fromEntries(algorithmIds.map((id) => [id, { status: "loading" as const }])),
    );

    // Parallel calls, one per algorithm, but all committed to `results`
    // together once every one has settled -- otherwise faster algorithms'
    // cards would pop from "loading" to ready individually while slower
    // ones are still spinning, which reads as flaky rather than "still
    // working." A bit slower overall (bounded by the slowest algorithm) in
    // exchange for every card updating at once. Still one independent
    // request per algorithm, so one failure doesn't block or blank the
    // others -- tagOneAlgorithm catches its own errors into a per-algorithm
    // result rather than rejecting, so Promise.all here always resolves.
    Promise.all(algorithmIds.map((id) => tagOneAlgorithm(id))).then((entries) => {
      setResults((previous) => ({ ...previous, ...Object.fromEntries(entries) }));
    });
  }

  async function handleGenerate(algorithm: string) {
    const result = results[algorithm];
    if (!hasImage || !result?.analysis) return;

    setResults((previous) => ({
      ...previous,
      [algorithm]: { ...previous[algorithm], generating: true, generateError: undefined },
    }));

    // Unlike /tag, /generate never re-tags the image -- it only needs the
    // original filename (for the audio naming convention), not the bytes,
    // so there's no reason to re-upload the whole file here. `tagging` still
    // carries `image_data_url` (added for the downloadable report), which
    // /generate would otherwise just echo back unread -- stripped here so a
    // multi-MB base64 image doesn't ride along on every generate call.
    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- destructured only to exclude it from the rest
    const { image_data_url, ...taggingForGenerate } = result.analysis.tagging;
    const body = new FormData();
    body.append("image_filename", file ? file.name : (selectedSample ?? ""));
    body.append("tags", JSON.stringify(result.analysis.tags));
    body.append("tag_scores", JSON.stringify(result.analysis.tag_scores));
    body.append("tagging", JSON.stringify(taggingForGenerate));
    body.append("prompt", result.analysis.prompt);

    const params = new URLSearchParams({ algorithm: result.analysis.algorithm });
    if (genre) params.set("genre", genre);

    try {
      const data: GeneratedSong = await apiRequest(
        `${API_URL}/generate?${params}`,
        { method: "POST", body },
        "Could not generate a song.",
      );
      setResults((previous) => ({
        ...previous,
        [algorithm]: { ...previous[algorithm], generating: false, generated: data },
      }));
    } catch (err) {
      setResults((previous) => ({
        ...previous,
        [algorithm]: {
          ...previous[algorithm],
          generating: false,
          generateError: err instanceof Error ? err.message : String(err),
        },
      }));
    }
  }

  // Suite mode only: fires handleGenerate for every algorithm that's ready
  // and hasn't been generated (or isn't already generating) yet -- same
  // independent-parallel-calls pattern as startRun, so one algorithm's
  // apiframe error doesn't hold up the rest. Costs real apiframe credits
  // (11 per song, so 5 algorithms is 55) -- there's no confirmation step
  // here, matching the per-row "Generate song" button, which also commits
  // credits on a single click.
  function handleGenerateAll() {
    pendingGenerateAlgorithms.forEach((algorithm) => {
      handleGenerate(algorithm);
    });
  }

  // Suite-mode-only, same gating spirit as "Generate all": available once
  // every algorithm has finished tagging (loading or not is what's checked,
  // not `busy`, since an in-progress song render shouldn't block a report
  // about the tags).
  const stillTagging = runAlgorithms.some((id) => results[id]?.status === "loading");

  function handleDownloadReport() {
    const html = buildReportHtml({ results, runAlgorithms, algorithmLabels, genre });
    downloadReportHtml(html);
  }

  return (
    <main className="page">
      <header className="hero">
        <h1 className="title">Atlantis</h1>

        <div className="genre-card">
          <h2 className="section-title">
            Genre <span className="meta-label">(optional, just for fun)</span>
          </h2>

          <section className="genre">
            <SpinKnob
              options={genres}
              value={genre}
              onChange={setQuery}
              label="Genre"
              disabled={busy || runActive}
            />

            <div className="genre-fields">
              <label htmlFor="genre-search" className="genre-label">
                Genre
              </label>

              <input
                id="genre-search"
                list="genre-options"
                value={query}
                disabled={busy || runActive}
                placeholder="spin, search or roll"
                onChange={(event) => setQuery(event.target.value)}
                className="genre-input"
              />
              <datalist id="genre-options">
                {genres.map((name) => (
                  <option key={name} value={name} />
                ))}
              </datalist>

              <div className="row">
                <button
                  type="button"
                  onClick={rollRandomGenre}
                  disabled={busy || runActive || genres.length === 0}
                  className="btn-sm"
                >
                  🎲 Random
                </button>
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  disabled={busy || runActive || !query}
                  className="btn-sm"
                >
                  Clear
                </button>
              </div>
            </div>
          </section>
        </div>
      </header>

      <div className="info-card">
        <h2 className="section-title">Select an image</h2>

        {apiDown && (
          <div className="status-error">
            Can’t reach the backend at {API_URL}. Is `uvicorn main:app
            --reload` running?
          </div>
        )}

        <div className="sample-grid">
          {(sampleGridExpanded ? sampleImages : sampleImages.slice(0, 6)).map(
            (name) => (
              <button
                key={name}
                type="button"
                onClick={() => selectSampleImage(name)}
                disabled={runActive || busy}
                className={
                  "sample-thumb" +
                  (selectedSample === name ? " sample-thumb-selected" : "")
                }
                aria-label={`Select sample image: ${name.replace(/\.png$/, "")}`}
                aria-pressed={selectedSample === name}
              >
                <img src={`${API_URL}/test-image-files/${name}`} alt="" />
              </button>
            ),
          )}
        </div>

        {sampleImages.length > 6 && (
          <div className="sample-grid-toggle-row">
            <button
              type="button"
              onClick={() => setSampleGridExpanded((expanded) => !expanded)}
              className="sample-grid-toggle"
              aria-label={sampleGridExpanded ? "Show fewer sample images" : "Show all sample images"}
              aria-expanded={sampleGridExpanded}
            >
              {sampleGridExpanded ? "▲" : "▼"}
            </button>
          </div>
        )}
      </div>

      <div className="upload-card">
        <h2 className="section-title">Upload your own</h2>

        <input
          type="file"
          accept="image/*"
          onChange={(event) => {
            setFile(event.target.files?.[0] ?? null);
            setSelectedSample(null);
          }}
          disabled={runActive || busy}
          className="file-picker"
        />

        <div className="run-buttons">
          <button
            onClick={() => startRun(["siglip"])}
            disabled={!hasImage || runActive || busy}
            className="btn-primary btn-lg"
          >
            Use SigLIP
          </button>
          <button
            onClick={() => startRun(algorithms.map((a) => a.id))}
            disabled={!hasImage || runActive || busy || algorithms.length === 0}
            className="btn-primary btn-lg"
          >
            5 Algorithm Suite
          </button>

          <label className="judges-toggle">
            <input
              type="checkbox"
              checked={judgesEnabled}
              disabled={runActive || busy}
              onChange={(event) => setJudgesEnabled(event.target.checked)}
            />
            LLM judges
          </label>
        </div>


        {error && <p className="status-error">{error}</p>}
        {notice && <p className="status-notice">{notice}</p>}
      </div>

      {runActive && (
        <div className="runs">
          {runAlgorithms.length > 1 && (
            <div className="actions">
              <button
                type="button"
                onClick={handleGenerateAll}
                disabled={pendingGenerateAlgorithms.length === 0}
                className="btn-primary"
              >
                {pendingGenerateAlgorithms.length > 0
                  ? `Generate all ${pendingGenerateAlgorithms.length} in parallel (${pendingGenerateAlgorithms.length * 11} credits)`
                  : "Generate all in parallel"}
              </button>
              <button
                type="button"
                onClick={handleDownloadReport}
                disabled={stillTagging}
                className="btn-sm"
              >
                Download tag analysis (.html)
              </button>
            </div>
          )}

          {runAlgorithms.map((algorithm) => (
            <RunRow
              key={algorithm}
              algorithmLabel={algorithmLabels[algorithm] ?? algorithm}
              previewUrl={previewUrl}
              result={
                results[algorithm] ?? { status: "loading" as const }
              }
              onGenerate={() => handleGenerate(algorithm)}
            />
          ))}
        </div>
      )}

      <div className="danger-zone">
        <button
          onClick={handleClear}
          disabled={!hasImage || busy}
          className={`btn-lg btn-danger${armed ? " is-armed" : ""}`}
        >
          {armed ? "Clear this run? Click again" : "Clear"}
        </button>
        <p className="note">
          Deletes the current image and any generated MP3s. Nothing is
          backed up. Required before picking a new image.
        </p>
      </div>
    </main>
  );
}
