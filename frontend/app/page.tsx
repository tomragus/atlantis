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
  embedding?: Embedding;
  // CoCa-only: a free-text caption from its captioning head. Display-only --
  // never fed into tag selection or the Suno prompt.
  caption?: string;
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

// The frontend (localhost:3000) and backend (127.0.0.1:8000) are different
// origins, so a plain `<a download href={fileUrl(...)}>` won't reliably
// force a save -- browsers ignore the `download` attribute cross-origin and
// just navigate instead. Fetching the audio as a blob first and saving
// *that* via a temporary object URL works regardless of origin, since the
// blob URL belongs to this page once created.
async function downloadTrack(track: Track) {
  const response = await fetch(fileUrl(track.audioFile));
  if (!response.ok) throw new Error("Could not download the track.");
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = track.audioFile.split("/").pop() ?? "track.mp3";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
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
        title="judge review 1 -- mood fit"
        emptyMessage="Judge #1 found no tags that didn't fit the image's mood this run."
        rows={tagging.judge
          .filter((v) => v.flagged)
          .map((v) => ({ key: v.tag, flaggedTag: v.tag, reason: v.reason }))}
      />

      <JudgeReviewSection
        title="judge review 2 -- contradictions"
        emptyMessage="Judge #2 identified no contradictory tags this run."
        rows={tagging.contradictions.map((c, index) => {
          const kept = c.drop === c.tag_a ? c.tag_b : c.tag_a;
          return {
            key: `${c.tag_a}-${c.tag_b}-${index}`,
            flaggedTag: c.drop,
            reason: `contradicted “${kept}” -- ${c.reason}`,
          };
        })}
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

  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const algorithmLabels = useMemo(
    () => Object.fromEntries(algorithms.map((a) => [a.id, a.label])),
    [algorithms],
  );

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
  }, []);

  // A button that stays armed is its own trap, so it disarms itself shortly
  // after being primed.
  useEffect(() => {
    if (!armed) return;
    const timer = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(timer);
  }, [armed]);

  // A live preview of whatever image is picked. Memoized rather than stored
  // in state so creating it can't itself trigger a render; the effect below
  // only handles the revoke.
  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  );
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

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

  async function tagOneAlgorithm(algorithm: string) {
    if (!file) return;

    const body = new FormData();
    body.append("image", file);
    const params = new URLSearchParams({ algorithm });
    if (genre) params.set("genre", genre);

    try {
      const data: Analysis = await apiRequest(
        `${API_URL}/tag?${params}`,
        { method: "POST", body },
        `Could not analyze the image with ${algorithm}.`,
      );
      setResults((previous) => ({
        ...previous,
        [algorithm]: { status: "ready", analysis: data },
      }));
    } catch (err) {
      setResults((previous) => ({
        ...previous,
        [algorithm]: {
          status: "error",
          error: err instanceof Error ? err.message : String(err),
        },
      }));
    }
  }

  // FormData is what makes these file uploads. Note neither /tag nor
  // /generate ever sets a Content-Type header: the browser has to set that
  // itself so it can include the multipart boundary marker.
  function startRun(algorithmIds: string[]) {
    if (!file || runActive) return;

    setError(null);
    setNotice(null);
    setResults(
      Object.fromEntries(algorithmIds.map((id) => [id, { status: "loading" as const }])),
    );

    // Independent parallel calls, one per algorithm -- each updates its own
    // slot in `results` as it finishes, so one algorithm's failure or delay
    // never blocks or blanks the others.
    algorithmIds.forEach((id) => {
      tagOneAlgorithm(id);
    });
  }

  async function handleGenerate(algorithm: string) {
    const result = results[algorithm];
    if (!file || !result?.analysis) return;

    setResults((previous) => ({
      ...previous,
      [algorithm]: { ...previous[algorithm], generating: true, generateError: undefined },
    }));

    // Unlike /tag, /generate never re-tags the image -- it only needs the
    // original filename (for the audio naming convention), not the bytes,
    // so there's no reason to re-upload the whole file here.
    const body = new FormData();
    body.append("image_filename", file.name);
    body.append("tags", JSON.stringify(result.analysis.tags));
    body.append("tag_scores", JSON.stringify(result.analysis.tag_scores));
    body.append("tagging", JSON.stringify(result.analysis.tagging));
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
        <h2 className="section-title">How the tagging works</h2>
        <ol className="info-steps">
          <li>
            The picked algorithm embeds the image and scores it against a
            fixed vocabulary of adjectives grouped into 8 facets: color,
            light, mood, valence, arousal, texture, energy, style.
          </li>
          <li>
            A per-facet quota (2 color, 1 light, 3 mood, 1 valence, 1
            arousal, 2 texture, 1 energy, 1 style) picks the top-scoring
            words from each facet, so the 10 tags stay diverse instead of
            collapsing onto one idea and its synonyms.
          </li>
          <li>
            Claude reviews the tags against the image and flags any that read
            as flatly wrong for its mood; each flagged tag is dropped, not
            replaced. A second pass then checks for tags that contradict each
            other&rsquo;s mood across facets and drops one of any such pair.
          </li>
          <li>
            The final tags, plus an optional genre, become the prompt sent
            to Suno to render the song.
          </li>
        </ol>
      </div>

      <div className="upload-card">
        <h2 className="section-title">Select an image:</h2>

        {apiDown && (
          <div className="status-error">
            Can’t reach the backend at {API_URL}. Is `uvicorn main:app
            --reload` running?
          </div>
        )}

        <input
          type="file"
          accept="image/*"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          disabled={runActive || busy}
          className="file-picker"
        />

        <div className="run-buttons">
          <button
            onClick={() => startRun(["siglip"])}
            disabled={!file || runActive || busy}
            className="btn-primary btn-lg"
          >
            Use SigLIP
          </button>
          <button
            onClick={() => startRun(algorithms.map((a) => a.id))}
            disabled={!file || runActive || busy || algorithms.length === 0}
            className="btn-primary btn-lg"
          >
            5 Algorithm Suite
          </button>
        </div>


        {error && <p className="status-error">{error}</p>}
        {notice && <p className="status-notice">{notice}</p>}
      </div>

      {runActive && (
        <div className="runs">
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
          disabled={!file || busy}
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
