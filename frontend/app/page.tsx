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

// A 2D PCA projection of the final tags' vectors in the chosen algorithm's
// embedding space, with the image vector projected onto that same basis --
// illustrative only, see `embedding_projection` in
// backend/embedding_tagger.py for the caveats.
type Embedding = {
  model: string;
  source_dimensions: number;
  image: { x: number; y: number };
  points: EmbeddingPoint[];
};

// How the final tags were arrived at: the first judge's per-tag verdicts,
// the second judge's cross-facet contradiction pairs, and the final tags --
// each labeled with its facet. A tag either judge flags is dropped outright,
// not replaced.
type Tagging = {
  judge: JudgeVerdict[];
  contradictions: Contradiction[];
  final: FacetTag[];
  // Whether this run actually invoked the Claude judges (the "LLM judges"
  // toggle's value at request time) -- an empty `judge`/`contradictions`
  // list looks the same whether the judges ran and flagged nothing, or
  // never ran at all, so this is what tells those two cases apart.
  judges_enabled: boolean;
  embedding?: Embedding;
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

// Which of a judge pass's verdicts actually get shown as a dropped tag, and
// what to say about each -- shared data-shaping for JudgeReviewSection below.
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

// A 2D scatter of the final tags, color-coded by facet, with the image's
// projected position marked separately, on its own dark panel.
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
        aria-label={`2D projection of the final tags and the image in ${algorithmLabel}'s embedding space. Hover a point for its tag and facet.`}
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

      <p className="note">Lorem ipsum</p>
    </div>
  );
}

// One tag pill for the final tag row.
function TagChip({
  tag,
  facet,
  score,
}: {
  tag: string;
  facet: string;
  score: number;
}) {
  return (
    <span className="tag-group">
      <span className="tag-facet">{facet}</span>
      <span className="tag">
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

// The tag row and judge review -- shared by every algorithm row in the
// current run, so SigLIP mode and Suite mode never drift apart in how they
// present a tag set. Its one call site (RunRow) only renders this once
// tagging has actually succeeded, so every prop here is guaranteed to exist
// by then -- no defaulting needed.
function TagBreakdown({ tagging }: { tagging: Tagging }) {
  return (
    <>
      {tagging.final.length > 0 && (
        <div className="tags">
          {tagging.final.map(({ tag, facet, score }) => (
            <TagChip key={tag} tag={tag} facet={facet} score={score} />
          ))}
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
  showEmbedding,
  reversed,
}: {
  algorithmLabel: string;
  previewUrl: string | null;
  result: RunResult;
  onGenerate: () => void;
  showEmbedding: boolean;
  reversed: boolean;
}) {
  const embedding = result.analysis?.tagging.embedding;

  return (
    <div className={`run-row${reversed ? " run-row-reversed" : ""}`}>
      <div className="run-row-media">
        {previewUrl && (
          // eslint-disable-next-line @next/next/no-img-element -- local blob: URL, not a remote image
          <img src={previewUrl} alt="" className="run-row-image" />
        )}

        {/* Fills the space below the image rather than leaving it blank --
            same data the old inline embedding section showed, just relocated
            so the image column isn't dead space once tagging finishes. Suite
            mode only -- SigLIP-only's single row skips it entirely. */}
        {showEmbedding && embedding && embedding.points.length > 0 && result.analysis && (
          <div className="embedding-section">
            <span className="meta-label">{algorithmLabel} Joint Embedding Space</span>
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
          <span className="run-name">{algorithmLabel} Extracted Tags</span>
        </div>

        {result.status === "loading" && (
          <p className="status-notice">Tagging with {algorithmLabel}…</p>
        )}

        {result.status === "error" && (
          <p className="status-error">{result.error}</p>
        )}

        {result.status === "ready" && result.analysis && (
          <>
            <TagBreakdown tagging={result.analysis.tagging} />

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
  const [judgesEnabled, setJudgesEnabled] = useState(false);

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
  // generated (or started generating) yet -- feeds the SigLIP-only auto-chain
  // effect below. Memoized on `results` so its identity only changes when
  // membership actually does, not on every render -- the effect below depends
  // on it directly.
  const pendingGenerateAlgorithms = useMemo(
    () =>
      Object.keys(results).filter((id) => {
        const result = results[id];
        return (
          result?.status === "ready" &&
          result.analysis &&
          !result.generated &&
          !result.generating &&
          !result.generateError
        );
      }),
    [results],
  );

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
    // so there's no reason to re-upload the whole file here.
    const body = new FormData();
    body.append("image_filename", file ? file.name : (selectedSample ?? ""));
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

  // SigLIP-only ("Generate Song") auto-chains straight from tagging into
  // generation -- no manual second click, unlike Suite mode's per-row
  // "Generate song" button. `pendingGenerateAlgorithms` already excludes
  // anything generating/generated/not-ready, so this only ever fires once
  // per run: reading `results` (not calling handleGenerate synchronously
  // inside startRun's tagging `.then()`) avoids acting on a stale closure
  // of `results` from before the tagging state actually committed.
  useEffect(() => {
    if (runAlgorithms.length !== 1) return;
    if (pendingGenerateAlgorithms.length !== 1) return;
    handleGenerate(pendingGenerateAlgorithms[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- handleGenerate is redefined every render; pendingGenerateAlgorithms already tracks everything this effect needs to re-fire on
  }, [pendingGenerateAlgorithms, runAlgorithms.length]);

  return (
    <main className="page">
      <header className="hero">
        <h1 className="title">Ideagream Studio</h1>
        <p className="subtitle">
          A better way to compose mood-accurate songs from images using joint
          embedding spaces
        </p>
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
            className="btn-primary btn-lg btn-run-primary"
          >
            Generate Song
          </button>
          <button
            onClick={() => startRun(algorithms.map((a) => a.id))}
            disabled={!hasImage || runActive || busy || algorithms.length === 0}
            className="btn-primary btn-lg btn-run-secondary"
          >
            {algorithms.length} Algorithm Suite
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
          {runAlgorithms.map((algorithm, index) => (
            <RunRow
              key={algorithm}
              algorithmLabel={algorithmLabels[algorithm] ?? algorithm}
              previewUrl={previewUrl}
              result={
                results[algorithm] ?? { status: "loading" as const }
              }
              onGenerate={() => handleGenerate(algorithm)}
              showEmbedding={runAlgorithms.length > 1}
              reversed={index % 2 === 1}
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
      </div>

      <div className="genre-card">
        <h2 className="section-title">
          Genre Knob <span className="meta-label">Optional</span>
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
    </main>
  );
}
