"use client";

import { useEffect, useMemo, useState } from "react";

import GenreKnob from "./GenreKnob";

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

type Replacement = {
  from: string;
  to: string;
};

type EmbeddingPoint = {
  tag: string;
  facet: string;
  x: number;
  y: number;
};

// A 2D PCA projection of the top-20 candidates' SigLIP vectors, with the
// image vector projected onto that same basis -- illustrative only, see
// `embedding_projection` in backend/siglip_tagger.py for the caveats.
type Embedding = {
  model: string;
  source_dimensions: number;
  image: { x: number; y: number };
  points: EmbeddingPoint[];
};

// How the final 10 tags were arrived at: SigLIP's full top-20 candidates, the
// pre-judge selection, the judge's per-tag verdicts, which flagged tags got
// swapped for what, and the final tags -- each labeled with its facet.
// Absent on runs saved before this was tracked.
type Tagging = {
  top20: FacetTag[];
  selected: FacetTag[];
  judge: JudgeVerdict[];
  final: FacetTag[];
  replacements: Replacement[];
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

// Suno renders two takes of the same prompt per generation, and both are kept.
type Generation = {
  timestamp: string;
  image: string;
  genre: string;
  tags: string[];
  // SigLIP's confidence per tag, keyed by the tag itself.
  tag_scores: Record<string, number>;
  tagging?: Tagging;
  prompt: string;
  job_id: string;
  tracks: Track[];
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
};

// Saved images and MP3s are served by the backend out of its history folder.
// Audio filenames contain spaces, `&` and apostrophes, so each path segment is
// escaped separately -- encoding the whole path would eat the `/` too.
const fileUrl = (path: string) =>
  `${API_URL}/files/${path.split("/").map(encodeURIComponent).join("/")}`;

// Shared by every mutating call (wipe history, extract tags, generate song):
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

// SigLIP's scores sit in a narrow band near zero, so a flat 3 decimals reads
// as "0.000" for most of them -- indistinguishable from an actual zero. Shows
// at least 3 decimals, extended just far enough to surface the first
// non-zero digit for smaller scores.
function formatScore(score: number): string {
  if (!Number.isFinite(score) || score === 0) return "0.000";
  const magnitude = Math.floor(Math.log10(Math.abs(score)));
  const decimals = Math.min(Math.max(-magnitude, 3), 6);
  return score.toFixed(decimals);
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
// SigLIP's image/text "modality gap" means the image often lands well
// outside the tag cluster (see embedding_projection's docstring in
// siglip_tagger.py) -- rather than stretch the shared scale to fit it and
// crush all 20 tags into a corner, the image marker is clamped to the plot's
// edge and flagged as approximate. Points aren't directly labeled (they
// overlap too easily when several near-synonyms land close together) --
// hover or focus a point for its exact tag and facet instead.
function EmbeddingPlot({
  embedding,
  finalTags,
}: {
  embedding: Embedding;
  finalTags: Set<string>;
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
        aria-label="2D projection of the top 20 tag candidates and the image in SigLIP's embedding space. Hover a point for its tag and facet."
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
        SigLIP embeds the image and each candidate word into a{" "}
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
        together really are similar to each other in SigLIP&rsquo;s full
        space. Distance from a tag to the image marker isn&rsquo;t reliable
        the same way -- SigLIP&rsquo;s image and text embeddings occupy
        separate regions of the space almost regardless of how well a word
        actually fits (a &ldquo;modality gap&rdquo;), so the
        image&rsquo;s position here says little about any one tag&rsquo;s
        fit.
      </p>
    </div>
  );
}

// The tag row, judge review, top-20 candidates and prompt line -- shared by
// the pending analysis card (before a song exists) and each history run card
// (after one does), so the two views never drift apart.
function TagBreakdown({
  tags,
  tagScores,
  tagging,
  prompt,
}: {
  tags: string[];
  tagScores: Record<string, number>;
  tagging?: Tagging;
  prompt: string;
}) {
  // Runs saved before tagging detail was tracked fall back to the plain
  // tag+score list, with no facet label.
  const finalTags: { tag: string; facet?: string; score?: number }[] =
    tagging?.final ?? tags.map((tag) => ({ tag, score: tagScores?.[tag] }));

  return (
    <>
      {finalTags.length > 0 && (
        <>
          {/* One tag set drove both takes, so it sits above them rather than
              being repeated. Strongest SigLIP match first. */}
          <div className="tags">
            {finalTags.map(({ tag, facet, score }) => (
              <span key={tag} className="tag-group">
                {facet && <span className="tag-facet">{facet}</span>}
                <span className="tag">
                  {tag}
                  {score !== undefined && (
                    <span className="tag-score">{formatScore(score)}</span>
                  )}
                </span>
              </span>
            ))}
          </div>
          <p className="note">
            The number beside each tag is SigLIP&rsquo;s match probability for that
            word against this image (0–1) -- not a rank. Scores across the
            whole vocabulary sit close to zero, so small differences still
            mean one word fit noticeably better than another.
          </p>
        </>
      )}

      {tagging &&
        (tagging.replacements.length > 0 ||
          tagging.judge.some((v) => v.flagged)) && (
          <div className="judge-review">
            <span className="meta-label">judge review</span>
            <ul className="judge-list">
              {tagging.judge
                .filter((v) => v.flagged)
                .map((verdict) => {
                  const replacement = tagging.replacements.find(
                    (r) => r.from === verdict.tag,
                  );
                  return (
                    <li key={verdict.tag} className="judge-item">
                      <span className="judge-flagged">{verdict.tag}</span>
                      {replacement && (
                        <>
                          <span className="judge-arrow" aria-hidden="true">
                            →
                          </span>
                          <span className="judge-replacement">
                            {replacement.to}
                          </span>
                        </>
                      )}
                      {verdict.reason && (
                        <div className="judge-reason">{verdict.reason}</div>
                      )}
                    </li>
                  );
                })}
            </ul>
          </div>
        )}

      {tagging && tagging.top20.length > 0 && (
        <div className="top20">
          <span className="meta-label">top 20 candidates</span>
          <div className="tags tags-muted">
            {tagging.top20.map(({ tag, facet, score }) => {
              const isFinal = tagging.final.some((f) => f.tag === tag);
              return (
                <span
                  key={tag}
                  className={`tag-group${isFinal ? " is-selected" : ""}`}
                >
                  <span className="tag-facet">{facet}</span>
                  <span className="tag tag-muted">
                    {tag}
                    <span className="tag-score">{formatScore(score)}</span>
                  </span>
                </span>
              );
            })}
          </div>
        </div>
      )}

      {tagging?.embedding && tagging.embedding.points.length > 0 && (
        <div className="embedding-section">
          <span className="meta-label">embedding space (illustrative)</span>
          <EmbeddingPlot
            embedding={tagging.embedding}
            finalTags={new Set(tagging.final.map((f) => f.tag))}
          />
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

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [history, setHistory] = useState<Generation[]>([]);
  // Two separate in-flight flags: analyzing is the free, fast /tag call;
  // loading is the Suno render, which costs credits and takes ~2 minutes.
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const busy = analyzing || loading;
  const [error, setError] = useState<string | null>(null);
  const [genres, setGenres] = useState<string[]>([]);
  // The text in the search box, and the single source of truth for the genre.
  // A half-typed word simply isn't a genre yet, which is what `genre` derives.
  const [query, setQuery] = useState("");
  const genre = genres.includes(query) ? query : "";
  // Deleting is irreversible and nothing is backed up, so the button has to be
  // clicked twice. This holds whether it is currently armed.
  const [armed, setArmed] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  // Which run cards the user has opened or closed by hand. Runs not in here
  // follow the default: the newest is open, the rest are collapsed.
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  // Set when the first loads can't reach the backend, so the page can say so
  // instead of looking like an app with every button broken.
  const [apiDown, setApiDown] = useState(false);

  // Genre leads the prompt, so a stale analysis would send the wrong prompt
  // to Suno if the genre changed after analyzing. Re-analyzing is cheap, so
  // just clear it rather than silently patching the prompt. Adjusted during
  // render (same pattern as GenreKnob's own prop-follow) rather than in an
  // effect, so it can't cause an extra paint.
  const [seenGenre, setSeenGenre] = useState(genre);
  if (genre !== seenGenre) {
    setSeenGenre(genre);
    setAnalysis(null);
  }

  // Load past runs and the genre list once, when the page first mounts. A
  // failure here means the backend is unreachable, which is worth saying
  // plainly: without it every control sits disabled with no genres and no
  // history, which reads as a broken app rather than a stopped server.
  useEffect(() => {
    const offline = (message: string) => () => {
      setError(message);
      setApiDown(true);
    };

    fetch(`${API_URL}/history`)
      .then((response) => response.json())
      .then(setHistory)
      .catch(offline("Could not load past songs."));

    fetch(`${API_URL}/genres`)
      .then((response) => response.json())
      .then(setGenres)
      .catch(offline("Could not load the genre list."));
  }, []);

  // A button that stays armed is its own trap, so it disarms itself shortly
  // after being primed.
  useEffect(() => {
    if (!armed) return;
    const timer = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(timer);
  }, [armed]);

  // A live preview of whatever image is picked, for the pending analysis
  // card. Memoized rather than stored in state so creating it can't itself
  // trigger a render; the effect below only handles the revoke.
  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  );
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function handleWipe() {
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
        "Could not clear the history.",
      );
      setHistory([]);
      setNotice(
        `Deleted ${data.entries_removed} runs, ${data.files_removed} files, ` +
          `${data.megabytes_freed} MB freed.`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function rollRandomGenre() {
    if (genres.length > 0) {
      setQuery(genres[Math.floor(Math.random() * genres.length)]);
    }
  }

  // FormData is what makes these file uploads. Note neither ever sets a
  // Content-Type header: the browser has to set that itself so it can
  // include the multipart boundary marker.

  async function handleAnalyze() {
    if (!file) return;

    setAnalyzing(true);
    setError(null);
    setNotice(null);
    setAnalysis(null);

    const body = new FormData();
    body.append("image", file);

    const genreParam = genre ? `?genre=${encodeURIComponent(genre)}` : "";

    try {
      const data = await apiRequest(
        `${API_URL}/tag${genreParam}`,
        { method: "POST", body },
        "Could not analyze the image.",
      );
      setAnalysis(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleGenerateSong() {
    if (!file || !analysis) return;

    setLoading(true);
    setError(null);
    setNotice(null);

    const body = new FormData();
    body.append("image", file);
    body.append("tags", JSON.stringify(analysis.tags));
    body.append("tag_scores", JSON.stringify(analysis.tag_scores));
    body.append("tagging", JSON.stringify(analysis.tagging));
    body.append("prompt", analysis.prompt);

    const genreParam = genre ? `?genre=${encodeURIComponent(genre)}` : "";

    try {
      const data = await apiRequest(
        `${API_URL}/generate${genreParam}`,
        { method: "POST", body },
        "Could not generate a song.",
      );
      setHistory((previous) => [data, ...previous]);
      setAnalysis(null);
      setFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <header className="hero">
        <h1 className="title">Image to Song</h1>
        <p className="subtitle">Turn a picture into a piece of music.</p>
      </header>

      <div className="rail-column">
        <div className="rail">
          {apiDown && (
            <div className="status-error">
              Can’t reach the backend at {API_URL}. Is `uvicorn main:app
              --reload` running?
            </div>
          )}

          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              // A pending analysis is tied to whatever image produced it --
              // swapping the file without re-analyzing would otherwise let
              // Generate song save the new image next to the old one's tags,
              // judge review and prompt.
              setFile(event.target.files?.[0] ?? null);
              setAnalysis(null);
            }}
            disabled={busy}
            className="file-picker"
          />

          <section className="genre">
            <GenreKnob
              genres={genres}
              value={genre}
              onChange={setQuery}
              disabled={busy}
            />

            <div className="genre-fields">
              <label htmlFor="genre-search" className="genre-label">
                Genre <span className="meta-label">(optional)</span>
              </label>

              <input
                id="genre-search"
                list="genre-options"
                value={query}
                disabled={busy}
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
                  disabled={busy || genres.length === 0}
                  className="btn-sm"
                >
                  🎲 Random
                </button>
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  disabled={busy || !query}
                  className="btn-sm"
                >
                  Clear
                </button>
              </div>

              <p className="hint">
                {genre
                  ? `Suno will be told to make ${genre}.`
                  : "No genre: the song comes only from the picture."}
              </p>
            </div>
          </section>

          <button
            onClick={handleAnalyze}
            disabled={!file || busy}
            className="btn-primary btn-lg"
          >
            {analyzing ? "Extracting…" : "Extract tags"}
          </button>
          <p className="note">
            Free and fast: scores the image against the tag vocabulary and
            asks the judge to review it. Nothing is sent to Suno until you
            generate.
          </p>

          {error && <p className="status-error">{error}</p>}
          {notice && <p className="status-notice">{notice}</p>}

          {history.length > 0 && (
            <div className="danger-zone">
              <button
                onClick={handleWipe}
                disabled={busy}
                className={`btn-sm btn-danger${armed ? " is-armed" : ""}`}
              >
                {armed
                  ? `Delete ${history.length} runs? Click again`
                  : "Clear history"}
              </button>
              <p className="note">
                Deletes every saved image and MP3. Nothing is backed up.
              </p>
            </div>
          )}
        </div>

        <div className="info-card">
          <h2 className="section-title">How the tagging works</h2>
          <ol className="info-steps">
            <li>
              SigLIP embeds the image and scores it against a fixed
              vocabulary of adjectives grouped into 8 facets: color, light,
              mood, valence, arousal, texture, energy, style.
            </li>
            <li>
              A per-facet quota (2 color, 1 light, 3 mood, 1 valence, 1
              arousal, 2 texture, 1 energy, 1 style) picks the top-scoring
              words from each facet, so the 10 tags stay diverse instead of
              collapsing onto one idea and its synonyms.
            </li>
            <li>
              Claude reviews the 10 tags against the image and flags any that
              read as flatly wrong for its mood; each flagged tag is swapped
              for the next-best word in the same facet.
            </li>
            <li>
              The final tags, plus an optional genre, become the prompt sent
              to Suno to render the song.
            </li>
          </ol>
        </div>
      </div>

      <div className="stream">
        {analysis && (
          <div className="run-card is-open pending-card">
            <div className="run-head">
              <span className="run-head-text">
                <span className="run-name">
                  Tag Analysis: SigLIP joint embedding
                </span>
                <span className="run-stamp">
                  {genre || "no genre"} · not generated yet
                </span>
              </span>
            </div>

            <div className="run-body">
              {previewUrl && (
                // eslint-disable-next-line @next/next/no-img-element -- local blob: URL, not a remote image
                <img src={previewUrl} alt="" className="run-image" />
              )}

              <TagBreakdown
                tags={analysis.tags}
                tagScores={analysis.tag_scores}
                tagging={analysis.tagging}
                prompt={analysis.prompt}
              />

              <div className="actions">
                <button
                  onClick={handleGenerateSong}
                  disabled={loading}
                  className="btn-primary"
                >
                  {loading ? "Composing…" : "Generate song"}
                </button>
                <button
                  onClick={() => setAnalysis(null)}
                  disabled={loading}
                  className="btn-sm"
                >
                  Discard
                </button>
              </div>

              {loading && (
                <p className="status-notice">
                  This takes about two minutes. Keep this tab open.
                </p>
              )}
            </div>
          </div>
        )}

        {history.length === 0 && !analysis && !apiDown && (
          <p className="stream-empty">
            Nothing here yet. Pick an image on the left and hit Extract tags.
          </p>
        )}

        <ul className="runs">
          {history.map((entry, entryIndex) => {
            const runKey = entry.image;
            const open = expanded[runKey] ?? entryIndex === 0;

            return (
              <li key={runKey} className={`run-card${open ? " is-open" : ""}`}>
                <button
                  type="button"
                  className="run-head"
                  aria-expanded={open}
                  onClick={() =>
                    setExpanded((previous) => ({ ...previous, [runKey]: !open }))
                  }
                >
                  <span className="run-index" aria-hidden="true">
                    {String(history.length - entryIndex).padStart(2, "0")}
                  </span>

                  {/* eslint-disable-next-line @next/next/no-img-element -- next/image would need remote-host config for a local dev-only backend */}
                  <img
                    src={fileUrl(entry.image)}
                    alt=""
                    width={46}
                    height={46}
                    className="run-thumb"
                  />

                  <span className="run-head-text">
                    <span className="run-name">
                      {entry.tracks[0]?.title ?? "Untitled"}
                    </span>
                    <span className="run-stamp">
                      {new Date(entry.timestamp).toLocaleString()}
                      {entry.genre && ` · ${entry.genre}`}
                    </span>
                  </span>

                  <span className="run-caret" aria-hidden="true">
                    {open ? "▼" : "►"}
                  </span>
                </button>

                {open && (
                  <div className="run-body">
                    {/* eslint-disable-next-line @next/next/no-img-element -- same local-backend reasoning as run-thumb above */}
                    <img
                      src={fileUrl(entry.image)}
                      alt=""
                      className="run-image"
                    />

                    <TagBreakdown
                      tags={entry.tags}
                      tagScores={entry.tag_scores}
                      tagging={entry.tagging}
                      prompt={entry.prompt}
                    />

                    {entry.tracks.map((track, index) => (
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
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </main>
  );
}
