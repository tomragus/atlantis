"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";

import SpinKnob from "./SpinKnob";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Track = {
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
  // Same idea as judges_enabled, for the "Embedding space visual" toggle --
  // `embedding`'s mere absence already implies this was false, but an
  // explicit field means nothing has to know that implicit contract.
  embedding_enabled: boolean;
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
// call yet. Held in state until the user commits with "Generate Songs", at
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

// The current run's tagging/generation state, for whichever algorithm the
// dropdown had selected when "Extract Tags" was clicked.
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

// The tag row and judge review. Its one call site (RunRow) only renders
// this once tagging has actually succeeded, so every prop here is
// guaranteed to exist by then -- no defaulting needed.
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

// The run's image on the left, its tag breakdown on the right, and (once
// tagging succeeded) its own Generate Songs action and result.
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
            so the image column isn't dead space once tagging finishes. Only
            present at all when the "Embedding space visual" toggle was on
            for this run -- the backend omits `tagging.embedding` entirely
            otherwise, so its mere presence here is the gate. */}
        {embedding && embedding.points.length > 0 && result.analysis && (
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
                  {result.generating ? "Composing…" : "Generate Songs"}
                </button>
              </div>
            )}

            {result.generating && (
              <p className="status-notice">
                Generating... takes about 2 minutes.
              </p>
            )}

            {result.generateError && (
              <p className="status-error">{result.generateError}</p>
            )}

            {result.generated?.tracks.map((track, index) => (
              <div key={track.audioFile} className="take">
                <div className="take-title">Take {index + 1}</div>
                <audio controls src={fileUrl(track.audioFile)} />
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// The HF checkpoint string per algorithm id -- same ids/strings as
// backend/embedding_tagger.py's HF_MODEL_IDS. Only the id needs hardcoding
// here; the display label comes from the already-fetched `algorithms` prop
// (GET /algorithms, i.e. backend/algorithms.py's ALGORITHM_INFO) rather than
// a second hand-typed copy that could drift from it.
const CHECKPOINT_IDS: Record<string, string> = {
  siglip: "google/siglip2-base-patch16-224",
  clip: "openai/clip-vit-base-patch32",
  align: "kakaobrain/align-base",
  metaclip: "facebook/metaclip-b32-400m",
};

// Plain background text -- no card, no border -- explaining the pipeline for
// anyone curious enough to read past the app itself. Only prop is the
// already-fetched algorithm list, for the checkpoint links' labels; memoized
// since this ~170-line static subtree never needs to re-render on Home's
// other state changes (a genre-search keystroke, a knob drag, a run status
// update) once `algorithms` has settled after its one-time fetch.
const AboutSection = memo(function AboutSection({
  algorithms,
}: {
  algorithms: AlgorithmInfo[];
}) {
  return (
    <section className="about">
      <h2 className="about-heading">How it works</h2>

      <p>
        SigLIP2 (Sigmoid Loss for Language-Image Pre-Training 2) from Google
        was trained on huge batches of image-caption pairs scraped from the
        internet.
      </p>

      <p>
        When you upload an image, the SigLIP2 embedding algorithm’s image
        encoder splits the image up into a 14x14 grid of tiles, each of which
        gets turned into a 768-dimensional vector, which is a rich numerical
        fingerprint capturing color, texture, edges and more. These vectors
        are initially independent (meaning, each one has no knowledge of the
        others), but are then fed into 12 transformer layers in which each
        vector gets iteratively updated based on context from the others.
        After this process, the 196 tiles are fed into a single pooling
        layer, which asks each tile “what do you know about this image?”,
        and composes a single 768-dimensional vector summarizing the
        responses.
      </p>

      <p>
        Simultaneously, the same embedding process is happening on a curated
        set of 261 descriptive vocabulary tags covering 8 distinct regions of
        “feel”, including color, light, mood, valence, arousal, texture,
        energy and style. Because SigLIP2 was trained on captions instead of
        individual words, each tag gets wrapped into 4 short template
        sentences:
      </p>

      <ul>
        <li>“a photo that feels {"{tag}"}”</li>
        <li>“a {"{tag}"} image”</li>
        <li>“{"{tag}"} atmosphere”</li>
        <li>“a picture with a {"{tag}"} quality”</li>
      </ul>

      <p>
        Each set of these template phrases gets batched and fed into the text
        encoder, after which they are stacked and averaged resulting in a
        single 768-dimensional embedding vector per tag.
      </p>

      <p>
        After encoding, the cosine similarity between the image vector and
        each individual tag are calculated and fed into the sigmoid (hence
        the name), which contains additional signal directly from SigLIP2’s
        pre-training to get a final score. If you showed the image to the
        model and asked “does the phrase ‘a photo that feels {"{tag}"}’
        genuinely describe this image, yes or no?”, then the final score
        would be how confident the model is that the answer is yes.
      </p>

      <p>
        Final scores for tags are ranked and categorized by their distinct
        region of “feel” (referred to in the code as “facet”). The top few
        tags per facet (specifically: 2 for “color”, 1 for “light”, 3 for
        “mood”, 1 for “valence”, 1 for “arousal”, 2 for “texture”, 1 for
        “energy”, and 1 for “style”) are used to compose a prompt for the
        Suno v5.5 model, which is simply the list of tags in order from
        highest to lowest score. That prompt is then fed to Suno v5.5 to produce
        2 highly mood-accurate songs for that image.
      </p>

      <p>
        Through testing across a diverse image corpus, I have found this
        system capable of accurately capturing a wide range of unique moods
        and aesthetics from pleasant to whimsical to punk. It is really fun
        to play with. Please try your own images and see what comes up!
      </p>

      <h3 className="about-heading">Optional LLM-Evaluation</h3>

      <p>
        The default system does not use language models anywhere in the tag
        extraction process, which can sometimes lead to slightly unusual
        tags ranking high for certain images. If you’d like, you can turn on
        LLM-judges for a run for slightly more refined tag lists.
      </p>

      <p>
        LLM-judge #1 has Claude Sonnet 5 look at the image, then each tag,
        and decide “whether it captures the mood or feeling the image
        evokes - not whether it’s a literal, physically accurate
        description” (taken directly from the system prompt). If not, the
        tag gets flagged and then dropped from the prompt.
      </p>

      <p>
        LLM-judge #2 has Claude Sonnet 5 scan the tag list for any seeming
        contradictions between tags across different facets (ex: “joyful”
        from valence and “haunting” from mood). If it identifies a
        contradictory pair, it then checks whether each tag’s feeling is
        already covered by some other tag elsewhere in the list. Whichever
        of the two tags has coverage gets dropped, since it is less
        descriptive in the final Suno v5.5 prompt. While this is somewhat
        uncommon, there were instances in my testing of the system that
        justified the creation of this judge.
      </p>

      <p>
        These LLM-judges sort of “hone-in” the sometimes unpredictability of
        the tags from the embedding model and thus the unpredictability of
        the generated tracks. However, you’ll notice I have the LLM-judges
        set to off by default: it sometimes feels like there is some magic
        contained within these otherwise seemingly unfitting tags, which
        shine through in the audio. If you play around with this enough,
        you’ll know exactly what I mean!
      </p>

      <h3 className="about-heading">Other embedding algorithms</h3>

      <p>
        While this project was optimized for and built around the SigLIP2
        algorithm, I also wanted to explore other embedding algorithms to
        see how they compared: specifically, CLIP (SigLIP’s predecessor and
        the OG joint embedding algorithm), ALIGN from Google (adaptation of
        CLIP that uses EfficientNet for the image encoder and BERT for the
        text encoder), and MetaCLIP from Meta (adaptation of CLIP that
        favors more rare, specific concepts).
      </p>

      <p>
        From my testing, CLIP and MetaCLIP felt like less refined versions
        of SigLIP, producing tracks that sort of had the right idea but
        ultimately tended to sound somewhat generic. ALIGN would
        occasionally knock it out of the park, but I found it to be
        unreliable and inconsistent. I should stress again that this project
        was built around SigLIP2, so it’s very likely that a system like
        this could work well for any of these algorithms if optimized
        properly.
      </p>

      <p>Specific checkpoints (all loaded through HuggingFace transformers.AutoModel):</p>

      <ul>
        {algorithms.map(({ id, label }) => {
          const checkpointId = CHECKPOINT_IDS[id];
          return (
            <li key={id}>
              {label}:{" "}
              <a href={`https://huggingface.co/${checkpointId}`} target="_blank" rel="noreferrer">
                {checkpointId}
              </a>
            </li>
          );
        })}
      </ul>

      <h3 className="about-heading">Genre Knob</h3>

      <p>
        You may have noticed the “Genre Knob”. This is a fun addition that
        appends a genre tag to the start of a tag list to essentially force
        the generated song to conform to a particular genre. You can select
        from one of 17 genres: rock, psychedelic rock, prog rock, math rock,
        metal, heavy metal, thrash metal, dream pop, synth-pop, idm, house,
        techno, trance, jazz, jazz fusion, orchestral, and reggae (this
        selection of genres is not comprehensive by any means, but from my
        testing these were the genres that Suno v5.5 was most consistently
        responsive to).
      </p>

      <p>
        The genre tag tends to overpower the nuances that the embedding
        algorithm captures, so I’ve positioned this as an extra feature not
        central to the app. It is really fun to mess around with though, and
        the tracks that get produced are quite good, which is a testament to
        the quality of the Suno model.
      </p>
    </section>
  );
});

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [genres, setGenres] = useState<string[]>([]);
  // The text in the search box, and the single source of truth for the genre.
  // A half-typed word simply isn't a genre yet, which is what `genre` derives.
  const [query, setQuery] = useState("");
  const genre = genres.includes(query) ? query : "";
  // Whether the Genre Knob modal is open. Picking a genre (spin, search,
  // Random) never closes it on its own -- only an explicit dismissal
  // (X, backdrop click, or Escape) does.
  const [genreModalOpen, setGenreModalOpen] = useState(false);
  // The modal panel itself, so the focus-trap effect below can find its
  // focusable children and the backdrop's mousedown handler can tell "the
  // drag started on the backdrop" apart from "it started inside the panel
  // and was dragged out" (a plain onClick can't tell those apart, since a
  // drag that ends outside the panel still fires its click on the backdrop).
  const genreModalRef = useRef<HTMLDivElement>(null);
  const genreBackdropMouseDownOnSelf = useRef(false);

  // Whether /tag runs its two Claude judge passes for the run about to
  // start. On by default; off skips both calls entirely on the backend
  // (see main.py's `judges` param), not just their result. Only meaningful
  // before a run starts -- the toggle disables once one is active, same as
  // the genre knob, since it can't retroactively affect tagging that
  // already happened.
  const [judgesEnabled, setJudgesEnabled] = useState(false);

  // Whether /tag computes the joint embedding space plot for the run about
  // to start. Off by default; off skips `embedding_projection` entirely on
  // the backend (see main.py's `show_embedding` param), not just its
  // display. Same disable-once-active convention as judgesEnabled.
  const [embeddingEnabled, setEmbeddingEnabled] = useState(false);

  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const algorithmLabels = useMemo(
    () => Object.fromEntries(algorithms.map((a) => [a.id, a.label])),
    [algorithms],
  );
  // Which algorithm "Extract Tags" will run -- SigLIP2 until the user picks
  // something else from the dropdown. Only meaningful before a run starts,
  // same as judgesEnabled/embeddingEnabled.
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("siglip");

  // Filenames offered by the "select an image" grid, and which one (if any)
  // is currently picked -- `null` when the active image instead came from
  // the "upload your own" file input. Both write into the same `file` state
  // below, so picking one always clears the other.
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [sampleGridExpanded, setSampleGridExpanded] = useState(false);

  // The run in progress (or just completed): the algorithm it used and its
  // tagging/generation state. `null` means no run is active, which is also
  // what gates the file input and the algorithm/judges/embedding controls
  // -- one run at a time.
  const [run, setRun] = useState<{ algorithm: string; result: RunResult } | null>(null);
  const runActive = run !== null;
  const busy =
    runActive && (run.result.status === "loading" || Boolean(run.result.generating));

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

  // Escape closes the Genre Knob modal, same as its X button/backdrop click.
  // Tab/Shift+Tab are trapped within the panel so keyboard focus can't reach
  // (and activate) the "Extract Tags" button or algorithm dropdown sitting
  // behind the backdrop while the modal is still open on screen.
  useEffect(() => {
    if (!genreModalOpen) return;
    const panel = genreModalRef.current;
    const focusable = panel
      ? Array.from(
          panel.querySelectorAll<HTMLElement>(
            'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])',
          ),
        )
      : [];
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    (first ?? panel)?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setGenreModalOpen(false);
        return;
      }
      if (event.key !== "Tab" || !first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [genreModalOpen]);

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
      setRun(null);
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

  // FormData is what makes these file uploads. Note neither /tag nor
  // /generate ever sets a Content-Type header: the browser has to set that
  // itself so it can include the multipart boundary marker.
  function startRun(algorithm: string) {
    if (!hasImage || runActive) return;

    setError(null);
    setNotice(null);
    setRun({ algorithm, result: { status: "loading" } });

    const params = new URLSearchParams({
      algorithm,
      judges: String(judgesEnabled),
      show_embedding: String(embeddingEnabled),
    });
    if (genre) params.set("genre", genre);

    let body: FormData | undefined;
    if (file) {
      body = new FormData();
      body.append("image", file);
    } else if (selectedSample) {
      params.set("sample_image", selectedSample);
    }

    apiRequest(
      `${API_URL}/tag?${params}`,
      { method: "POST", body },
      `Could not analyze the image with ${algorithm}.`,
    )
      .then((data: Analysis) => setRun({ algorithm, result: { status: "ready", analysis: data } }))
      .catch((err) =>
        setRun({
          algorithm,
          result: { status: "error", error: err instanceof Error ? err.message : String(err) },
        }),
      );
  }

  async function handleGenerate() {
    if (!run || !hasImage || !run.result.analysis) return;
    const analysis = run.result.analysis;
    const { algorithm, result } = run;

    setRun({ algorithm, result: { ...result, generating: true, generateError: undefined } });

    // Unlike /tag, /generate never re-tags the image -- it only needs the
    // original filename (for the audio naming convention), not the bytes,
    // so there's no reason to re-upload the whole file here.
    const body = new FormData();
    body.append("image_filename", file ? file.name : (selectedSample ?? ""));
    body.append("tags", JSON.stringify(analysis.tags));
    body.append("tag_scores", JSON.stringify(analysis.tag_scores));
    body.append("tagging", JSON.stringify(analysis.tagging));
    body.append("prompt", analysis.prompt);

    const params = new URLSearchParams({ algorithm: analysis.algorithm });
    if (genre) params.set("genre", genre);

    try {
      const data: GeneratedSong = await apiRequest(
        `${API_URL}/generate?${params}`,
        { method: "POST", body },
        "Could not generate a song.",
      );
      setRun({ algorithm, result: { ...result, generating: false, generated: data } });
    } catch (err) {
      setRun({
        algorithm,
        result: {
          ...result,
          generating: false,
          generateError: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }

  return (
    <main className="page">
      <header className="hero">
        <h1 className="title">Ideagram Studio</h1>
        <p className="subtitle">
          A better way to compose mood-accurate songs from images using joint embedding spaces
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
                {/* eslint-disable-next-line @next/next/no-img-element -- local backend-served URL, not a remote image */}
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
        <div className="genre-knob-trigger">
          <button
            type="button"
            onClick={() => setGenreModalOpen(true)}
            disabled={runActive || busy}
            className="btn-sm"
          >
            Genre Knob
          </button>
          {genre && <p className="note">{genre}</p>}
        </div>

        <h2 className="section-title">Upload your own</h2>
        <p className="note">Uses the Suno v5.5 model</p>

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
            onClick={() => startRun(selectedAlgorithm)}
            disabled={!hasImage || runActive || busy}
            className="btn-primary btn-lg btn-run-primary"
          >
            Extract Tags
          </button>

          <select
            value={selectedAlgorithm}
            disabled={runActive || busy || algorithms.length === 0}
            onChange={(event) => setSelectedAlgorithm(event.target.value)}
            className="algorithm-select"
            aria-label="Embedding algorithm"
          >
            {algorithms.map((a) => (
              <option key={a.id} value={a.id}>
                {a.label}
              </option>
            ))}
          </select>

          <div className="toggle-stack">
            <label className="toggle-field">
              <input
                type="checkbox"
                checked={judgesEnabled}
                disabled={runActive || busy}
                onChange={(event) => setJudgesEnabled(event.target.checked)}
              />
              LLM judges
            </label>

            <label className="toggle-field">
              <input
                type="checkbox"
                checked={embeddingEnabled}
                disabled={runActive || busy}
                onChange={(event) => setEmbeddingEnabled(event.target.checked)}
              />
              Embedding space visual
            </label>
          </div>
        </div>


        {error && <p className="status-error">{error}</p>}
        {notice && <p className="status-notice">{notice}</p>}
      </div>

      {run && (
        <div className="runs">
          <RunRow
            algorithmLabel={algorithmLabels[run.algorithm] ?? run.algorithm}
            previewUrl={previewUrl}
            result={run.result}
            onGenerate={handleGenerate}
          />
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

      <AboutSection algorithms={algorithms} />

      {genreModalOpen && (
        <div
          className="modal-backdrop"
          onMouseDown={(event) => {
            genreBackdropMouseDownOnSelf.current = event.target === event.currentTarget;
          }}
          onClick={() => {
            // Only closes when the mousedown that started this click also
            // landed on the backdrop itself -- otherwise a text-selection
            // drag that starts inside the panel and ends outside it would
            // close the modal, since a click's target is computed from
            // where the mouse went up, not where the drag began.
            if (genreBackdropMouseDownOnSelf.current) setGenreModalOpen(false);
          }}
        >
          <div
            ref={genreModalRef}
            className="genre-card"
            role="dialog"
            aria-modal="true"
            aria-label="Genre Knob"
            tabIndex={-1}
            onClick={(event) => event.stopPropagation()}
          >
            <h2 className="section-title">
              Genre Knob <span className="meta-label">Optional</span>
              <button
                type="button"
                onClick={() => setGenreModalOpen(false)}
                aria-label="Close"
                className="modal-close"
              >
                ✕
              </button>
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
        </div>
      )}
    </main>
  );
}
