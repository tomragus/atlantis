// The presentational pieces of a single run's output: the embedding-space
// plot, the tag chips, the two judges' findings, and the row that lays all
// of that out next to the run's image. Grouped in one file since each is
// small and they're only ever composed together (RunRow uses all the
// others), plus the standalone "How it works" write-up.

import { memo, useMemo, useState, type KeyboardEvent } from "react";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

import {
  type AlgorithmInfo,
  type Card,
  CHECKPOINT_IDS,
  type Embedding,
  type FacetTag,
  FACET_COLORS,
  FACET_ORDER,
  formatScore,
  flaggedJudgeRows,
  contradictionRows,
  isCardBusy,
  PLOT_HEIGHT,
  PLOT_MARGIN,
  PLOT_WIDTH,
  previewUrlFor,
  TOOLTIP_HEIGHT,
  type RunResult,
  type Tagging,
  clamp,
  estimateTextWidth,
} from "./lib";

// A 2D scatter of the final tags, color-coded by facet, on its own dark
// panel. The image's projected position is deliberately not shown here --
// SigLIP/CLIP-family models' image/text "modality gap" means it usually
// lands well outside the tag cluster (see embedding_projection's docstring
// in backend/embedding_tagger.py) and reads as noise rather than signal
// next to the tags. What *is* shown is the frame the tags themselves are
// plotted in: the origin marks the mean of this run's tag vectors (what
// `embedding_projection` centers everything on before projecting), and the
// PC1/PC2 lines through it are the two directions of greatest spread the
// PCA step found among just these tags -- both axis directions and the
// origin are specific to this run's tag set, not a fixed frame shared
// across runs. Points aren't directly labeled (they overlap too easily when
// several near-synonyms land close together) -- hover or focus a point for
// its exact tag instead. Scores aren't shown here -- they're meaningful in
// the full embedding space but not in this PCA-reduced 2D slice of it (see
// the plot's own caption). Color already signals facet, so the
// tooltip doesn't repeat it.
function EmbeddingPlot({
  embedding,
  finalTags,
  algorithmLabel,
}: {
  embedding: Embedding;
  finalTags: FacetTag[];
  algorithmLabel: string;
}) {
  const { points } = embedding;
  const [hovered, setHovered] = useState<string | null>(null);
  const finalTagSet = useMemo(
    () => new Set(finalTags.map((f) => f.tag)),
    [finalTags],
  );

  // Only `hovered` changes on every interaction; the scatter's geometry
  // (bounds, coordinate mapping, which facets are present) only depends on
  // `embedding` itself, so it's wasted work to redo the array scans and
  // rebuild both mapping functions on every hover. `embedding` is a stable
  // reference across those re-renders (it lives in the parent's state, not
  // here), so this only recomputes when the tag analysis actually changes.
  const geometry = useMemo(() => {
    const { points } = embedding;
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

    return {
      toSvgX,
      toSvgY,
      facets: FACET_ORDER.filter((facet) =>
        points.some((p) => p.facet === facet),
      ),
    };
  }, [embedding]);

  if (!geometry) return null;
  const { toSvgX, toSvgY, facets } = geometry;
  // The tag vectors are centered on their own mean before PCA runs (see
  // embedding_projection in backend/embedding_tagger.py), so (0, 0) in data
  // space *is* that mean -- no separate calculation needed to place it.
  const centroidX = toSvgX(0);
  const centroidY = toSvgY(0);

  const hoveredPoint =
    hovered === "mean" ? null : points.find((p) => p.tag === hovered);
  const tooltip = hoveredPoint
    ? {
        cx: toSvgX(hoveredPoint.x),
        cy: toSvgY(hoveredPoint.y),
        text: hoveredPoint.tag,
      }
    : hovered === "mean"
      ? { cx: centroidX, cy: centroidY, text: "(mean)" }
      : null;
  const tooltipWidth = tooltip ? estimateTextWidth(tooltip.text) : 0;
  // Keep the box on-canvas: flip to the left of the point once it would
  // overflow the right edge, and clamp vertically within the plot margins.
  const tooltipX = tooltip
    ? tooltip.cx + 12 + tooltipWidth > PLOT_WIDTH
      ? tooltip.cx - 12 - tooltipWidth
      : tooltip.cx + 12
    : 0;
  const tooltipY = tooltip
    ? clamp(tooltip.cy - 17, 10, PLOT_HEIGHT - TOOLTIP_HEIGHT - 6)
    : 0;

  return (
    <div className="embedding-plot">
      <svg
        viewBox={`0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`}
        role="img"
        aria-label={`2D projection of ${algorithmLabel}'s final tags in embedding space, plotted against PC1/PC2, the two directions this run's tags vary along most. Hover a point for its tag, or the center for the tags' mean.`}
      >
        <line
          x1={PLOT_MARGIN}
          y1={centroidY}
          x2={PLOT_WIDTH - PLOT_MARGIN}
          y2={centroidY}
          className="embedding-axis-line"
        />
        <line
          x1={centroidX}
          y1={PLOT_MARGIN}
          x2={centroidX}
          y2={PLOT_HEIGHT - PLOT_MARGIN}
          className="embedding-axis-line"
        />
        <text
          x={PLOT_WIDTH - PLOT_MARGIN - 4}
          y={centroidY - 6}
          textAnchor="end"
          className="embedding-axis-label"
        >
          PC1
        </text>
        <text
          x={centroidX + 6}
          y={PLOT_MARGIN + 10}
          textAnchor="start"
          className="embedding-axis-label"
        >
          PC2
        </text>

        {points.map((p) => {
          const isFinal = finalTagSet.has(p.tag);
          return (
            <circle
              key={p.tag}
              cx={toSvgX(p.x)}
              cy={toSvgY(p.y)}
              r={isFinal ? 6 : 4}
              fill={FACET_COLORS[p.facet]}
              opacity={isFinal ? 1 : 0.4}
              stroke="var(--card-bg)"
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
          cx={centroidX}
          cy={centroidY}
          r={5}
          fill="var(--card-bg-dim)"
          stroke="var(--card-bg)"
          strokeWidth={1.5}
          tabIndex={0}
          onMouseEnter={() => setHovered("mean")}
          onMouseLeave={() => setHovered(null)}
          onFocus={() => setHovered("mean")}
          onBlur={() => setHovered(null)}
        />

        {tooltip && (
          <g pointerEvents="none">
            <rect
              x={tooltipX}
              y={tooltipY}
              width={tooltipWidth}
              height={TOOLTIP_HEIGHT}
              rx={6}
              className="embedding-tooltip-box"
            />
            <text
              x={tooltipX + tooltipWidth / 2}
              y={tooltipY + TOOLTIP_HEIGHT / 2 + 7}
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
        Each tag’s score corresponds to the distance between the embedded
        image and that embedded tag in the {embedding.source_dimensions}
        -dimensional {algorithmLabel} joint embedding space. Because there is
        no way to view this space directly, we must rely on PCA (Principal
        Component Analysis) to produce a 2D representation of the{" "}
        {embedding.source_dimensions}-dimensional space.
      </p>
      <p className="note">
        Notions of “distance” and “proximity” do not translate neatly in this
        dimensionality reduction process. Instead, PCA decides on two
        characteristics to measure each tag against and uses them as the axes
        around which tags get placed in the new space. No two images will
        have the same axes, and it can be interesting to observe the results
        and try to deduce what PC1 and PC2 are measuring.
      </p>
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

      {tagging.judges_enabled && (
        <>
          <JudgeReviewSection
            title="judge review 1 -- poor-quality tags"
            emptyMessage="Judge #1 found no tags that didn't fit the image's mood this run."
            rows={flaggedJudgeRows(tagging.judge)}
          />

          <JudgeReviewSection
            title="judge review 2 -- contradictions"
            emptyMessage="Judge #2 identified no contradictory tags this run."
            rows={contradictionRows(tagging.contradictions)}
          />
        </>
      )}
    </>
  );
}

// The run's image on the left, its tag breakdown on the right, and (once
// tagging succeeded) its own Generate Songs action and result.
export function RunRow({
  algorithmLabel,
  previewUrl,
  result,
  onGenerate,
  generateBusy = false,
}: {
  algorithmLabel: string;
  previewUrl: string | null;
  result: RunResult;
  onGenerate: () => void;
  // Whether some other card's tag-extraction or generation is in flight --
  // only one network operation runs at a time across the whole session, so
  // this card's own Generate Songs button stays disabled while that's true,
  // on top of `result.generating` covering its own in-flight generation.
  generateBusy?: boolean;
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
              finalTags={result.analysis.tagging.final}
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
                  disabled={result.generating || generateBusy}
                  className="btn-primary"
                >
                  {result.queuePosition !== undefined
                    ? "Queued…"
                    : result.generating
                      ? "Composing…"
                      : "Generate Songs"}
                </button>
              </div>
            )}

            {result.queuePosition !== undefined && (
              <p className="status-notice">
                Queued — {result.queuePosition} ahead of you.
              </p>
            )}

            {result.generating && result.queuePosition === undefined && (
              <p className="status-notice">
                Generating... takes about 2 minutes.
              </p>
            )}

            {result.generateError && (
              <p className="status-error">{result.generateError}</p>
            )}

            {result.generated?.tracks.map((track, index) => (
              <div key={track.url} className="take">
                <div className="take-title">Take {index + 1}</div>
                <audio controls src={track.url} />
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// A single word summarizing where a card has gotten to -- shown in its
// collapsed header, and what drives the pinned-open behavior below.
function cardStatusWord(result: RunResult): string {
  if (result.status === "loading") return "tagging…";
  if (result.status === "error") return "error";
  if (result.queuePosition !== undefined) return "queued…";
  if (result.generating) return "generating…";
  if (result.generated) return "generated";
  return "ready";
}

// One card in the session list: a header (position, thumbnail, algorithm
// label, status word, delete control, caret) that's always visible, and
// (once open) the full RunRow body beneath it. Newest card open by
// default, the rest collapsed; any header click toggles that card only; a
// card with in-flight work of its own stays pinned open regardless of its
// entry in the parent's `expanded` map, the same accordion pattern
// image-to-song-v1 uses (interaction logic only -- Atlantis's own tokens
// style it, see globals.css).
export function RunCard({
  card,
  position,
  algorithmLabel,
  isNewest,
  expandedOverride,
  onToggle,
  onGenerate,
  onDelete,
  generateBusy,
  deleteDisabled,
}: {
  card: Card;
  position: number;
  algorithmLabel: string;
  isNewest: boolean;
  expandedOverride: boolean | undefined;
  onToggle: (currentlyOpen: boolean) => void;
  onGenerate: () => void;
  onDelete: () => void;
  generateBusy: boolean;
  deleteDisabled: boolean;
}) {
  const pinnedOpen = isCardBusy(card.result);
  const open = pinnedOpen || (expandedOverride ?? isNewest);
  const status = cardStatusWord(card.result);
  const previewUrl = card.imageSource ? previewUrlFor(card.imageSource) : null;

  function toggleOnKeyDown(event: KeyboardEvent) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onToggle(open);
    }
  }

  return (
    <div className={`run-card${open ? " is-open" : ""}`}>
      <div className="run-card-head">
        {/* A `div` rather than a `button` -- the delete control below has
            to be a real, separately clickable button in the same row, and
            buttons can't nest. Keyboard-operable via the role/tabIndex/
            onKeyDown trio instead. */}
        <div
          role="button"
          tabIndex={0}
          className="run-card-head-toggle"
          aria-expanded={open}
          onClick={() => onToggle(open)}
          onKeyDown={toggleOnKeyDown}
        >
          <span className="run-card-index">{String(position).padStart(2, "0")}</span>

          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element -- local backend-served/static URL
            <img src={previewUrl} alt="" className="run-card-thumb" />
          ) : (
            <span className="run-card-thumb run-card-thumb-empty" aria-hidden="true" />
          )}

          <span className="run-card-head-text">
            <span className="run-card-name">{algorithmLabel}</span>
            <span className="run-card-status">{status}</span>
          </span>

          <span className="run-card-caret" aria-hidden="true">
            {open ? "▼" : "►"}
          </span>
        </div>

        <button
          type="button"
          className="run-card-delete"
          onClick={onDelete}
          disabled={deleteDisabled}
          aria-label={`Delete run ${position}`}
        >
          Delete
        </button>
      </div>

      {open && (
        <div className="run-card-body">
          <RunRow
            algorithmLabel={algorithmLabel}
            previewUrl={previewUrl}
            result={card.result}
            onGenerate={onGenerate}
            generateBusy={generateBusy}
          />
        </div>
      )}
    </div>
  );
}

// Plain background text -- no card, no border -- explaining the pipeline for
// anyone curious enough to read past the app itself. Only prop is the
// already-fetched algorithm list, for the checkpoint links' labels; memoized
// since this ~170-line static subtree never needs to re-render on Home's
// other state changes (a genre-search keystroke, a knob drag, a run status
// update) once `algorithms` has settled after its one-time fetch.
export const AboutSection = memo(function AboutSection({
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
        gets turned into a 768-dimensional vector which is a rich numerical
        fingerprint capturing color, texture, edges and more. These vectors are
        initially independent (meaning, each one has no knowledge of the others)
        but are then fed into 12 transformer layers in which each vector gets
        iteratively updated based on context from the others.
      </p>

      <p>
        After this process, the 196 tiles are fed into a single pooling
        layer which asks each tile “what do you know about this image?”
        and composes a single 768-dimensional vector
        summarizing the responses.
      </p>

      <p>
        Simultaneously, that same embedding process is happening on a curated
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
        encoder, after which they are stacked and averaged resulting
        in a single 768-dimensional embedding vector per tag.
        After encoding, the cosine similarity between the image vector and each tag vector is calculated
        and fed into a sigmoid (hence the name) with a
        scale <code>k</code> and bias <code>b</code> learned directly from
        SigLIP2’s pre-training, to get a final score between 0 and 1:
      </p>

      <BlockMath math={String.raw`\text{score} = \sigma(k \cdot s + b) = \frac{1}{1 + e^{-(k \cdot s + b)}}`} />

      <p>
        Here’s how to understand this: if you showed the image to the model and asked “does the phrase ‘a
        photo that feels {"{tag}"}’ genuinely describe this image, yes or
        no?”, then the final score would be how confident the model is that
        the answer is yes.
      </p>

      <p>
        Final scores for tags are ranked and categorized by their distinct
        region of “feel” (referred to in the code as “facet”). The top few
        tags per facet (specifically: 2 for “color”, 1 for “light”, 3 for
        “mood”, 1 for “valence”, 1 for “arousal”, 2 for “texture”, 1 for
        “energy”, and 1 for “style”) are used to compose a prompt for the
        Suno v5.5 model, which is simply the list of tags in order from
        highest to lowest score. That prompt is then fed to Suno v5.5 to produce
        two highly mood-accurate songs for that image.
      </p>

      <p>
        Through testing across a diverse image corpus, I have found this
        system capable of accurately capturing a wide range of unique moods
        and aesthetics from pleasant to whimsical to punk. It is really fun
        to play with - please try your own images and see what comes up!
      </p>

      <h3 className="about-heading">Optional LLM-Evaluation</h3>

      <p>
        By default, language models aren’t being used anywhere in the tag
        extraction process. If you’d like, you can turn on
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
        the generated tracks. But the reason I set these to default is because it sometimes feels like there is some magic
        contained within these otherwise seemingly unfitting tags (which
        shine through in the audio) - if you play around with this enough,
        you’ll know exactly what I mean!
      </p>

      <h3 className="about-heading">Other embedding algorithms</h3>

      <p>
        While this project was optimized for and built around the SigLIP2
        algorithm, I also wanted to explore other embedding algorithms to
        see how they compared. Specifically, CLIP (SigLIP’s predecessor and
        the OG joint embedding algorithm from OpenAI), ALIGN from Google (adaptation of
        CLIP that uses EfficientNet for the image encoder and BERT for the
        text encoder) and MetaCLIP from Meta (adaptation of CLIP that
        favors more rare, specific concepts due to how it was trained).
      </p>

      <p>
        From my testing, CLIP and MetaCLIP felt like less refined versions
        of SigLIP, producing tracks that sort of had the right idea but
        ultimately tended to sound somewhat generic. ALIGN would
        occasionally knock it out of the park but I found it to be
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
        You may have noticed the “Genre Knob”. This is a fun addition I came up with that
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
        algorithm captures which is why this is positioned as an extra feature.
        It is really fun to mess around with though, and the quality of the tracks
        produced is a testament to how good the Suno model is.
      </p>

      <h3 className="about-heading">Vocals</h3>

      <p>  
        All prompts sent to Suno have “Instrumental=On”, but the app will occasionally produce songs with
        vocals anyway. Since it’s the same model for generating songs both with and without vocals, the instrumental
        option can’t be a hard toggle, only a strong suggestion.
      </p>

      <h3 className="about-heading">Suno v5.5</h3>

      <p>
        This app uses the Suno v5.5 model for audio generation. All rights to
        the underlying model belong to Suno; no ownership is claimed by this
        app.
      </p>

      <div className="about-attribution">
        <span>&copy; {new Date().getFullYear()} Tom Ragus</span>
        <a
          href="https://buymeacoffee.com/tomragus"
          target="_blank"
          rel="noreferrer"
          aria-label="Buy Tom Ragus a coffee"
          className="about-linkedin"
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
            <path d="M20.216 6.415l-.132-.666c-.119-.598-.388-1.163-1.001-1.379-.197-.069-.42-.098-.57-.241-.152-.143-.196-.366-.231-.572-.065-.378-.125-.756-.192-1.133-.057-.325-.102-.69-.25-.987-.195-.4-.597-.634-.996-.788a5.723 5.723 0 00-.626-.194c-1-.263-2.05-.36-3.077-.416a25.834 25.834 0 00-3.7.062c-.915.083-1.88.184-2.75.5-.318.116-.646.256-.888.501-.297.302-.393.77-.177 1.146.154.267.415.456.692.58.36.162.737.284 1.123.366 1.075.238 2.189.331 3.287.37 1.218.05 2.437.01 3.65-.118.299-.033.598-.073.896-.119.352-.054.578-.513.474-.834-.124-.383-.457-.531-.834-.473-.466.074-.96.108-1.382.146-1.177.08-2.358.082-3.536.006a22.228 22.228 0 01-1.157-.107c-.086-.01-.18-.025-.258-.036-.243-.036-.484-.08-.724-.13-.111-.027-.111-.185 0-.212h.005c.277-.06.557-.108.838-.147h.002c.131-.009.263-.032.394-.048a25.076 25.076 0 013.426-.12c.674.019 1.347.067 2.017.144l.228.031c.267.04.533.088.798.145.392.085.895.113 1.07.542.055.137.08.288.111.431l.319 1.484a.237.237 0 01-.199.284h-.003c-.037.006-.075.01-.112.015a36.704 36.704 0 01-4.743.295 37.059 37.059 0 01-4.699-.304c-.14-.017-.293-.042-.417-.06-.326-.048-.649-.108-.973-.161-.393-.065-.768-.032-1.123.161-.29.16-.527.404-.675.701-.154.316-.199.66-.267 1-.069.34-.176.707-.135 1.056.087.753.613 1.365 1.37 1.502a39.69 39.69 0 0011.343.376.483.483 0 01.535.53l-.071.697-1.018 9.907c-.041.41-.047.832-.125 1.237-.122.637-.553 1.028-1.182 1.171-.577.131-1.165.2-1.756.205-.656.004-1.31-.025-1.966-.022-.699.004-1.556-.06-2.095-.58-.475-.458-.54-1.174-.605-1.793l-.731-7.013-.322-3.094c-.037-.351-.286-.695-.678-.678-.336.015-.718.3-.678.679l.228 2.185.949 9.112c.147 1.344 1.174 2.068 2.446 2.272.742.12 1.503.144 2.257.156.966.016 1.942.053 2.892-.122 1.408-.258 2.465-1.198 2.616-2.657.34-3.332.683-6.663 1.024-9.995l.215-2.087a.484.484 0 01.39-.426c.402-.078.787-.212 1.074-.518.455-.488.546-1.124.385-1.766zm-1.478.772c-.145.137-.363.201-.578.233-2.416.359-4.866.54-7.308.46-1.748-.06-3.477-.254-5.207-.498-.17-.024-.353-.055-.47-.18-.22-.236-.111-.71-.054-.995.052-.26.152-.609.463-.646.484-.057 1.046.148 1.526.22.577.088 1.156.159 1.737.212 2.48.226 5.002.19 7.472-.14.45-.06.899-.13 1.345-.21.399-.072.84-.206 1.08.206.166.281.188.657.162.974a.544.544 0 01-.169.364zm-6.159 3.9c-.862.37-1.84.788-3.109.788a5.884 5.884 0 01-1.569-.217l.877 9.004c.065.78.717 1.38 1.5 1.38 0 0 1.243.065 1.658.065.447 0 1.786-.065 1.786-.065.783 0 1.434-.6 1.499-1.38l.94-9.95a3.996 3.996 0 00-1.322-.238c-.826 0-1.491.284-2.26.613z" />
          </svg>
        </a>
        <a
          href="https://www.linkedin.com/in/tragus/"
          target="_blank"
          rel="noreferrer"
          aria-label="Tom Ragus on LinkedIn"
          className="about-linkedin"
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 1 1 0-4.124 2.062 2.062 0 0 1 0 4.124zM7.114 20.452H3.558V9h3.556v11.452z" />
          </svg>
        </a>
      </div>
    </section>
  );
});
