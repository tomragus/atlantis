// Shared types, constants, and small pure helpers for the app -- everything
// non-visual that page.tsx and components.tsx both need. Kept in one file
// since none of it is large enough on its own to earn a separate module.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

// Matches backend/main.py's MAX_SESSIONS -- the two aren't shared code, so
// this has to be kept in sync by hand if the cap ever changes.
export const MAX_CARDS = 20;

export type Track = {
  audioFile: string;
};

export type FacetTag = {
  tag: string;
  facet: string;
  score: number;
};

export type JudgeVerdict = {
  tag: string;
  flagged: boolean;
  reason: string;
};

// A cross-facet mood contradiction the second judge found (e.g. "joyful" from
// valence next to "haunting" from mood) -- `drop` names which of the pair got
// dropped from the tag list entirely (no replacement is picked).
export type Contradiction = {
  tag_a: string;
  tag_b: string;
  drop: string;
  reason: string;
};

// What GET /algorithms returns: the embedding algorithms the picker offers, each
// with a display label and a one-line blurb (surfaced in the algorithm-selector
// card and used to keep score-meaning copy algorithm-aware).
export type AlgorithmInfo = {
  id: string;
  label: string;
  blurb: string;
};

export type EmbeddingPoint = {
  tag: string;
  facet: string;
  x: number;
  y: number;
};

// A 2D PCA projection of the final tags' vectors in the chosen algorithm's
// embedding space, with the image vector projected onto that same basis --
// illustrative only, see `embedding_projection` in
// backend/embedding_tagger.py for the caveats.
export type Embedding = {
  model: string;
  source_dimensions: number;
  image: { x: number; y: number };
  points: EmbeddingPoint[];
};

// How the final tags were arrived at: the first judge's per-tag verdicts,
// the second judge's cross-facet contradiction pairs, and the final tags --
// each labeled with its facet. A tag either judge flags is dropped outright,
// not replaced.
export type Tagging = {
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

// Where a run's image came from, and how to serve it back -- mirrors
// backend/main.py's session record. A sample image is already servable by
// name from /test-image-files regardless of session; an uploaded image's
// bytes are persisted server-side under IMAGES_DIR as soon as /tag
// succeeds, so `path` (relative to the backend's /files mount) always
// resolves to the real photo, reload or not -- no placeholder case.
export type ImageSource =
  | { kind: "sample"; name: string }
  | { kind: "upload"; filename: string; path: string };

// What `POST /tag` returns: the tagging half of the pipeline only, no Suno
// call yet. Held in state until the user commits with "Generate Songs", at
// which point only its `id` is sent back to `POST /generate` (the backend
// looks the rest up off its own session record) so the judge never has to
// run a second time for the same image.
export type Analysis = {
  id: string;
  tags: string[];
  tag_scores: Record<string, number>;
  tagging: Tagging;
  prompt: string;
  algorithm: string;
  image_source: ImageSource;
};

// What `POST /generate` returns: Suno renders two takes per job and both are
// kept -- they cost nothing extra, and one is usually the better listen.
// `job_id` is optional because a card rehydrated from `GET /history` has
// `tracks` (the session record keeps those) but no `job_id` -- the backend
// never stores the Suno job id past the request that made it, and nothing
// in the frontend reads `job_id` today.
export type GeneratedSong = {
  job_id?: string;
  tracks: Track[];
};

// The current run's tagging/generation state, for whichever algorithm the
// dropdown had selected when "Extract Tags" was clicked.
export type RunResult = {
  status: "loading" | "ready" | "error";
  analysis?: Analysis;
  error?: string;
  generating?: boolean;
  generated?: GeneratedSong;
  generateError?: string;
};

// Whether a card has async work of its own in flight -- still tagging, or
// mid-generate. Gates things like the delete button and the "pin this card
// open" rule, each of which needs this exact check.
export function isCardBusy(result: RunResult): boolean {
  return result.status === "loading" || Boolean(result.generating);
}

// One entry in the session's list of up to 20 runs -- the frontend's shape
// for a backend session record (GET /history's items, and what a `/tag`
// success gets folded into). `result` is a `RunResult` reused as-is: it
// already models one card's in-flight/settled state (loading/ready/error,
// then optionally generating/generated/generateError) whether the card
// arrived fresh from `/tag` this session or was rehydrated from history.
//
// `imageSource` is nullable only for the brief window between starting a run
// and `/tag` actually responding for an *uploaded* image: the backend hasn't
// assigned the run its id (and therefore its persisted image path) yet, and
// deliberately no client-side object URL stands in for it in the meantime
// (see startRun in page.tsx) -- so there's simply no thumbnail to show yet,
// never a placeholder one. A sample image needs no such gap; it's already
// knowable client-side. Once `/tag` resolves, it's always set (that's the
// "no placeholder case remains" guarantee) for the rest of the card's life,
// reload included. The preview URL itself is never stored -- it's always
// derived on demand via `previewUrlFor`, so there's nothing to keep in sync.
export type Card = {
  id: string;
  algorithm: string;
  createdAt: string;
  imageSource: ImageSource | null;
  result: RunResult;
};

// Builds a card's image URL straight off its `ImageSource` -- no
// client-side object-URL lifecycle needed even for a card created in the
// current tab, since the backend always has a servable copy the moment
// `/tag` returns (an upload's bytes are persisted under IMAGES_DIR; a
// sample was already servable by name).
export function previewUrlFor(imageSource: ImageSource): string {
  return imageSource.kind === "upload"
    ? fileUrl(imageSource.path)
    : `${API_URL}/test-image-files/${encodeURIComponent(imageSource.name)}`;
}

// What `GET /history` returns per entry -- the backend's own session record
// (main.py's SESSIONS), before it's reshaped into a `Card`. Same shape as an
// `Analysis` (id/tags/tag_scores/tagging/prompt/algorithm/image_source) plus
// the two fields only the persisted record carries.
export type SessionRecord = Analysis & {
  created_at: string;
  tracks: Track[] | null;
};

// Reshapes one history record into a Card -- used to hydrate `cards` from
// `GET /history` on mount, so a reload restores every run that had already
// reached `/tag`, tracks included where a song was already generated.
export function cardFromRecord(record: SessionRecord): Card {
  return {
    id: record.id,
    algorithm: record.algorithm,
    createdAt: record.created_at,
    imageSource: record.image_source,
    result: {
      status: "ready",
      analysis: record,
      generated: record.tracks ? { tracks: record.tracks } : undefined,
    },
  };
}

// Builds a fresh card straight from a `/tag` success response -- no
// `tracks` yet (nothing has been generated for it). `createdAt` is set to
// now rather than echoed from the backend: `/tag`'s response doesn't carry
// a timestamp (only the session record kept server-side does).
export function cardFromAnalysis(analysis: Analysis): Card {
  return {
    id: analysis.id,
    algorithm: analysis.algorithm,
    createdAt: new Date().toISOString(),
    imageSource: analysis.image_source,
    result: { status: "ready", analysis },
  };
}

// `POST /generate`'s whole request now: just the run id to fill in tracks
// for -- the backend looks the rest (prompt, algorithm, image source) up
// off its own session record, so there's no FormData/body to build here
// the way the old tags/tag_scores/tagging/prompt echo needed.
export const generateUrl = (runId: string) =>
  `${API_URL}/generate?run_id=${encodeURIComponent(runId)}`;

// Which of a judge pass's verdicts actually get shown as a dropped tag, and
// what to say about each -- shared data-shaping for JudgeReviewSection.
export type JudgeRow = { key: string; flaggedTag: string; reason: string };

// Fixed hue order for the 8 facets, never reassigned/cycled per-render --
// matches backend/vocabulary.py's ADJECTIVES order. Dark-surface steps, since
// the embedding plot sits on a dark panel; identity is confirmed on hover
// (the tooltip names the exact tag and facet), so color here is a grouping
// cue rather than the only way to tell two points apart.
export const FACET_ORDER = [
  "color",
  "light",
  "mood",
  "valence",
  "arousal",
  "texture",
  "energy",
  "style",
];

export const FACET_COLORS: Record<string, string> = {
  color: "#3987e5",
  light: "#d95926",
  mood: "#199e70",
  valence: "#c98500",
  arousal: "#d55181",
  texture: "#008300",
  energy: "#9085e9",
  style: "#e66767",
};

// The HF checkpoint string per algorithm id -- same ids/strings as
// backend/embedding_tagger.py's HF_MODEL_IDS. Only the id needs hardcoding
// here; the display label comes from the already-fetched `algorithms` prop
// (GET /algorithms, i.e. backend/algorithms.py's ALGORITHM_INFO) rather than
// a second hand-typed copy that could drift from it.
export const CHECKPOINT_IDS: Record<string, string> = {
  siglip: "google/siglip2-base-patch16-224",
  clip: "openai/clip-vit-base-patch32",
  align: "kakaobrain/align-base",
  metaclip: "facebook/metaclip-b32-400m",
};

export const PLOT_WIDTH = 520;
export const PLOT_HEIGHT = 300;
export const PLOT_MARGIN = 30;
export const TOOLTIP_HEIGHT = 34;

// Downloaded MP3s are served by the backend out of its (non-history) audio
// folder. Filenames contain spaces, `&` and apostrophes, so each path
// segment is escaped separately -- encoding the whole path would eat the
// `/` too.
export const fileUrl = (path: string) =>
  `${API_URL}/files/${path.split("/").map(encodeURIComponent).join("/")}`;

// Shared by every mutating call (wipe run, extract tags, generate song):
// fetch, parse JSON, and normalize both an API error (the response body's
// `detail`) and a network failure into one `Error` with a message fit to
// show the user -- so each call site's catch block is just `setError`.
export async function apiRequest(
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
export function formatScore(score: number): string {
  if (!Number.isFinite(score) || score === 0) return "0.000";
  if (Math.abs(score) < 0.001) return score.toExponential(2);
  return score.toFixed(3);
}

export function clamp(value: number, low: number, high: number) {
  return Math.min(Math.max(value, low), high);
}

// Rough monospace-ish width estimate so the tooltip box fits its text without
// measuring the DOM -- fine for a short "tag · score" string. Tuned for the
// embedding tooltip's 16px font.
export function estimateTextWidth(text: string) {
  return text.length * 9.6 + 22;
}

// Which of a judge pass's verdicts actually get shown as a dropped tag, and
// what to say about each.
export function flaggedJudgeRows(judge: JudgeVerdict[]): JudgeRow[] {
  return judge
    .filter((v) => v.flagged)
    .map((v) => ({ key: v.tag, flaggedTag: v.tag, reason: v.reason }));
}

export function contradictionRows(contradictions: Contradiction[]): JudgeRow[] {
  return contradictions.map((c, index) => {
    const kept = c.drop === c.tag_a ? c.tag_b : c.tag_a;
    return {
      key: `${c.tag_a}-${c.tag_b}-${index}`,
      flaggedTag: c.drop,
      reason: `contradicted “${kept}” -- ${c.reason}`,
    };
  });
}
