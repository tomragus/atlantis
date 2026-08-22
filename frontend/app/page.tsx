"use client";

import { useEffect, useMemo, useState } from "react";

import { AboutSection, RunRow } from "./components";
import GenreKnobModal from "./GenreKnobModal";
import {
  type AlgorithmInfo,
  API_URL,
  type Analysis,
  apiRequest,
  type GeneratedSong,
  type RunResult,
} from "./lib";

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

      <GenreKnobModal
        open={genreModalOpen}
        onClose={() => setGenreModalOpen(false)}
        genres={genres}
        genre={genre}
        query={query}
        onQueryChange={setQuery}
        disabled={busy || runActive}
      />
    </main>
  );
}
