"use client";

import { useEffect, useMemo, useState } from "react";

import { AboutSection, RunCard } from "./components";
import GenreKnobModal from "./GenreKnobModal";
import {
  type AlgorithmInfo,
  API_URL,
  type Analysis,
  apiRequest,
  type Card,
  cardFromAnalysis,
  cardFromRecord,
  generateUrl,
  isCardBusy,
  MAX_CARDS,
  type SessionRecord,
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
  // before a run starts -- the toggle disables once starting a new run
  // isn't allowed (cap reached, or something else in flight), since it
  // can't retroactively affect tagging that already happened.
  const [judgesEnabled, setJudgesEnabled] = useState(false);

  // Whether /tag computes the joint embedding space plot for the run about
  // to start. Off by default; off skips `embedding_projection` entirely on
  // the backend (see main.py's `show_embedding` param), not just its
  // display. Same disable convention as judgesEnabled.
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

  // The session's runs, newest first -- each independently at whatever
  // stage it's reached (tagging in flight, tags extracted, or fully
  // generated). Hydrated from GET /history on mount so a reload restores
  // every run that had already reached /tag (see the effect below).
  const [cards, setCards] = useState<Card[]>([]);
  // Which cards the user has explicitly toggled open/closed, overriding the
  // newest-open/rest-collapsed default (see RunCard's `open` formula).
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // Only one network operation (a tag extraction or a generation) runs at a
  // time across the whole session -- many settled cards can coexist, but
  // starting a new one waits for whatever's already in flight.
  const busy = cards.some((card) => isCardBusy(card.result));
  // Cards whose /tag call failed never reached the backend's SESSIONS list
  // (see dropCardLocally below), so they shouldn't count against the cap the
  // backend actually enforces.
  const atCap =
    cards.filter((card) => card.result.status !== "error").length >= MAX_CARDS;
  // Gates every "start a new run" control (sample picker, file input,
  // Extract Tags, algorithm/genre/judge/embedding pickers) -- not "a run
  // exists" the way the old single-run model gated them.
  const newRunDisabled = atCap || busy;

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

    // Rehydrates every run this session already reached /tag for -- as
    // long as the backend process itself hasn't restarted since (the
    // in-memory tradeoff; see main.py's SESSIONS). A fresh backend process
    // just returns an empty list, so this is a no-op on a first run.
    fetch(`${API_URL}/history`)
      .then((response) => response.json())
      .then((records: SessionRecord[]) => setCards(records.map(cardFromRecord)))
      .catch(offline("Could not load run history."));
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
  // the Extract Tags button (on top of newRunDisabled).
  const hasImage = Boolean(file || selectedSample);

  // Just records which sample was picked -- no bytes fetched client-side.
  // /tag re-reads them from disk server-side (see main.py's `sample_image`
  // param), so this never needs the browser to GET the image cross-origin.
  function selectSampleImage(name: string) {
    if (newRunDisabled) return;
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
      setCards([]);
      setExpanded({});
      setQuery("");
      setNotice(
        `Deleted ${data.files_removed} files, ${data.megabytes_freed} MB freed.`,
      );
    } catch (err) {
      // A failed wipe leaves things in place rather than silently resetting
      // the UI while stale files remain on the backend.
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  // FormData is what makes the file upload. Note /tag never sets a
  // Content-Type header itself: the browser has to set that so it can
  // include the multipart boundary marker.
  //
  // A placeholder card goes in immediately (pinned open by RunCard while
  // status is "loading") so the in-flight run is visible in the list right
  // away. For a sample image its imageSource is already knowable
  // client-side, so RunCard can derive its preview URL immediately; for an
  // upload imageSource stays null until /tag responds -- deliberately no
  // client-side object URL standing in for a thumbnail in the meantime --
  // and the real, backend-served photo appears the moment /tag actually
  // succeeds and replaces this entry.
  //
  // The placeholder's id is generated fresh per call (never a shared
  // constant): if /tag fails, the placeholder stays in the list as an
  // error card (see handleDeleteCard -- it never had a backend record, so
  // it's never in play for /generate or a per-run DELETE), and starting
  // another run afterwards must not collide with it or with any earlier
  // failed attempt still sitting in the list.
  function startRun(algorithm: string) {
    if (!hasImage || newRunDisabled) return;

    setError(null);
    setNotice(null);
    const pendingId = `pending-${crypto.randomUUID()}`;
    setCards((prev) => [
      {
        id: pendingId,
        algorithm,
        createdAt: new Date().toISOString(),
        imageSource: selectedSample ? { kind: "sample", name: selectedSample } : null,
        result: { status: "loading" },
      },
      ...prev,
    ]);

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
      .then((data: Analysis) => {
        const card = cardFromAnalysis(data);
        setCards((prev) => prev.map((c) => (c.id === pendingId ? card : c)));
      })
      .catch((err) => {
        const message = err instanceof Error ? err.message : String(err);
        setCards((prev) =>
          prev.map((c) =>
            c.id === pendingId ? { ...c, result: { status: "error", error: message } } : c,
          ),
        );
      });
  }

  async function handleGenerate(cardId: string) {
    const card = cards.find((c) => c.id === cardId);
    if (!card || card.result.status !== "ready" || busy) return;

    setCards((prev) =>
      prev.map((c) =>
        c.id === cardId
          ? { ...c, result: { ...c.result, generating: true, generateError: undefined } }
          : c,
      ),
    );

    try {
      const data = await apiRequest(
        generateUrl(cardId),
        { method: "POST" },
        "Could not generate a song.",
      );
      setCards((prev) =>
        prev.map((c) =>
          c.id === cardId ? { ...c, result: { ...c.result, generating: false, generated: data } } : c,
        ),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setCards((prev) =>
        prev.map((c) =>
          c.id === cardId
            ? { ...c, result: { ...c.result, generating: false, generateError: message } }
            : c,
        ),
      );
    }
  }

  function dropCardLocally(cardId: string) {
    setCards((prev) => prev.filter((c) => c.id !== cardId));
    setExpanded((prev) => {
      if (!(cardId in prev)) return prev;
      const next = { ...prev };
      delete next[cardId];
      return next;
    });
  }

  async function handleDeleteCard(cardId: string) {
    const card = cards.find((c) => c.id === cardId);
    // A card whose own /tag call failed never made it into the backend's
    // session list -- there's nothing there to delete, and calling the
    // endpoint anyway would just 404. Drop it locally instead, same as any
    // other never-persisted piece of UI state.
    if (card?.result.status === "error") {
      dropCardLocally(cardId);
      return;
    }

    try {
      await apiRequest(
        `${API_URL}/history/${cardId}`,
        { method: "DELETE" },
        "Could not delete this run.",
      );
      dropCardLocally(cardId);
    } catch (err) {
      // Same convention as handleClear: a failed delete leaves the card in
      // place rather than silently dropping it while its files remain.
      setError(err instanceof Error ? err.message : String(err));
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
                disabled={newRunDisabled}
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
            disabled={newRunDisabled}
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
          disabled={newRunDisabled}
          className="file-picker"
        />

        <div className="run-buttons">
          <button
            onClick={() => startRun(selectedAlgorithm)}
            disabled={!hasImage || newRunDisabled}
            className="btn-primary btn-lg btn-run-primary"
          >
            Extract Tags
          </button>

          <select
            value={selectedAlgorithm}
            disabled={newRunDisabled || algorithms.length === 0}
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
                disabled={newRunDisabled}
                onChange={(event) => setJudgesEnabled(event.target.checked)}
              />
              LLM judges
            </label>

            <label className="toggle-field">
              <input
                type="checkbox"
                checked={embeddingEnabled}
                disabled={newRunDisabled}
                onChange={(event) => setEmbeddingEnabled(event.target.checked)}
              />
              Embedding space visual
            </label>
          </div>
        </div>

        {atCap && (
          <p className="status-notice">
            This session already holds {MAX_CARDS} runs. Delete one below (or Clear) before starting another.
          </p>
        )}

        {error && <p className="status-error">{error}</p>}
        {notice && <p className="status-notice">{notice}</p>}
      </div>

      {cards.length > 0 && (
        <div className="runs">
          {cards.map((card, index) => (
            <RunCard
              key={card.id}
              card={card}
              position={index + 1}
              algorithmLabel={algorithmLabels[card.algorithm] ?? card.algorithm}
              isNewest={index === 0}
              expandedOverride={expanded[card.id]}
              onToggle={(currentlyOpen) =>
                setExpanded((prev) => ({ ...prev, [card.id]: !currentlyOpen }))
              }
              onGenerate={() => handleGenerate(card.id)}
              onDelete={() => handleDeleteCard(card.id)}
              generateBusy={busy}
              deleteDisabled={isCardBusy(card.result)}
            />
          ))}
        </div>
      )}

      <div className="danger-zone">
        <button
          onClick={handleClear}
          disabled={(cards.length === 0 && !hasImage) || busy}
          className={`btn-lg btn-danger${armed ? " is-armed" : ""}`}
        >
          {armed ? "Clear everything? Click again" : "Clear"}
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
        disabled={newRunDisabled}
      />
    </main>
  );
}
