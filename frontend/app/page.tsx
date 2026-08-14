"use client";

import { useEffect, useState } from "react";

import GenreKnob from "./GenreKnob";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Track = {
  title: string;
  duration: number;
  audioUrl: string;
  audioFile: string;
};

// Suno renders two takes of the same prompt per generation, and both are kept.
type Generation = {
  timestamp: string;
  image: string;
  genre: string;
  tags: string[];
  // SigLIP's confidence per tag, keyed by the tag itself.
  tag_scores: Record<string, number>;
  prompt: string;
  job_id: string;
  tracks: Track[];
};

// Saved images and MP3s are served by the backend out of its history folder.
// Audio filenames contain spaces, `&` and apostrophes, so each path segment is
// escaped separately -- encoding the whole path would eat the `/` too.
const fileUrl = (path: string) =>
  `${API_URL}/files/${path.split("/").map(encodeURIComponent).join("/")}`;

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [history, setHistory] = useState<Generation[]>([]);
  const [loading, setLoading] = useState(false);
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

  async function handleWipe() {
    if (!armed) {
      setArmed(true);
      return;
    }

    setArmed(false);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/history`, { method: "DELETE" });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Could not clear the history.");
      }
      setHistory([]);
      setNotice(
        `Deleted ${data.entries_removed} runs, ${data.files_removed} files, ` +
          `${data.megabytes_freed} MB freed.`,
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not reach the server.",
      );
    }
  }

  function rollRandomGenre() {
    if (genres.length > 0) {
      setQuery(genres[Math.floor(Math.random() * genres.length)]);
    }
  }

  async function handleGenerate() {
    if (!file) return;

    setLoading(true);
    setError(null);
    setNotice(null);

    // FormData is what makes this a file upload. Note we never set a
    // Content-Type header: the browser has to set that itself so it can
    // include the multipart boundary marker.
    const body = new FormData();
    body.append("image", file);

    const genreParam = genre ? `?genre=${encodeURIComponent(genre)}` : "";

    try {
      const response = await fetch(`${API_URL}/generate${genreParam}`, {
        method: "POST",
        body,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Could not generate a song.");
      }
      setHistory((previous) => [data, ...previous]);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not reach the server.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <header className="hero">
        <h1 className="title">
          Image to Song
          <span className="title-hollow">Studio</span>
        </h1>
        <p className="subtitle">Turn a picture into a piece of music.</p>

        {/* The strip is duplicated because the scroll loops at -50%: the
            second copy is what is on screen while the first wraps around. */}
        <div className="marquee">
          <div className="marquee-track">
            {[0, 1].map((copy) => (
              <span key={copy}>
                image → tags → song ✱ all hail siglip ✱ image → tags → song
                ✱ all hail siglip ✱{" "}
              </span>
            ))}
          </div>
        </div>
      </header>

      <div className="rail">
        {apiDown && (
          <div className="status-error">
            Can’t reach the backend at {API_URL}. Is `uvicorn main:app --reload`
            running?
          </div>
        )}

        <input
          type="file"
          accept="image/*"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          disabled={loading}
          className="file-picker"
        />

        <section className="genre">
          <GenreKnob
            genres={genres}
            value={genre}
            onChange={setQuery}
            disabled={loading}
          />

          <div className="genre-fields">
            <label htmlFor="genre-search" className="genre-label">
              Genre <span className="meta-label">(optional)</span>
            </label>

            <input
              id="genre-search"
              list="genre-options"
              value={query}
              disabled={loading}
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
                disabled={loading || genres.length === 0}
                className="btn-sm"
              >
                🎲 Random
              </button>
              <button
                type="button"
                onClick={() => setQuery("")}
                disabled={loading || !query}
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
          onClick={handleGenerate}
          disabled={!file || loading}
          className="btn-primary btn-lg"
        >
          {loading ? "Composing…" : "Generate"}
        </button>

        {loading && (
          <p className="status-notice">
            This takes about two minutes. Keep this tab open.
          </p>
        )}

        {error && <p className="status-error">{error}</p>}
        {notice && <p className="status-notice">{notice}</p>}

        {history.length > 0 && (
          <div className="danger-zone">
            <button
              onClick={handleWipe}
              disabled={loading}
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

      <div className="stream">
        {history.length === 0 && !apiDown && (
          <p className="stream-empty">
            Nothing here yet. Pick an image on the left and hit Generate.
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
                    {entry.tags.length > 0 && (
                      // One tag set drove both takes, so it sits above them
                      // rather than being repeated. Strongest SigLIP match first.
                      <div className="tags">
                        {entry.tags.map((tag) => {
                          const score = entry.tag_scores?.[tag];
                          return (
                            <span key={tag} className="tag">
                              {tag}
                              {score !== undefined && (
                                <span className="tag-score">
                                  {score.toFixed(3)}
                                </span>
                              )}
                            </span>
                          );
                        })}
                      </div>
                    )}

                    {entry.prompt && (
                      <div className="meta prompt-line">
                        <span className="meta-label">prompt sent to Suno: </span>
                        <span className="mono">{entry.prompt}</span>
                      </div>
                    )}

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
