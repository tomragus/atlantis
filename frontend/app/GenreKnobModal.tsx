"use client";

import { useEffect, useRef } from "react";

import SpinKnob from "./SpinKnob";

// The Genre Knob modal: spin/search/roll a genre, or leave it blank. Owns its
// own focus trap and backdrop-click handling -- Home only needs to know
// whether it's open and hand it the genre state to read and write.
export default function GenreKnobModal({
  open,
  onClose,
  genres,
  genre,
  query,
  onQueryChange,
  disabled,
}: {
  open: boolean;
  onClose: () => void;
  genres: string[];
  genre: string;
  query: string;
  onQueryChange: (value: string) => void;
  disabled: boolean;
}) {
  // The modal panel itself, so the focus-trap effect below can find its
  // focusable children and the backdrop's mousedown handler can tell "the
  // drag started on the backdrop" apart from "it started inside the panel
  // and was dragged out" (a plain onClick can't tell those apart, since a
  // drag that ends outside the panel still fires its click on the backdrop).
  const panelRef = useRef<HTMLDivElement>(null);
  const backdropMouseDownOnSelf = useRef(false);

  // Escape closes the modal, same as its X button/backdrop click. Tab/
  // Shift+Tab are trapped within the panel so keyboard focus can't reach
  // (and activate) the "Extract Tags" button or algorithm dropdown sitting
  // behind the backdrop while the modal is still open on screen.
  useEffect(() => {
    if (!open) return;
    const panel = panelRef.current;
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
        onClose();
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
  }, [open, onClose]);

  if (!open) return null;

  function rollRandomGenre() {
    if (genres.length > 0) {
      onQueryChange(genres[Math.floor(Math.random() * genres.length)]);
    }
  }

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        backdropMouseDownOnSelf.current = event.target === event.currentTarget;
      }}
      onClick={() => {
        // Only closes when the mousedown that started this click also
        // landed on the backdrop itself -- otherwise a text-selection
        // drag that starts inside the panel and ends outside it would
        // close the modal, since a click's target is computed from
        // where the mouse went up, not where the drag began.
        if (backdropMouseDownOnSelf.current) onClose();
      }}
    >
      <div
        ref={panelRef}
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
            onClick={onClose}
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
            onChange={onQueryChange}
            label="Genre"
            disabled={disabled}
          />

          <div className="genre-fields">
            <label htmlFor="genre-search" className="genre-label">
              Genre
            </label>

            <input
              id="genre-search"
              list="genre-options"
              value={query}
              disabled={disabled}
              placeholder="spin, search or roll"
              onChange={(event) => onQueryChange(event.target.value)}
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
                disabled={disabled || genres.length === 0}
                className="btn-sm"
              >
                🎲 Random
              </button>
              <button
                type="button"
                onClick={() => onQueryChange("")}
                disabled={disabled || !query}
                className="btn-sm"
              >
                Clear
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
