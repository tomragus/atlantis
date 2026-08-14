"use client";

import { useRef, useState } from "react";

// How far the knob turns per genre. 12° means a full turn covers 30 genres, which
// is fast enough to browse without being twitchy.
const DEGREES_PER_GENRE = 12;

type Props = {
  genres: string[];
  value: string;
  onChange: (genre: string) => void;
  disabled?: boolean;
};

export default function GenreKnob({ genres, value, onChange, disabled }: Props) {
  const [rotation, setRotation] = useState(0);
  const [seenValue, setSeenValue] = useState(value);
  const knob = useRef<HTMLDivElement>(null);
  // Angle of the pointer last time we saw it; null when not dragging.
  const lastAngle = useRef<number | null>(null);

  const indexAt = (degrees: number) => {
    const count = genres.length;
    if (count === 0) return 0;
    // Modulo twice so negative rotations wrap round to the end of the list rather
    // than producing a negative index.
    const step = Math.round(degrees / DEGREES_PER_GENRE) % count;
    return (step + count) % count;
  };

  // Snap round when something else (search, dice, clear) changed the genre. This
  // adjusts state during render rather than in an effect, which is React's
  // recommended shape for "derive from a prop that changed" and avoids the extra
  // paint an effect would cause. The guard means it never fights a live drag,
  // because during a drag the knob and the value already agree.
  if (value !== seenValue) {
    setSeenValue(value);
    const target = genres.indexOf(value);
    if (target >= 0 && genres[indexAt(rotation)] !== value) {
      setRotation(target * DEGREES_PER_GENRE);
    }
  }

  function angleOfPointer(clientX: number, clientY: number) {
    const box = knob.current!.getBoundingClientRect();
    const centreX = box.left + box.width / 2;
    const centreY = box.top + box.height / 2;
    return (Math.atan2(clientY - centreY, clientX - centreX) * 180) / Math.PI;
  }

  function turnTo(degrees: number) {
    setRotation(degrees);
    if (genres.length > 0) onChange(genres[indexAt(degrees)]);
  }

  function handlePointerDown(event: React.PointerEvent) {
    if (disabled) return;
    knob.current?.setPointerCapture(event.pointerId);
    lastAngle.current = angleOfPointer(event.clientX, event.clientY);
  }

  function handlePointerMove(event: React.PointerEvent) {
    if (lastAngle.current === null) return;
    const now = angleOfPointer(event.clientX, event.clientY);
    let delta = now - lastAngle.current;
    // atan2 jumps from +180 to -180; without this the knob leaps a half turn.
    if (delta > 180) delta -= 360;
    if (delta < -180) delta += 360;
    lastAngle.current = now;
    turnTo(rotation + delta);
  }

  function handlePointerUp(event: React.PointerEvent) {
    lastAngle.current = null;
    knob.current?.releasePointerCapture(event.pointerId);
  }

  // Arrow keys step one genre, so the knob is usable without a mouse.
  function handleKeyDown(event: React.KeyboardEvent) {
    if (disabled) return;
    const step =
      event.key === "ArrowRight" || event.key === "ArrowUp"
        ? 1
        : event.key === "ArrowLeft" || event.key === "ArrowDown"
          ? -1
          : 0;
    if (step === 0) return;
    event.preventDefault();
    turnTo(rotation + step * DEGREES_PER_GENRE);
  }

  return (
    <div
      ref={knob}
      role="slider"
      aria-label="Genre"
      aria-valuemin={0}
      aria-valuemax={Math.max(genres.length - 1, 0)}
      aria-valuenow={Math.max(genres.indexOf(value), 0)}
      aria-valuetext={value || "none"}
      aria-disabled={disabled}
      tabIndex={disabled ? -1 : 0}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onKeyDown={handleKeyDown}
      style={{
        width: 120,
        height: 120,
        flexShrink: 0,
        borderRadius: "50%",
        border: "2px solid var(--ink)",
        background: "var(--yellow)",
        boxShadow: "var(--lift-sm)",
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        // Without this, dragging on a touchscreen scrolls the page instead.
        touchAction: "none",
        cursor: disabled ? "default" : "grab",
        opacity: disabled ? 0.5 : 1,
        userSelect: "none",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 6,
          left: "50%",
          width: 4,
          height: 18,
          background: "var(--orange-deep)",
          borderRadius: 2,
          // 54px is the distance from the marker's own top to the knob's centre.
          transformOrigin: "50% 54px",
          transform: `translateX(-50%) rotate(${rotation}deg)`,
        }}
      />
      <span
        style={{
          fontSize: "0.75rem",
          textAlign: "center",
          padding: "0 1.1rem",
          lineHeight: 1.25,
          pointerEvents: "none",
        }}
      >
        {value || "—"}
      </span>
    </div>
  );
}
