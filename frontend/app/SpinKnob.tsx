"use client";

import { useRef, useState } from "react";

// How far the knob turns per option. 12° means a full turn covers 30 options, which
// is fast enough to browse without being twitchy.
const DEGREES_PER_OPTION = 12;

type Props = {
  options: string[];
  value: string;
  onChange: (option: string) => void;
  label: string;
  disabled?: boolean;
};

// A drag/keyboard-driven picker over a flat list of string options, originally
// built for the genre picker and generalized to drive the algorithm selector too
// -- both need the same "spin to browse, snap to the nearest option" behavior,
// just over a different list and with independent state.
export default function SpinKnob({ options, value, onChange, label, disabled }: Props) {
  const [rotation, setRotation] = useState(0);
  const [seenValue, setSeenValue] = useState(value);
  const knob = useRef<HTMLDivElement>(null);
  // Angle of the pointer last time we saw it; null when not dragging.
  const lastAngle = useRef<number | null>(null);

  const indexAt = (degrees: number) => {
    const count = options.length;
    if (count === 0) return 0;
    // Modulo twice so negative rotations wrap round to the end of the list rather
    // than producing a negative index.
    const step = Math.round(degrees / DEGREES_PER_OPTION) % count;
    return (step + count) % count;
  };

  // Snap round when something else (search, dice, clear) changed the value. This
  // adjusts state during render rather than in an effect, which is React's
  // recommended shape for "derive from a prop that changed" and avoids the extra
  // paint an effect would cause. The guard means it never fights a live drag,
  // because during a drag the knob and the value already agree.
  if (value !== seenValue) {
    setSeenValue(value);
    const target = options.indexOf(value);
    if (target >= 0 && options[indexAt(rotation)] !== value) {
      setRotation(target * DEGREES_PER_OPTION);
    }
  }

  function angleOfPointer(clientX: number, clientY: number) {
    const box = knob.current!.getBoundingClientRect();
    const centerX = box.left + box.width / 2;
    const centerY = box.top + box.height / 2;
    return (Math.atan2(clientY - centerY, clientX - centerX) * 180) / Math.PI;
  }

  function turnTo(degrees: number) {
    setRotation(degrees);
    if (options.length > 0) onChange(options[indexAt(degrees)]);
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

  // Arrow keys step one option, so the knob is usable without a mouse.
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
    turnTo(rotation + step * DEGREES_PER_OPTION);
  }

  return (
    <div
      ref={knob}
      role="slider"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={Math.max(options.length - 1, 0)}
      aria-valuenow={Math.max(options.indexOf(value), 0)}
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
        border: "1px solid var(--line-strong)",
        background: "var(--paper-dim)",
        boxShadow: "var(--shadow-sm)",
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
          background: "var(--accent)",
          borderRadius: 2,
          // 54px is the distance from the marker's own top to the knob's center.
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
