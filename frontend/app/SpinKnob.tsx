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

// A drag/keyboard-driven picker over a flat list of string options -- "spin to
// browse, snap to the nearest option" -- currently used for the genre picker.
// General enough to reuse for another option list should one need the same
// knob-style interaction later.
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
      className="spin-knob"
      // Only the two properties that actually vary across renders stay
      // inline -- everything constant (size, border, background, layout)
      // lives in the .spin-knob CSS class instead, so a drag's per-pointer-
      // move re-render doesn't reallocate an ~20-property style object for
      // values that never change.
      style={{
        cursor: disabled ? "default" : "grab",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <div
        className="spin-knob-marker"
        style={{
          // 54px is the distance from the marker's own top to the knob's center.
          transformOrigin: "50% 54px",
          transform: `translateX(-50%) rotate(${rotation}deg)`,
        }}
      />
      <span className="spin-knob-label">{value || "—"}</span>
    </div>
  );
}
