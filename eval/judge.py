"""LLM-as-judge flow for the SigLIP tag corpus.

For every image in test-images/: get its SigLIP tags, send the image and the
tags to Claude Sonnet 5 acting as a vision judge, and ask it to flag any tag that
doesn't actually describe the image. Writes eval/results.json (raw data) and
eval/report.html (human-readable overview).

Run with backend/.venv/bin/python (has torch/transformers/pillow/anthropic already
installed) with ANTHROPIC_API_KEY set in backend/.env. See eval/README.md.

The judge prompt/schema/call itself live in backend/judge.py, shared with the live
app's /generate flow. This script only ever calls siglip_tags()/judge_image()
directly -- it never imports main.py, apiframe_call, or request_song, and never
starts or talks to the FastAPI app.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# backend/ is a sibling directory, not a package, and siglip_tagger.py does a bare
# `from vocabulary import ...` that only resolves if backend/ is on sys.path.
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Must run before `from judge import judge_image` below: judge.py builds its
# Anthropic client at import time, so ANTHROPIC_API_KEY has to already be in
# the environment by then, not after.
load_dotenv(BACKEND_DIR / ".env")

from siglip_tagger import siglip_tags  # noqa: E402
from judge import judge_image  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
IMAGES_DIR = REPO_ROOT / "test-images"
RESULTS_PATH = Path(__file__).parent / "results.json"
REPORT_PATH = Path(__file__).parent / "report.html"


def process_image(path: Path) -> dict:
    image_bytes = path.read_bytes()
    tags, scores = siglip_tags(image_bytes)
    verdicts = judge_image(image_bytes, tags)

    tag_entries = [
        {
            "tag": v["tag"],
            "score": scores[v["tag"]],
            "flagged": v["flagged"],
            "reason": v["reason"],
        }
        for v in verdicts
    ]
    return {"filename": path.name, "tags": tag_entries}


def main() -> None:
    image_paths = sorted(
        p
        for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    print(f"Found {len(image_paths)} images in {IMAGES_DIR}", flush=True)

    # Checkpoint after every image rather than only at the end. A 36-image run
    # at ~15-20s per Ollama call takes 10-20 minutes wall clock, comfortably
    # longer than some callers' process/tool timeouts -- if this gets killed
    # partway through, results.json still holds everything completed so far
    # instead of losing the whole run. Also lets a re-run skip images that are
    # already done.
    results: list[dict] = []
    done_filenames: set[str] = set()
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text())
            results = [e for e in existing if "error" not in e]
            done_filenames = {e["filename"] for e in results}
            if done_filenames:
                print(f"Resuming: {len(done_filenames)} images already done", flush=True)
        except Exception:
            pass  # corrupt/partial file from a prior interrupted run -- start clean

    for i, path in enumerate(image_paths, 1):
        if path.name in done_filenames:
            print(f"[{i}/{len(image_paths)}] {path.name} (skip, already done)", flush=True)
            continue
        print(f"[{i}/{len(image_paths)}] {path.name}", flush=True)
        try:
            results.append(process_image(path))
        except Exception as exc:
            # One bad image (e.g. a judge timeout or malformed JSON) shouldn't
            # kill the whole 36-image run. Record it so it's visible in the
            # output rather than silently missing an entry.
            print(f"    ERROR: {exc}", flush=True)
            results.append({"filename": path.name, "error": str(exc)})

        RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"Wrote {RESULTS_PATH}", flush=True)

    from report import generate_report

    generate_report(results, IMAGES_DIR, REPORT_PATH)
    print(f"Wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
