import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Safe to import at module level: plain strings and a lookup, no heavy dependencies.
from algorithms import ALGORITHMS, ALGORITHM_INFO, validate_algorithm
from genres import GENRES, prompt_term, validate_genre

# Reads backend/.env and puts the keys into the process environment,
# so os.environ can see them below.
load_dotenv()

app = FastAPI(title="Image to Song")

# Pillow opens more than these, but the frontend's file picker and the tagger are
# only exercised against the common web formats, so the rest are turned away.
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

APIFRAME_BASE_URL = "https://api.apiframe.ai/v2"
POLL_EVERY_SECONDS = 5
GIVE_UP_AFTER_SECONDS = 300

# Pinned rather than left to apiframe's default, so song character can't change
# under us. Options: V4, V4_5, V4_5ALL, V4_5PLUS, V5, V5_5. All cost 11 credits.
SUNO_MODEL_VERSION = "V5_5"

# Everything generated is kept here so past runs can be cross-referenced.
# Paths are relative to this file, so the server can be started from anywhere.
HISTORY_DIR = Path(__file__).parent / "history"
IMAGES_DIR = HISTORY_DIR / "images"
AUDIO_DIR = HISTORY_DIR / "audio"
HISTORY_FILE = HISTORY_DIR / "history.jsonl"

for directory in (IMAGES_DIR, AUDIO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# The Next.js dev server is a different origin, which browsers block by default.
app.add_middleware(
    CORSMiddleware,
    # Both spellings: the browser treats them as different origins.
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    """Confirms the server is up and which keys were loaded."""
    return {
        "status": "ok",
        "apiframe_key_loaded": bool(os.environ.get("APIFRAME_API_KEY")),
    }


def read_image(image: UploadFile) -> bytes:
    """Rejects unsupported image types, then returns the raw bytes."""
    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type {image.content_type}. "
            f"Use one of: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}",
        )
    return image.file.read()


def apiframe_call(method: str, path: str, **kwargs) -> dict:
    api_key = os.environ.get("APIFRAME_API_KEY")
    if not api_key:
        raise HTTPException(
            500, "APIFRAME_API_KEY is not set. Copy .env.example to .env and add it."
        )

    response = requests.request(
        method,
        f"{APIFRAME_BASE_URL}{path}",
        headers={"X-API-Key": api_key},
        timeout=30,
        **kwargs,
    )
    if not response.ok:
        # Pass apiframe's own message through; it explains things a bare status
        # code cannot, such as a v1 key being used against the v2 API.
        raise HTTPException(502, f"apiframe {response.status_code}: {response.text[:300]}")
    return response.json()


def wait_for_job(job_id: str, what: str) -> dict:
    """Polls one apiframe job until it finishes. Blocks for a minute or more.

    apiframe returns a job id immediately rather than the finished thing, so every
    generation ends in a loop like this one, songs and images alike.
    """
    deadline = time.monotonic() + GIVE_UP_AFTER_SECONDS
    status = "QUEUED"
    while time.monotonic() < deadline:
        time.sleep(POLL_EVERY_SECONDS)
        job = apiframe_call("GET", f"/jobs/{job_id}")
        status = job["status"]
        if status == "COMPLETED":
            return job
        if status == "FAILED":
            raise HTTPException(502, job.get("error") or f"Could not generate the {what}.")

    raise HTTPException(
        504, f"The {what} was still '{status}' after {GIVE_UP_AFTER_SECONDS} seconds."
    )


def request_song(prompt: str) -> dict:
    """Submits the prompt to Suno and waits for the job to finish rendering."""
    job_id = apiframe_call(
        "POST",
        "/music/generate",
        json={
            "model": "suno",
            "prompt": prompt,
            "sunoParams": {
                "model_version": SUNO_MODEL_VERSION,
                # No vocals: the tags describe mood and style, not anything to
                # write lyrics about.
                "instrumental": True,
            },
        },
    )["jobId"]
    return wait_for_job(job_id, "song")


def safe_for_filename(text: str) -> str:
    """Strips anything from a genre or title that would break a path.

    Hyphens survive even though `-` also separates the fields in these names: nine
    genres contain one, and the names are read by people rather than parsed.
    """
    kept = "".join(c for c in text if c.isalnum() or c in " -_&'()").strip()
    return kept or "untitled"


def audio_filename(stamp: datetime, genre: str, title: str, take: int) -> str:
    """Names a track for browsing: date, time, genre, Suno's title, take number.

    Neither `/` nor `:` can appear in a filename, so the date and time use hyphens
    and underscores instead.
    """

    when = stamp.strftime("%m-%d-%Y_%H_%M_%S")
    label = safe_for_filename(genre) if genre else "no_genre"
    return f"{when}-{label}-{safe_for_filename(title)}-{take}.mp3"


def save_to_history(
    image_bytes: bytes,
    media_type: str,
    tags: list[str],
    tag_scores: dict,
    prompt: str,
    job: dict,
    genre: str,
    tagging: dict,
    algorithm: str,
) -> dict:
    """Writes the image, every take Suno rendered, and one JSONL row for the run.

    The MP3s are copied locally because apiframe's retention policy for its CDN
    links is undocumented, so a history of URLs alone could rot into dead links.
    """
    stamp = datetime.now(timezone.utc)
    # Timestamp sorts chronologically; the short uuid keeps two uploads in the
    # same second from overwriting each other.
    stem = f"{stamp.strftime('%Y-%m-%dT%H-%M-%S')}_{uuid4().hex[:8]}"

    image_name = f"{stem}.{media_type.removeprefix('image/')}"
    (IMAGES_DIR / image_name).write_bytes(image_bytes)

    # Audio is named for browsing rather than sorting, so local time.
    local_stamp = stamp.astimezone()

    # Suno renders two takes per job and both are kept: they cost nothing extra,
    # and one is usually the better listen.
    tracks = []
    for index, track in enumerate(job["result"]["tracks"]):
        download = requests.get(track["audioUrl"], timeout=120)
        if not download.ok:
            # Without this check an error page would be written under an .mp3 name
            # and recorded as a success, surfacing later as a track that won't play.
            continue

        audio_name = audio_filename(local_stamp, genre, track["title"], index + 1)
        destination = AUDIO_DIR / audio_name
        if destination.exists():
            # Two runs in the same second sharing a genre and title would
            # otherwise silently overwrite each other's audio.
            audio_name = f"{destination.stem}_{uuid4().hex[:4]}.mp3"
            destination = AUDIO_DIR / audio_name
        destination.write_bytes(download.content)
        tracks.append(
            {
                "title": track["title"],
                "duration": track["duration"],
                "audioUrl": track["audioUrl"],
                "audioFile": f"audio/{audio_name}",
            }
        )

    if not tracks:
        raise HTTPException(
            502, "Suno rendered the song but none of its takes could be downloaded."
        )

    record = {
        "timestamp": stamp.isoformat(),
        "image": f"images/{image_name}",
        # Empty when the user picked no genre, in which case nothing was
        # prepended and the prompt is purely the image's tags.
        "genre": genre,
        "algorithm": algorithm,
        "tags": tags,
        "tag_scores": tag_scores,
        "tagging": tagging,
        "prompt": prompt,
        "job_id": job["id"],
        "tracks": tracks,
    }
    # CoCa's caption is display-only (never touches tag selection or the Suno
    # prompt) but worth keeping on the history record when it was generated.
    if "caption" in tagging:
        record["caption"] = tagging["caption"]

    with HISTORY_FILE.open("a") as history:
        history.write(json.dumps(record) + "\n")

    return record


def iter_history():
    """Past runs newest first, decoded one row at a time."""
    if not HISTORY_FILE.exists():
        return
    for line in reversed(HISTORY_FILE.read_text().splitlines()):
        if line.strip():
            yield json.loads(line)


@app.get("/history")
def read_history() -> list[dict]:
    """Every past run, newest first."""
    return list(iter_history())


@app.delete("/history")
def wipe_history() -> dict:
    """Deletes every saved image, MP3 and history row. Not reversible."""
    # Counting lines is enough; parsing every record just to discard it is not.
    entries = 0
    if HISTORY_FILE.exists():
        entries = sum(1 for line in HISTORY_FILE.read_text().splitlines() if line.strip())

    files_removed = 0
    bytes_freed = 0

    for directory in (IMAGES_DIR, AUDIO_DIR):
        for path in directory.iterdir():
            if path.is_file():
                bytes_freed += path.stat().st_size
                path.unlink()
                files_removed += 1

    HISTORY_FILE.unlink(missing_ok=True)

    return {
        "entries_removed": entries,
        "files_removed": files_removed,
        "megabytes_freed": round(bytes_freed / 1_000_000, 1),
    }


# Deliberately a plain `def` rather than `async def`: this handler blocks while
# polling, and FastAPI runs sync handlers on a worker thread so that waiting
# does not freeze the whole server.
@app.get("/genres")
def list_genres() -> list[str]:
    """The genres the picker offers. Served so the list lives in one place only."""
    return GENRES


@app.get("/algorithms")
def list_algorithms() -> list[dict]:
    """The embedding algorithms the picker offers, with a label/blurb each."""
    return [{"id": algorithm, **ALGORITHM_INFO[algorithm]} for algorithm in ALGORITHMS]


def build_tagging_summary(
    ranked: dict,
    selected_tags: list[str],
    selected_scores: dict,
    verdicts: list[dict],
    final_tags: list[str],
    final_scores: dict,
    replacements: dict,
) -> dict:
    """Assembles everything the frontend needs to show how the 10 tags were picked.

    `ranked` already holds every candidate word per facet (see
    `siglip_tags_by_facet`), so this is pure bookkeeping: no re-scoring.
    """
    # Deferred like the other embedding_tagger imports in `tag()` below -- this
    # function is only ever called from there, after torch is already loaded,
    # but importing it lazily here too keeps `build_tagging_summary` correct
    # to call on its own rather than relying on caller-side import ordering.
    from embedding_tagger import word_to_facet

    facet_of = word_to_facet(ranked)

    top20 = sorted(
        (
            {"tag": word, "facet": facet, "score": score}
            for facet, candidates in ranked.items()
            for word, score in candidates
        ),
        key=lambda entry: entry["score"],
        reverse=True,
    )[:20]

    return {
        "top20": top20,
        "selected": [
            {"tag": tag, "facet": facet_of[tag], "score": selected_scores[tag]}
            for tag in selected_tags
        ],
        "judge": verdicts,
        "final": [
            {"tag": tag, "facet": facet_of[tag], "score": final_scores[tag]}
            for tag in final_tags
        ],
        "replacements": [
            {"from": old, "to": new} for old, new in replacements.items()
        ],
    }


@app.post("/tag")
def tag(image: UploadFile = File(...), genre: str = "", algorithm: str = "siglip"):
    """Runs the tagging half of the pipeline only: embedding scoring + the judge.

    No Suno call and nothing saved to history -- this is the free, fast preview
    that lets a user see (and the judge react to) an image's tags before they
    commit to actually rendering a song. `/generate` below takes this response's
    tags/tag_scores/tagging/prompt straight back so the judge never runs twice
    for the same image.
    """
    # Imported here rather than at module level so `import main` stays fast and
    # torch/anthropic are only loaded once an actual request needs them.
    from embedding_tagger import (
        tags_by_facet,
        select_tags,
        replace_flagged_tags,
        embedding_projection,
        generate_caption,
    )
    from judge import judge_image

    validate_genre(genre)
    validate_algorithm(algorithm)

    image_bytes = read_image(image)
    ranked = tags_by_facet(image_bytes, algorithm=algorithm)
    tags, tag_scores = select_tags(ranked)
    selected_tags, selected_scores = tags, tag_scores

    # The judge is a quality improvement, not a hard dependency: if the Claude
    # call errors (rate limit, timeout, missing key), fall back to the untouched
    # embedding tags rather than failing a request that Suno could still render.
    # Runs identically regardless of which algorithm produced the tags it reviews.
    verdicts: list[dict] = []
    replacements: dict[str, str] = {}
    try:
        verdicts = judge_image(image_bytes, tags)
        flagged = {v["tag"] for v in verdicts if v["flagged"]}
        if flagged:
            tags, tag_scores, replacements = replace_flagged_tags(
                ranked, tags, tag_scores, flagged
            )
    except Exception as exc:
        print(f"judge call failed, using unjudged tags: {exc}")

    tagging = build_tagging_summary(
        ranked, selected_tags, selected_scores, verdicts, tags, tag_scores, replacements
    )
    tagging["embedding"] = embedding_projection(
        image_bytes,
        [(point["tag"], point["facet"]) for point in tagging["top20"]],
        algorithm=algorithm,
    )

    # CoCa-only: a free-text caption, display-only -- never fed into the tags or
    # the Suno prompt below.
    if algorithm == "coca":
        tagging["caption"] = generate_caption(image_bytes)

    # The genre leads the prompt rather than using Suno's `style` parameter,
    # which is its documented home but made no audible difference in testing.
    prompt = ", ".join([prompt_term(genre), *tags] if genre else tags)

    return {
        "tags": tags,
        "tag_scores": tag_scores,
        "tagging": tagging,
        "prompt": prompt,
        "algorithm": algorithm,
    }


@app.post("/generate")
def generate(
    image: UploadFile = File(...),
    genre: str = "",
    algorithm: str = "siglip",
    tags: str = Form(...),
    tag_scores: str = Form(...),
    tagging: str = Form(...),
    prompt: str = Form(...),
):
    """Renders and saves a song from a `/tag` response the frontend already has.

    Deliberately does no tagging of its own -- the image, tags, and judge
    verdicts were already settled in `/tag`, so this step is Suno + history only.
    `algorithm` isn't used for anything here except recording which one produced
    the tags being rendered -- validating it doesn't need embedding_tagger's heavy
    imports, so it's checked against the lightweight `algorithms` module instead.
    """
    validate_genre(genre)
    validate_algorithm(algorithm)

    image_bytes = read_image(image)
    job = request_song(prompt)

    return save_to_history(
        image_bytes,
        image.content_type,
        json.loads(tags),
        json.loads(tag_scores),
        prompt,
        job,
        genre,
        json.loads(tagging),
        algorithm,
    )


# Serves the saved images and MP3s, so the frontend can play a song from local
# disk rather than depending on apiframe's CDN still holding it.
app.mount("/files", StaticFiles(directory=HISTORY_DIR), name="files")
