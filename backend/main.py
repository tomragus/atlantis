import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Safe to import at module level: plain strings and a lookup, no heavy dependencies.
from algorithms import ALGORITHMS, ALGORITHM_INFO, validate_algorithm
from genres import GENRES, prompt_term, validate_genre

# Reads backend/.env and puts the keys into the process environment,
# so os.environ can see them below.
load_dotenv()

app = FastAPI(title="Ideagram Studio")

# Pillow opens more than these, but the frontend's file picker and the tagger are
# only exercised against the common web formats, so the rest are turned away.
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

APIFRAME_BASE_URL = "https://api.apiframe.ai/v2"
POLL_EVERY_SECONDS = 5
GIVE_UP_AFTER_SECONDS = 300

# Pinned rather than left to apiframe's default, so song character can't change
# under us. Options: V4, V4_5, V4_5ALL, V4_5PLUS, V5, V5_5. All cost 11 credits.
SUNO_MODEL_VERSION = "V5_5"

# Generated audio is downloaded here so playback doesn't depend on apiframe's
# CDN links staying alive. Paths are relative to this file, so the server can
# be started from anywhere.
HISTORY_DIR = Path(__file__).parent / "history"
AUDIO_DIR = HISTORY_DIR / "audio"
# Uploaded (non-sample) images, persisted as soon as /tag succeeds so a
# reloaded card's thumbnail is always the real photo -- never a placeholder.
# Sample images need no copy here; they're already servable by name out of
# TEST_IMAGES_DIR regardless of session.
IMAGES_DIR = HISTORY_DIR / "images"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# The repo-root sample image set, served so the frontend's "select an image"
# grid can offer them without the user needing their own file.
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test-images"

# The whole session's list of runs, newest-appended, in-memory only -- reset
# by any backend restart (including the --reload autoreload dev triggers).
# One shared list, no per-visitor isolation: matches this phase's explicit
# local-dev-only scope. Each record:
#   {id, created_at, algorithm, image_source, tags, tag_scores, tagging,
#    prompt, tracks}
# where image_source is {"kind": "sample", "name": <str>} or
# {"kind": "upload", "filename": <str>, "path": <str>} (path is relative to
# HISTORY_DIR, servable through the /files mount below). `tracks` stays None
# until /generate fills it in.
SESSIONS: list[dict] = []
MAX_SESSIONS = 20


def find_session(run_id: str) -> dict | None:
    """Looks up one session record by id, or None if it's gone (already
    deleted, or the backend restarted since)."""
    return next((record for record in SESSIONS if record["id"] == run_id), None)

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
    """Strips anything from a genre, title, or filename that would break a path.

    Hyphens survive even though `-` also separates the fields in these names: nine
    genres contain one, and the names are read by people rather than parsed.
    """
    kept = "".join(c for c in text if c.isalnum() or c in " -_&'()").strip()
    return kept or "untitled"


def image_base_name(filename: str | None) -> str:
    """The uploaded image's own filename, sanitized for reuse in the audio filename.

    `Path(...).stem` drops both the extension and any directory components a
    client could smuggle in via the filename field. Falls back to "untitled" if
    the frontend sent an empty string (its default when it has no file name to
    give).
    """
    stem = Path(filename).stem if filename else ""
    return safe_for_filename(stem) if stem else "untitled"


def audio_filename(image_name: str, algorithm: str, take: int, stamp: datetime) -> str:
    """Names a track for browsing: which image, which algorithm tagged it, which
    take, and the date -- e.g. "sunset-beach-clip-1-08-17-2026.mp3".

    Date-only rather than a full timestamp: a same-day collision (same image name,
    algorithm, and take twice in one day) falls back to the short-uuid suffix in
    `save_audio` below.
    """
    when = stamp.strftime("%m-%d-%Y")
    return f"{image_name}-{algorithm}-{take}-{when}.mp3"


def save_audio(
    original_filename: str | None,
    job: dict,
    algorithm: str,
) -> list[dict]:
    """Downloads every take Suno rendered for this run to local disk.

    The MP3s are copied locally because apiframe's retention policy for its CDN
    links is undocumented, so pointing the player straight at `audioUrl` could
    rot into a dead link. These files, and the session's own record of them,
    are removed either by this run's own `DELETE /history/{run_id}` or by a
    bulk `DELETE /history`.
    """
    stamp = datetime.now(timezone.utc).astimezone()
    image_label = image_base_name(original_filename)

    # Suno renders two takes per job and both are kept: they cost nothing extra,
    # and one is usually the better listen. The two downloads are independent
    # I/O with no shared state, so fetch them concurrently rather than waiting
    # on one before starting the next; the collision-checking/write loop below
    # stays sequential (and in take order) since it does touch shared state
    # (the filesystem, and `destination.exists()` checks against files this
    # same call is writing).
    raw_tracks = job["result"]["tracks"]
    with ThreadPoolExecutor(max_workers=max(len(raw_tracks), 1)) as executor:
        downloads = list(
            executor.map(lambda t: requests.get(t["audioUrl"], timeout=120), raw_tracks)
        )

    tracks = []
    for index, (track, download) in enumerate(zip(raw_tracks, downloads)):
        if not download.ok:
            # Without this check an error page would be written under an .mp3 name
            # and recorded as a success, surfacing later as a track that won't play.
            continue

        audio_name = audio_filename(image_label, algorithm, index + 1, stamp)
        destination = AUDIO_DIR / audio_name
        if destination.exists():
            # Date-only granularity means the same image name, algorithm, and take
            # collide easily within a single day, so this fallback is the normal
            # case for a second run of the same image on the same day, not just a
            # rare coincidence.
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

    return tracks


def _remove_file(path: Path) -> int | None:
    """Deletes path if it exists and returns the bytes freed, else None."""
    if not path.is_file():
        return None
    size = path.stat().st_size
    path.unlink()
    return size


@app.delete("/history")
def wipe_history() -> dict:
    """Deletes every MP3 and persisted image downloaded so far this session,
    and clears the in-memory session list. Not reversible.

    Called by the frontend's Clear button. Unlike the per-run delete below,
    this is a full wipe regardless of how many runs exist.
    """
    files_removed = 0
    bytes_freed = 0

    for directory in (AUDIO_DIR, IMAGES_DIR):
        for path in directory.iterdir():
            freed = _remove_file(path)
            if freed is not None:
                bytes_freed += freed
                files_removed += 1

    SESSIONS.clear()

    return {
        "files_removed": files_removed,
        "megabytes_freed": round(bytes_freed / 1_000_000, 1),
    }


@app.get("/history")
def list_history() -> list[dict]:
    """The whole session's runs, newest first -- lets a frontend page reload
    rehydrate its cards, as long as the backend process itself hasn't
    restarted (the in-memory tradeoff; see SESSIONS above)."""
    return list(reversed(SESSIONS))


@app.delete("/history/{run_id}")
def delete_run(run_id: str) -> dict:
    """Removes one run: its record, its MP3s, and its persisted image (if
    it has one). 404 if the id isn't there -- already deleted, or the
    backend restarted since."""
    record = find_session(run_id)
    if record is None:
        raise HTTPException(404, f"No run with id {run_id!r}.")

    files_removed = 0
    bytes_freed = 0

    paths = [HISTORY_DIR / track["audioFile"] for track in record["tracks"] or []]
    if record["image_source"]["kind"] == "upload":
        paths.append(HISTORY_DIR / record["image_source"]["path"])

    for path in paths:
        freed = _remove_file(path)
        if freed is not None:
            bytes_freed += freed
            files_removed += 1

    SESSIONS.remove(record)

    return {
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


@lru_cache(maxsize=1)
def known_sample_images() -> set[str]:
    """Every filename /tag's `sample_image` param is allowed to name.

    Shared by `list_test_images` (the picker's listing) and `tag`'s own
    validation below, so the two can't drift out of sync -- a filename the
    grid offers is always one `tag` will accept, and vice versa. Cached
    rather than re-globbing the directory on every request: the corpus is
    static for the life of the server, same assumption `embedding_tagger.py`
    already makes about a loaded model's vocabulary.
    """
    return {p.name for p in TEST_IMAGES_DIR.glob("*.png")}


@app.get("/test-images")
def list_test_images() -> list[str]:
    """Filenames in test-images/, for the frontend's sample-image grid.

    Sorted so the grid order is stable across requests; served from
    `/test-image-files/<name>` (mounted below) rather than this route.
    """
    return sorted(known_sample_images())


def build_tagging_summary(
    facet_of: dict,
    verdicts: list[dict],
    contradictions: list[dict],
    final_tags: list[str],
    final_scores: dict,
) -> dict:
    """Assembles everything the frontend needs to show how the tags were picked."""
    return {
        "judge": verdicts,
        "contradictions": contradictions,
        "final": [
            {"tag": tag, "facet": facet_of[tag], "score": final_scores[tag]}
            for tag in final_tags
        ],
    }


@app.post("/tag")
def tag(
    image: UploadFile | None = File(None),
    sample_image: str = "",
    genre: str = "",
    algorithm: str = "siglip",
    judges: bool = True,
    show_embedding: bool = False,
):
    """Runs the tagging half of the pipeline only: embedding scoring + the judge.

    No Suno call here -- this is the fast preview that lets a user see (and
    the judge react to) an image's tags before they commit to actually
    rendering a song. A session record is created either way, though: once
    tagging succeeds, this run joins the session list (and, for an upload,
    its image bytes are persisted) so the card exists and is deletable/
    reloadable even before a song is ever generated. `/generate` below
    takes just this run's `id` and looks the rest up off that record, so
    the judge never runs twice for the same image.

    `judges=False` skips both Claude judge passes entirely (no call made,
    not just an ignored result) -- the frontend's LLM-judges toggle, for
    when a user wants the embedding tags as-is without the extra Claude
    cost/latency. Tags are then exactly `select_tags`'s output.

    `show_embedding=False` (the default) skips `embedding_projection` entirely
    -- the frontend's "Embedding space visual" toggle, off by default, for
    when a user doesn't want the PCA/SVD work spent on a plot nobody will
    look at. `tagging["embedding"]` is only present in the response when
    this is `True`.

    Exactly one of `image` (an upload) or `sample_image` (a filename from
    `GET /test-images`) must be given -- both or neither is a 400, not a
    silent pick-one, so a caller can't have an upload quietly ignored in
    favor of a sample image it didn't ask for. `sample_image` is read
    straight off disk here rather than the frontend fetching its bytes
    itself and re-uploading them as `image` -- that would need a
    cross-origin GET of the raw bytes to succeed under whatever CORS policy
    the browser is enforcing, which the frontend's own POSTs to this same
    endpoint don't require. Reading server-side sidesteps that entirely.

    Rejects with 409 once the session already holds `MAX_SESSIONS` runs --
    checked here, before any tagging work (including the heavy model
    imports below) runs, rather than after paying for a tag extraction that
    couldn't be kept anyway.
    """
    if len(SESSIONS) >= MAX_SESSIONS:
        raise HTTPException(
            409,
            f"This session already holds {MAX_SESSIONS} runs. "
            "Delete one (or Clear) before starting another.",
        )

    # Imported here rather than at module level so `import main` stays fast and
    # torch/anthropic are only loaded once an actual request needs them.
    from embedding_tagger import (
        embed_image,
        tags_by_facet,
        select_tags,
        drop_flagged_tags,
        word_to_facet,
        embedding_projection,
    )
    from judge_1 import judge_image
    from judge_2 import judge_contradictions

    validate_genre(genre)
    validate_algorithm(algorithm)

    run_id = uuid4().hex

    if sample_image and image is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either an uploaded image or a sample_image filename, not both.",
        )
    elif sample_image:
        if sample_image not in known_sample_images():
            raise HTTPException(status_code=400, detail=f"Unknown sample image: {sample_image}")
        image_bytes = (TEST_IMAGES_DIR / sample_image).read_bytes()
        image_source = {"kind": "sample", "name": sample_image}
    elif image is not None:
        image_bytes = read_image(image)
        # Extension from the upload's content type (e.g. "image/jpeg" ->
        # "jpeg") rather than the original filename, which could be missing
        # its extension or lying about it. No collision handling needed --
        # run_id is already unique.
        ext = (image.content_type or "").removeprefix("image/") or "png"
        image_source = {
            "kind": "upload",
            "filename": image.filename or "",
            "path": f"images/{run_id}.{ext}",
        }
    else:
        raise HTTPException(status_code=400, detail="Provide either an uploaded image or a sample_image filename.")
    # Computed once and reused below for the embedding-space plot -- tags_by_facet
    # and embedding_projection both need this image's vector, and it's the same
    # vector either way (same image, same algorithm), so there's no reason to run
    # the model over the image twice in one request.
    image_vector = embed_image(image_bytes, algorithm=algorithm)
    ranked = tags_by_facet(image_bytes, algorithm=algorithm, image_vector=image_vector)
    tags, tag_scores = select_tags(ranked)
    facet_of = word_to_facet(ranked)

    # The judge is a quality improvement, not a hard dependency: if the Claude
    # call errors (rate limit, timeout, missing key), fall back to the untouched
    # embedding tags rather than failing a request that Suno could still render.
    # Runs identically regardless of which algorithm produced the tags it reviews.
    # A flagged tag is dropped outright rather than replaced -- the judge is
    # saying the word doesn't belong, not that some other word from the same
    # facet does, so the tag list just gets shorter.
    verdicts: list[dict] = []
    if judges:
        try:
            verdicts = judge_image(image_bytes, tags)
            flagged = {v["tag"] for v in verdicts if v["flagged"]}
            if flagged:
                tags, tag_scores = drop_flagged_tags(tags, tag_scores, flagged)
        except Exception as exc:
            print(f"judge call failed, using unjudged tags: {exc}")

    # Second pass: the per-tag judge above scores each tag in isolation, so it
    # can't catch two tags from different facets that contradict each other's
    # mood (e.g. "joyful" from valence next to "haunting" from mood). Runs on
    # the tags the first judge already settled, same degrade-gracefully
    # contract and same drop-outright behavior -- this judge only decides
    # *which* tag of a contradicting pair to drop.
    contradictions: list[dict] = []
    if judges:
        try:
            contradictions = judge_contradictions(tags, facet_of)
            drop = {c["drop"] for c in contradictions}
            if drop:
                tags, tag_scores = drop_flagged_tags(tags, tag_scores, drop)
        except Exception as exc:
            print(f"contradiction judge call failed, using tags as-is: {exc}")

    tagging = build_tagging_summary(
        facet_of,
        verdicts,
        contradictions,
        tags,
        tag_scores,
    )
    # Lets the frontend tell "judges ran and flagged nothing" apart from
    # "judges were skipped" -- both look identical as empty `judge`/
    # `contradictions` lists otherwise.
    tagging["judges_enabled"] = judges
    # Same reasoning as judges_enabled: `tagging["embedding"]`'s mere absence
    # already signals "not requested" today (embedding_projection's output is
    # never itself empty/falsy), but an explicit field means a future reader
    # doesn't have to know that implicit contract to recover this request's
    # actual show_embedding value -- e.g. if it's ever logged to history.
    tagging["embedding_enabled"] = show_embedding
    if show_embedding:
        tagging["embedding"] = embedding_projection(
            image_bytes,
            [(point["tag"], point["facet"]) for point in tagging["final"]],
            algorithm=algorithm,
            image_vector=image_vector,
        )

    # The genre leads the prompt rather than using Suno's `style` parameter,
    # which is its documented home but made no audible difference in testing.
    prompt = ", ".join([prompt_term(genre), *tags] if genre else tags)

    # Only reached once tagging has actually succeeded -- an uploaded image's
    # bytes are written to disk, and the run is added to the session list,
    # here rather than up front, so a request that errors out above leaves
    # neither an orphaned file nor a half-formed record behind.
    if image_source["kind"] == "upload":
        (IMAGES_DIR / Path(image_source["path"]).name).write_bytes(image_bytes)

    record = {
        "id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "image_source": image_source,
        "tags": tags,
        "tag_scores": tag_scores,
        "tagging": tagging,
        "prompt": prompt,
        "tracks": None,
    }
    SESSIONS.append(record)

    return {
        "id": run_id,
        "tags": tags,
        "tag_scores": tag_scores,
        "tagging": tagging,
        "prompt": prompt,
        "algorithm": algorithm,
        "image_source": image_source,
    }


@app.post("/generate")
def generate(run_id: str):
    """Renders a song for a run `/tag` already produced.

    Deliberately does no tagging of its own -- the image, tags, and judge
    verdicts were already settled in `/tag` and are looked up here off the
    session record, so this step is Suno + audio download only, and the
    judge never runs twice for the same image. `run_id` is the only input;
    everything else (algorithm, prompt, the original image's name for the
    audio naming convention) comes off the stored record. 404s if the id
    isn't in the session list -- already deleted, or the backend restarted
    since `/tag` ran.
    """
    record = find_session(run_id)
    if record is None:
        raise HTTPException(404, f"No run with id {run_id!r}. It may have been deleted, or the backend restarted since it was tagged.")

    image_source = record["image_source"]
    original_filename = (
        image_source["filename"] if image_source["kind"] == "upload" else image_source["name"]
    )

    job = request_song(record["prompt"])
    tracks = save_audio(original_filename, job, record["algorithm"])
    record["tracks"] = tracks

    return {
        "id": run_id,
        "algorithm": record["algorithm"],
        "tags": record["tags"],
        "tag_scores": record["tag_scores"],
        "tagging": record["tagging"],
        "prompt": record["prompt"],
        "job_id": job["id"],
        "tracks": tracks,
    }


# Serves the saved images and MP3s, so the frontend can play a song from local
# disk rather than depending on apiframe's CDN still holding it.
app.mount("/files", StaticFiles(directory=HISTORY_DIR), name="files")

# Serves the sample images themselves; GET /test-images above only lists names.
app.mount(
    "/test-image-files", StaticFiles(directory=TEST_IMAGES_DIR), name="test-image-files"
)
