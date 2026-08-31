import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Safe to import at module level: plain strings and a lookup, no heavy dependencies.
from algorithms import ALGORITHMS, ALGORITHM_INFO, validate_algorithm
from genres import GENRES, prompt_term, validate_genre
import storage

# Reads backend/.env and puts the keys into the process environment,
# so os.environ can see them below.
load_dotenv()

# Next's static export output (`next build` with output: "export" in
# next.config.ts) -- present in the Docker image (see Dockerfile), absent in
# local dev, where the frontend runs as its own `npm run dev` process
# instead. Doubles here as the signal for "is this the deployed container,"
# since that's exactly when eagerly preloading below is worth its cost.
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "out"


def _preload_algorithm(algorithm: str) -> None:
    """Warms one embedding algorithm in the background at startup, so a
    visitor's first /tag call for it doesn't pay for loading it mid-request.

    Runs off the request path, in its own thread rather than blocking
    startup -- Cloud Run's own startup check shouldn't have to wait for any
    of these to finish. embedding_tagger's per-algorithm lock already makes
    this safe to race against a real request that gets to this algorithm
    first, and against the other three algorithms' own preload threads,
    which never touch each other's state.
    """
    from embedding_tagger import _load

    try:
        _load(algorithm)
    except Exception as exc:
        print(f"background preload of {algorithm!r} failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only in the deployed container -- local dev's `--reload` restarts
    # shouldn't eagerly reload every algorithm on every file save. One
    # thread per algorithm, all four started together, rather than one
    # thread looping through them -- so a cold start's total load time is
    # bounded by however long the slowest single algorithm takes, not the
    # sum of all four.
    if FRONTEND_DIST.is_dir():
        for algorithm in ALGORITHMS:
            threading.Thread(target=_preload_algorithm, args=(algorithm,), daemon=True).start()
    yield


app = FastAPI(title="Ideagram Studio", lifespan=lifespan)

# Pillow opens more than these, but the frontend's file picker and the tagger are
# only exercised against the common web formats, so the rest are turned away.
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

APIFRAME_BASE_URL = "https://api.apiframe.ai/v2"
POLL_EVERY_SECONDS = 5
GIVE_UP_AFTER_SECONDS = 300

# Pinned rather than left to apiframe's default, so song character can't change
# under us. Options: V4, V4_5, V4_5ALL, V4_5PLUS, V5, V5_5. All cost 11 credits.
SUNO_MODEL_VERSION = "V5_5"

# Generated audio is downloaded here (via storage.py) so playback doesn't
# depend on apiframe's CDN links staying alive, and so local dev needs no GCS
# bucket. Path is relative to this file, so the server can be started from
# anywhere. Only used directly here for the /files mount below -- storage.py
# owns the per-token layout underneath it.
HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# The repo-root sample image set, served so the frontend's "select an image"
# grid can offer them without the user needing their own file.
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test-images"

MAX_SESSIONS = 20

# One run list per visitor, in memory for the life of this process --
# recovered from storage on first touch per token (see _session_for), since
# a Cloud Run instance recycle (idle scale-to-zero, a redeploy, a crash)
# drops this dict the same way a restart drops SESSIONS did before per-visitor
# isolation existed. Each record:
#   {id, created_at, algorithm, image_source, tags, tag_scores, tagging,
#    prompt, tracks}
# where image_source is {"kind": "sample", "name": <str>} or
# {"kind": "upload", "filename": <str>, "ref": <str>, "url": <str>} (ref is
# what storage.py's delete_file takes back; url is what the frontend plays/
# displays directly). `tracks` stays None until /generate fills it in.
_sessions: dict[str, list[dict]] = {}


def _session_for(request: Request) -> tuple[str, list[dict]]:
    """This visitor's token and run list. The token itself was already
    resolved by session_cookie_middleware before any route runs; the list is
    recovered from storage the first time this process sees this token, then
    cached here for the rest of the process's life.
    """
    token = request.state.session_token
    if token not in _sessions:
        _sessions[token] = storage.load_session_records(token) or []
    return token, _sessions[token]


def find_session(records: list[dict], run_id: str) -> dict | None:
    """Looks up one session record by id, or None if it's gone (already
    deleted, or the backend restarted since and storage had nothing for it)."""
    return next((record for record in records if record["id"] == run_id), None)


# The Next.js dev server is a different origin, which browsers block by default.
# Added before session_cookie_middleware below so that middleware ends up the
# outer layer at runtime (Starlette's add_middleware stacks the most
# recently added one outermost) -- every request resolves its session token
# first, CORS headers get applied around the actual route handling underneath.
app.add_middleware(
    CORSMiddleware,
    # Both spellings: the browser treats them as different origins.
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Lets the session cookie round-trip cross-origin in local dev -- without
    # this the browser drops Set-Cookie on the response and never sends the
    # cookie back, so every request would look like a brand new visitor.
    allow_credentials=True,
)

SESSION_COOKIE = "session_id"
# 30 days -- matches the GCS bucket's own age-based lifecycle rule, so the
# cookie and the visitor's persisted data expire together rather than the
# cookie dropping first and orphaning data nothing can reach anymore.
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30


@app.middleware("http")
async def session_cookie_middleware(request: Request, call_next):
    """Resolves this visitor's session token before the route runs, minting
    one when the request carries none, and sets it on every response -- an
    error alike a success. A cookie set through a route's own Response
    parameter only applies on the success path in FastAPI; raising
    HTTPException discards it silently, so middleware is what makes the
    cookie stick no matter how the request ends.
    """
    token = request.cookies.get(SESSION_COOKIE)
    minted = token is None
    if minted:
        token = secrets.token_urlsafe(32)
    request.state.session_token = token

    response = await call_next(request)

    if minted:
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            # "none" rather than "lax": unlike the sibling project (whose Vite
            # dev server proxies /api, making every request same-origin from
            # the browser's point of view), this frontend calls the backend
            # directly cross-origin in local dev (:3000 -> :8000). A Lax
            # cookie is never attached to a cross-site fetch/XHR (only to a
            # top-level navigation), so it would round-trip on the very first
            # response and then silently stop being sent on every request
            # after -- confirmed live: every reload looked like a brand new,
            # empty session. SameSite=None requires Secure, which still works
            # here since Chromium/Firefox/Safari all treat 127.0.0.1/localhost
            # as a potentially-trustworthy origin even over plain HTTP.
            samesite="none",
            secure=True,
        )
    return response


@app.get("/api/health")
def health():
    """Confirms the server is up and which keys were loaded.

    Not at bare "/" -- in production that path serves the built frontend
    (see the conditional mount at the bottom of this file), so the health
    check needs its own path to avoid colliding with it.
    """
    return {
        "status": "ok",
        "apiframe_key_loaded": bool(os.environ.get("APIFRAME_API_KEY")),
        "anthropic_key_loaded": bool(os.environ.get("ANTHROPIC_API_KEY")),
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
    token: str,
    run_id: str,
    original_filename: str | None,
    job: dict,
    algorithm: str,
) -> list[dict]:
    """Downloads every take Suno rendered for this run through storage.py.

    The MP3s are copied off apiframe's CDN because its retention policy for
    those links is undocumented, so pointing the player straight at
    `audioUrl` could rot into a dead link. These files, and the session's own
    record of them, are removed either by this run's own
    `DELETE /history/{run_id}` or by a bulk `DELETE /history`.
    """
    stamp = datetime.now(timezone.utc).astimezone()
    image_label = image_base_name(original_filename)

    # Suno renders two takes per job and both are kept: they cost nothing extra,
    # and one is usually the better listen. The two downloads are independent
    # I/O with no shared state, so fetch them concurrently.
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

        # Nested under this run's own id, so two takes named alike (same image,
        # algorithm, and take number, generated the same day) can never collide
        # the way they could under the old single flat AUDIO_DIR -- run_id is
        # already unique, so no separate collision check is needed here.
        audio_name = audio_filename(image_label, algorithm, index + 1, stamp)
        ref = storage.put_file(token, f"audio/{run_id}/{audio_name}", download.content)
        tracks.append(
            {
                "title": track["title"],
                "duration": track["duration"],
                "audioUrl": track["audioUrl"],
                "ref": ref,
                "url": storage.playback_url(ref),
            }
        )

    if not tracks:
        raise HTTPException(
            502, "Suno rendered the song but none of its takes could be downloaded."
        )

    return tracks


@app.delete("/history")
def wipe_history(request: Request) -> dict:
    """Deletes every MP3 and persisted image this visitor has downloaded so
    far, and clears their run list. Not reversible.

    Called by the frontend's Clear button. Unlike the per-run delete below,
    this is a full wipe regardless of how many runs exist.
    """
    token, records = _session_for(request)
    files_removed = 0
    bytes_freed = 0

    for record in records:
        refs = [track["ref"] for track in record["tracks"] or []]
        if record["image_source"]["kind"] == "upload":
            refs.append(record["image_source"]["ref"])
        for ref in refs:
            freed = storage.delete_file(ref)
            if freed is not None:
                bytes_freed += freed
                files_removed += 1

    records.clear()
    storage.delete_session_records(token)

    return {
        "files_removed": files_removed,
        "megabytes_freed": round(bytes_freed / 1_000_000, 1),
    }


def refreshed(record: dict) -> dict:
    """A copy of one run record with every track/image url recomputed from
    its ref rather than trusting whatever was persisted with it.

    Locally that's a no-op (/files/<ref> never expires), but a GCS signed
    URL does -- generate_signed_url's expiration is far shorter than the
    30-day retention window a run can otherwise sit in the bucket for, so a
    record read back after that needs a fresh URL, not the stale one it was
    saved with.
    """
    record = {**record}
    if record["image_source"]["kind"] == "upload":
        record["image_source"] = {
            **record["image_source"],
            "url": storage.playback_url(record["image_source"]["ref"]),
        }
    if record["tracks"]:
        record["tracks"] = [
            {**track, "url": storage.playback_url(track["ref"])}
            for track in record["tracks"]
        ]
    return record


@app.get("/history")
def list_history(request: Request) -> list[dict]:
    """This visitor's runs, newest first -- lets a frontend page reload
    rehydrate its cards, as long as this process still has (or storage can
    recover) that visitor's run list."""
    _, records = _session_for(request)
    return [refreshed(record) for record in reversed(records)]


@app.delete("/history/{run_id}")
def delete_run(request: Request, run_id: str) -> dict:
    """Removes one run: its record, its MP3s, and its persisted image (if
    it has one). 404 if the id isn't there -- already deleted, or this
    visitor's run list doesn't have it."""
    token, records = _session_for(request)
    record = find_session(records, run_id)
    if record is None:
        raise HTTPException(404, f"No run with id {run_id!r}.")

    files_removed = 0
    bytes_freed = 0

    refs = [track["ref"] for track in record["tracks"] or []]
    if record["image_source"]["kind"] == "upload":
        refs.append(record["image_source"]["ref"])

    for ref in refs:
        freed = storage.delete_file(ref)
        if freed is not None:
            bytes_freed += freed
            files_removed += 1

    records.remove(record)
    storage.save_session_records(token, records)

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
    request: Request,
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

    Rejects with 409 once this visitor's session already holds `MAX_SESSIONS`
    runs -- checked here, before any tagging work (including the heavy model
    imports below) runs, rather than after paying for a tag extraction that
    couldn't be kept anyway.
    """
    token, records = _session_for(request)
    if len(records) >= MAX_SESSIONS:
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
        image_source = {"kind": "upload", "filename": image.filename or ""}
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
    # bytes are persisted, and the run is added to the session list, here
    # rather than up front, so a request that errors out above leaves neither
    # an orphaned file nor a half-formed record behind.
    if image_source["kind"] == "upload":
        ref = storage.put_file(token, f"images/{run_id}.{ext}", image_bytes)
        image_source["ref"] = ref
        image_source["url"] = storage.playback_url(ref)

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
    records.append(record)
    storage.save_session_records(token, records)

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
def generate(request: Request, run_id: str):
    """Renders a song for a run `/tag` already produced.

    Deliberately does no tagging of its own -- the image, tags, and judge
    verdicts were already settled in `/tag` and are looked up here off the
    session record, so this step is Suno + audio download only, and the
    judge never runs twice for the same image. `run_id` is the only input;
    everything else (algorithm, prompt, the original image's name for the
    audio naming convention) comes off the stored record. 404s if the id
    isn't in this visitor's run list -- already deleted, or storage had
    nothing to recover for this token.
    """
    token, records = _session_for(request)
    record = find_session(records, run_id)
    if record is None:
        raise HTTPException(404, f"No run with id {run_id!r}. It may have been deleted, or the backend restarted since it was tagged.")

    image_source = record["image_source"]
    original_filename = (
        image_source["filename"] if image_source["kind"] == "upload" else image_source["name"]
    )

    job = request_song(record["prompt"])
    tracks = save_audio(token, run_id, original_filename, job, record["algorithm"])
    record["tracks"] = tracks
    storage.save_session_records(token, records)

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

# Serves the built frontend at "/", registered last so every API route above
# still matches first -- StaticFiles(html=True) only handles requests none of
# those routes claimed. Only present when FRONTEND_DIST actually exists (the
# deployed container), so local dev without a `next build` in hand is
# unaffected and keeps using the two-process (`npm run dev` + uvicorn) setup.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
