import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Safe to import at module level: plain strings and a lookup, no heavy dependencies.
from algorithms import ALGORITHMS, ALGORITHM_INFO, DEFAULT_ALGORITHM, validate_algorithm
from genres import GENRES, prompt_term, validate_genre
from thumbnails import thumbnail_path
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
    startup -- Cloud Run's own startup check shouldn't have to wait for it
    to finish. embedding_tagger's per-algorithm lock already makes this safe
    to race against a real request that gets to this algorithm first.
    """
    from embedding_tagger import _load

    try:
        _load(algorithm)
    except Exception as exc:
        print(f"background preload of {algorithm!r} failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only in the deployed container -- local dev's `--reload` restarts
    # shouldn't eagerly reload every algorithm on every file save.
    #
    # Only DEFAULT_ALGORITHM (SigLIP), not all four: a burst of traffic big
    # enough to spin up several fresh Cloud Run instances at once (exactly
    # the case maxScale>1 exists for) used to mean each of those instances
    # spent its first several seconds with all four preload threads fighting
    # an actual incoming request for the same CPU -- confirmed live, cold
    # requests were taking 15-40s under a concurrent-instance-startup burst.
    # The other three algorithms still load lazily and correctly on first
    # use (embedding_tagger._load's own per-algorithm lock handles that
    # regardless of whether anything preloaded it) -- this only removes the
    # eager warm-up for algorithms most visitors aren't defaulting to, per
    # frontend/app/page.tsx's own selectedAlgorithm default.
    if FRONTEND_DIST.is_dir():
        threading.Thread(target=_preload_algorithm, args=(DEFAULT_ALGORITHM,), daemon=True).start()
    yield


app = FastAPI(title="Ideagram Studio", lifespan=lifespan)

# Pillow opens more than these, but the frontend's file picker and the tagger are
# only exercised against the common web formats, so the rest are turned away.
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

APIFRAME_BASE_URL = "https://api.apiframe.ai/v2"
POLL_EVERY_SECONDS = 5
GIVE_UP_AFTER_SECONDS = 300

# How long a single foreground request may spend actively checking on a
# Suno job before giving up and reporting "still generating" back to the
# frontend instead -- bounds every request that might touch apiframe so no
# HTTP request through the load balancer is ever held open for a whole
# render. Comfortably under Cloudflare's ~100s default origin timeout and
# whatever threshold was actually tripping Safari's NSURLErrorNetworkConnection
# Lost through the new GCP load balancer (see PLAN-generate-bounded-wait.md);
# a later poll (GET .../status, ~3s cadence) picks up from generate_job_id
# instead of this request waiting any longer.
STATUS_CHECK_SECONDS = 15

# Per-call network timeout used only by the bounded status check above --
# short enough that a single hung apiframe call can't by itself blow past
# STATUS_CHECK_SECONDS. apiframe_call's own default (30s) stays what
# submission and every other one-shot call use.
STATUS_CHECK_HTTP_TIMEOUT = 10

# apiframe's account only supports this many concurrent Suno jobs -- a 6th
# submission errors outright. This is account-wide, not per-instance, so it
# has to be enforced by the global queue in storage.py (mutate_queue et al.)
# rather than anything process-local like _generating_runs below, which
# only ever prevents one run_id from double-submitting, not five run_ids
# turning into six.
MAX_CONCURRENT_GENERATIONS = 5

# How long a waiting queue ticket can go unpolled before it's swept as
# abandoned (tab closed, network dropped, etc). Comfortably longer than the
# frontend's own ~3s poll interval so a couple of missed polls don't cost
# someone their place in line, short enough that a truly abandoned ticket
# doesn't block everyone behind it for long. Mirrors GENERATION_CLAIM_STALE_
# AFTER's role below, just for the "waiting" half of a ticket's life instead
# of the "active" half (which reuses GENERATION_CLAIM_STALE_AFTER itself,
# defined further down, rather than a second near-identical constant).
QUEUE_WAITING_STALE_AFTER = timedelta(seconds=20)

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

# One run list per visitor. In local dev (no SESSION_BUCKET) this is the
# real cache: _session_for loads a token's list from disk once, then trusts
# this dict for the rest of the process's life, safe only because local dev
# never has more than one process. In GCS mode _session_for and every write
# below always go straight to storage instead (see both their own
# docstrings) -- correctness there can't depend on which Cloud Run instance
# a request lands on, so this dict is simply never populated in that mode
# (see _update_local_cache). Each record:
#   {id, created_at, algorithm, image_source, tags, tag_scores, tagging,
#    prompt, tracks}
# where image_source is {"kind": "sample", "name": <str>} or
# {"kind": "upload", "filename": <str>, "ref": <str>, "url": <str>} (ref is
# what storage.py's delete_file takes back; url is what the frontend plays/
# displays directly). `tracks` stays None until /generate fills it in.
_sessions: dict[str, list[dict]] = {}

# Guards every read-modify-write against one visitor's slice of _sessions and
# its on-disk/GCS backing (check a cap, mutate a list, persist it) --
# without this, two concurrent requests for the same visitor (two tabs, or a
# double-click) can each load the same list, mutate their own copy, and
# save, with the second save silently discarding the first's run. Keyed per
# token rather than one lock for every visitor, so visitor A's slow GCS
# round trip never blocks visitor B's unrelated request in the same process.
# This still only protects one process/Cloud Run instance; it doesn't make
# save_session_records itself a distributed lock across instances, which
# per-visitor history doesn't have today.
_session_locks: dict[str, threading.Lock] = {}
# Guards only _session_locks' own creation of a new per-token Lock -- held
# far more briefly than any _session_locks[token] itself.
_session_locks_meta_lock = threading.Lock()

# run_ids with a currently-held generation claim, so a retry or double-click
# racing an in-flight Suno submission or finalize doesn't pay for a second
# one (see _submit_if_needed/_advance_generation) -- NOT "this run_id is
# somewhere between submit and done" the way it used to mean when one claim
# was held for an entire render; a run can now sit with a job submitted and
# rendering, nothing in this set, for most of its generation. Deliberately
# not a field on the session record itself: that dict is
# what storage.save_session_records pickles, and a mid-flight-only flag has
# no business being persisted (or briefly leaking into a save some other
# request triggers while /generate is still running).
#
# Process-local, same limitation as _sessions/_lock_for above -- safe today
# only because maxScale=1 means one process. In GCS mode generate() below
# claims run_id through storage.claim_generation instead, which is safe
# across instances; this set is the fallback for local dev, where there's
# never more than one process to race against.
_generating_runs: set[str] = set()

# Safety margin added on top of GIVE_UP_AFTER_SECONDS before a GCS
# generation claim is considered abandoned (see storage.claim_generation) --
# wider than the poll loop's own deadline so a legitimately slow-but-still-
# polling request is never mistaken for a crashed one. Also reused as-is for
# an *active* queue ticket's own staleness window (storage.mutate_queue's
# active_stale_after) -- same reasoning applies verbatim (a real generation
# never legitimately runs longer than this), so there's no reason for the
# queue to carry a second, separately-tuned constant for the same case.
GENERATION_CLAIM_STALE_AFTER = timedelta(seconds=GIVE_UP_AFTER_SECONDS + 60)


def _lock_for(token: str) -> threading.Lock:
    with _session_locks_meta_lock:
        if token not in _session_locks:
            _session_locks[token] = threading.Lock()
        return _session_locks[token]


def _claim_generation(token: str, run_id: str) -> bool:
    """True if this process may proceed with Suno submission for run_id;
    False if someone else already holds a live claim. Call while already
    holding `_lock_for(token)` -- in GCS mode the claim itself is atomic
    across instances via storage.claim_generation, but this still runs
    under the caller's lock so only one thread in this process attempts the
    GCS call for a given token at a time; in local dev the lock is what
    makes the plain set check-then-add atomic at all.
    """
    if storage.gcs_configured():
        return storage.claim_generation(token, run_id, GENERATION_CLAIM_STALE_AFTER)
    if run_id in _generating_runs:
        return False
    _generating_runs.add(run_id)
    return True


def _generation_in_progress(token: str, run_id: str) -> bool:
    """Read-only: does run_id currently hold a live generation claim on some
    instance right now? Used by delete_run to refuse deleting a run whose
    Suno submission or finalize step is actively in flight -- without this,
    a delete racing either of those could let Suno finish (or already have
    finished) and get paid for with no record left in the visitor's list to
    attach the result to (see AIE-001). Doesn't claim or release anything,
    unlike _claim_generation. Call while holding `_lock_for(token)`, same as
    _claim_generation, for local-mode consistency with generate()'s own set
    check.

    NOTE: since _submit_if_needed/_advance_generation split one long-held
    claim into two short ones (see PLAN-generate-bounded-wait.md), this no
    longer covers a run's entire time in flight -- a run with a submitted,
    still-rendering Suno job (generate_job_id set, no tracks/generate_error
    yet) holds no claim at all for most of its render, so this returns
    False for it then. delete_run's guard against deleting a truly in-flight
    run is narrower than it used to be as a result; widening it (e.g.
    treating "generate_job_id set and unresolved" as in-progress too)
    wasn't part of this change's scope, and is flagged as a finding rather
    than fixed here.
    """
    if storage.gcs_configured():
        return storage.generation_claimed(token, run_id, GENERATION_CLAIM_STALE_AFTER)
    return run_id in _generating_runs


def _release_generation_claim(token: str, run_id: str) -> None:
    """Undoes _claim_generation, on both the success and failure paths of
    the Suno call it guarded. Same locking expectation as _claim_generation."""
    if storage.gcs_configured():
        storage.release_generation_claim(token, run_id)
    else:
        _generating_runs.discard(run_id)


def _session_for(request: Request) -> tuple[str, list[dict]]:
    """This visitor's token and current run list.

    In GCS mode this always reads fresh from storage -- caching it here
    would mean a visitor's GET /history could serve whatever THIS instance
    happened to have cached on its first-ever touch of that visitor,
    forever, even after a different instance's write. mutate_session_records
    already never trusts this cache for writes (it does its own fresh read
    right before every write, which is what makes writes correct regardless
    of routing); reads need the same rule to get the same guarantee, not a
    separate one. In local dev (never more than one process, so nothing
    else can have written in between) the cache is safe and still used, so
    a session with many runs isn't re-read from disk on every request.
    """
    token = request.state.session_token
    if storage.gcs_configured():
        return token, storage.load_session_records(token) or []
    with _lock_for(token):
        if token not in _sessions:
            _sessions[token] = storage.load_session_records(token) or []
        return token, _sessions[token]


def _update_local_cache(token: str, records: list[dict]) -> None:
    """Keeps the local-dev cache in sync after a mutation. A no-op in GCS
    mode, where _session_for (above) never reads _sessions at all -- so
    there's nothing to keep in sync there, and no reason to grow this dict
    with entries nothing will ever read again (one entry per distinct
    visitor this process has ever seen, for the life of the process)."""
    if not storage.gcs_configured():
        _sessions[token] = records


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


@app.exception_handler(storage.SessionWriteConflict)
async def session_write_conflict_handler(request: Request, exc: storage.SessionWriteConflict):
    """Maps storage.py's own give-up-after-N-retries exception to a 503
    with a normal JSON `detail` body (matching every other error response
    here) instead of letting it fall through to FastAPI's generic
    unhandled-exception 500 -- this is a real, expected-under-heavy-
    contention outcome (see AIE-002), not a bug, so it's worth a message
    that tells the frontend to just retry rather than looking like a crash.
    """
    return JSONResponse(
        status_code=503,
        content={"detail": "This session is under heavy concurrent load right now -- please try again."},
    )


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


def apiframe_call(method: str, path: str, timeout: int = 30, **kwargs) -> dict:
    api_key = os.environ.get("APIFRAME_API_KEY")
    if not api_key:
        raise HTTPException(
            500, "APIFRAME_API_KEY is not set. Copy .env.example to .env and add it."
        )

    response = requests.request(
        method,
        f"{APIFRAME_BASE_URL}{path}",
        headers={"X-API-Key": api_key},
        timeout=timeout,
        **kwargs,
    )
    if not response.ok:
        # Pass apiframe's own message through; it explains things a bare status
        # code cannot, such as a v1 key being used against the v2 API.
        raise HTTPException(502, f"apiframe {response.status_code}: {response.text[:300]}")
    return response.json()


def submit_song(prompt: str) -> str:
    """Submits the prompt to Suno and returns the job id right away, without
    waiting for it to render.

    Deliberately split from checking on that job (_check_job_status below):
    submission and waiting now happen under separate, short-lived claims --
    possibly in separate HTTP requests, since the job id has to survive past
    the request that submitted it (see generate_job_id on the session
    record, and PLAN-generate-bounded-wait.md for why).
    """
    return apiframe_call(
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


def _check_job_status(job_id: str, what: str) -> dict | None:
    """Checks one apiframe job for up to STATUS_CHECK_SECONDS, sleeping
    POLL_EVERY_SECONDS between checks -- replaces the old wait_for_job's
    multi-minute blocking loop with one bounded enough that a single HTTP
    request can never be held open for a whole render.

    Returns the finished job dict once it reaches COMPLETED, raises
    HTTPException on FAILED (same contract wait_for_job used to have), or
    returns None if it's still pending when the bound elapses -- the caller
    reports {"status": "generating"} and a later poll (GET .../status)
    checks again from generate_job_id instead of this request waiting any
    longer. Read-only and side-effect-free, so callers don't need a claim
    just to call this.
    """
    deadline = time.monotonic() + STATUS_CHECK_SECONDS
    while True:
        job = apiframe_call("GET", f"/jobs/{job_id}", timeout=STATUS_CHECK_HTTP_TIMEOUT)
        status = job["status"]
        if status == "COMPLETED":
            return job
        if status == "FAILED":
            raise HTTPException(502, job.get("error") or f"Could not generate the {what}.")
        if time.monotonic() + POLL_EVERY_SECONDS >= deadline:
            return None
        time.sleep(POLL_EVERY_SECONDS)


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


def _delete_file_safe(ref: str, context: str) -> int | None:
    """storage.delete_file, but one ref failing (a transient GCS error)
    doesn't abort the rest of the caller's loop -- prints and returns None
    (as if the file were already gone) instead of raising."""
    try:
        return storage.delete_file(ref)
    except Exception as exc:
        print(f"could not delete {ref!r} during {context}: {exc}")
        return None


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
        # No-op for a record with no queue ticket (the common case: already
        # generated, or never even started) -- see delete_run's own version
        # of this for why leaving a wiped record's ticket behind would be a
        # problem (a slot nothing will ever actually claim once promoted).
        storage.release_queue_ticket(record["id"])
        refs = [track["ref"] for track in record["tracks"] or []]
        if record["image_source"]["kind"] == "upload":
            refs.append(record["image_source"]["ref"])
        for ref in refs:
            freed = _delete_file_safe(ref, "wipe_history")
            if freed is not None:
                bytes_freed += freed
                files_removed += 1

    # Unlike tag/delete_run/generate below, this doesn't need
    # mutate_session_records' conditional-write dance: a full wipe's result
    # is the same regardless of what else races it (delete-then-delete is a
    # no-op the second time; a concurrent tag() elsewhere that loses the
    # generation race against this delete just retries against the now-empty
    # state and correctly ends up as the only thing left) -- there's no
    # "lost write" for an operation whose whole point is "nothing survives."
    with _lock_for(token):
        _update_local_cache(token, [])
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
    visitor's run list doesn't have it.

    409s instead if run_id is currently mid-/generate (see
    _generation_in_progress) -- otherwise a delete racing a generate for
    the same run could let Suno finish and get paid for with nothing left
    in the visitor's list to attach the result to.
    """
    token, records = _session_for(request)
    record = find_session(records, run_id)
    if record is None:
        raise HTTPException(404, f"No run with id {run_id!r}.")

    with _lock_for(token):
        if _generation_in_progress(token, run_id):
            raise HTTPException(
                409,
                "This run is currently generating a song and can't be "
                "deleted until it finishes.",
            )

    # _generation_in_progress only rules out an ACTIVE claim -- a run_id can
    # still hold a merely-*waiting* queue ticket (queued, but not yet
    # promoted) at this point, since nothing else has 409'd on that. A
    # no-op if there's no ticket at all; without it a deleted-while-queued
    # run's ticket would sit until swept by the queue's own staleness
    # rules, needlessly occupying a slot (once promoted) that nothing will
    # ever actually claim.
    storage.release_queue_ticket(run_id)

    files_removed = 0
    bytes_freed = 0

    refs = [track["ref"] for track in record["tracks"] or []]
    if record["image_source"]["kind"] == "upload":
        refs.append(record["image_source"]["ref"])

    for ref in refs:
        freed = _delete_file_safe(ref, "delete_run")
        if freed is not None:
            bytes_freed += freed
            files_removed += 1

    # Removal by id rather than reusing the `record` object found above --
    # mutate_session_records may hand its mutator a fresher list (a
    # concurrent instance's write this process hasn't seen yet) whose
    # matching record isn't the same object. Idempotent if that fresher list
    # has already lost this id some other way (e.g. someone else deleted it
    # too), so a second racing delete_run for the same run_id is a no-op
    # here rather than a KeyError.
    def _remove(current: list[dict]) -> list[dict]:
        return [r for r in current if r["id"] != run_id]

    with _lock_for(token):
        _update_local_cache(token, storage.mutate_session_records(token, _remove))

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


@app.get("/test-image-thumbnails/{name}")
def test_image_thumbnail(name: str):
    """A small cached preview of one sample image -- what the picker grid
    and RunCard's thumb actually display; see thumbnails.py. `/tag` still
    reads the full-resolution original directly from TEST_IMAGES_DIR for
    the actual embedding, untouched by this route.

    Validated against the same known_sample_images() as `/tag`'s own
    sample_image check, so a name this route will serve is always one
    `/tag` will accept, and vice versa.
    """
    if name not in known_sample_images():
        raise HTTPException(status_code=404, detail=f"Unknown sample image: {name}")
    path = thumbnail_path(TEST_IMAGES_DIR / name)
    # Sample images never change under a fixed filename, so the browser can
    # cache the thumbnail indefinitely instead of re-requesting it every visit.
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=31536000, immutable"})


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
        # Set once submission actually happens (see _submit_if_needed) so a
        # later poll -- possibly a different request, possibly a different
        # Cloud Run instance -- can check on an in-flight Suno job instead of
        # resubmitting it. Initialized eagerly (unlike generate_error, which
        # is only ever added lazily on first failure) so callers can always
        # read it without a fallback default; existing records saved before
        # this field existed are still handled defensively via .get() below.
        "generate_job_id": None,
    }
    # Re-checked here, inside the lock and against mutate_session_records'
    # own fresh read, rather than trusting the early check above or this
    # process's cached `records` -- two concurrent /tag calls for the same
    # visitor (same instance, or, in GCS mode, a different one) can both
    # pass the early check before either has appended. This is the
    # authoritative one: raising from inside the mutator means it's judged
    # against whatever the storage layer just read, not a stale local copy.
    def _append(current: list[dict]) -> list[dict]:
        if len(current) >= MAX_SESSIONS:
            raise HTTPException(
                409,
                f"This session already holds {MAX_SESSIONS} runs. "
                "Delete one (or Clear) before starting another.",
            )
        return current + [record]

    with _lock_for(token):
        _update_local_cache(token, storage.mutate_session_records(token, _append))

    return {
        "id": run_id,
        "tags": tags,
        "tag_scores": tag_scores,
        "tagging": tagging,
        "prompt": prompt,
        "algorithm": algorithm,
        "image_source": image_source,
    }


def _generation_response(run_id: str, record: dict, job_id: str | None, tracks: list[dict]) -> dict:
    return {
        "id": run_id,
        "algorithm": record["algorithm"],
        "tags": record["tags"],
        "tag_scores": record["tag_scores"],
        "tagging": record["tagging"],
        "prompt": record["prompt"],
        "job_id": job_id,
        "tracks": tracks,
    }


def _fail_generation(token: str, run_id: str, detail: str) -> None:
    """Persists a transient generate_error on the session record (via
    mutate_session_records) and releases this run's queue ticket -- shared
    by both places _advance_generation can fail (the bounded status check
    coming back FAILED, or save_audio itself raising once a job's
    COMPLETED). Also clears generate_job_id: without that, a later retry
    would find this dead job's id still on the record and check on it again
    forever instead of ever submitting a fresh one.

    Doesn't touch a generation claim itself -- whether one's actually held
    at the point of failure differs between the two call sites (the status
    check runs claim-free; save_audio runs under the finalize claim), so
    each caller releases its own, if any, right after calling this.
    """
    def _record_error(current: list[dict]) -> list[dict]:
        return [
            {**r, "generate_error": detail, "generate_job_id": None} if r["id"] == run_id else r
            for r in current
        ]

    with _lock_for(token):
        _update_local_cache(token, storage.mutate_session_records(token, _record_error))
    storage.release_queue_ticket(run_id)


def _submit_if_needed(token: str, run_id: str, record: dict) -> str | None:
    """Ensures exactly one Suno submission exists for this run, then returns
    its job id -- or None if this call couldn't get the submission claim
    (someone else is already submitting) or the run vanished from under it.

    Claims only for the duration of the submission itself (see
    _claim_generation/_release_generation_claim), never the render that
    follows -- the render is checked separately, without holding anything,
    by _check_job_status. A caller that gets None back should treat that
    exactly like "still generating": no error, nothing to check yet.
    """
    if record.get("generate_job_id"):
        return record["generate_job_id"]

    with _lock_for(token):
        if not _claim_generation(token, run_id):
            return None

    # Under the claim now, so no other request can be submitting this run_id
    # concurrently -- but the claim doesn't itself prove `record` (read by
    # the caller before any of this ran) is still current. Re-read fresh in
    # both modes, not just GCS: a previous claim holder can submit and
    # release entirely in the gap between two of *this* call's own lock
    # acquisitions (the lock is only held for each short check-then-set, not
    # continuously across the whole claim), so a losing-then-winning
    # claimant's original `record` snapshot can predate that write even in
    # local dev, where storage.load_session_records still reads the on-disk
    # pickle straight through rather than trusting this process's cached
    # dict. Also catches a run deleted before this claim was ever attempted.
    fresh = storage.load_session_records(token) or []
    fresh_record = find_session(fresh, run_id)
    if fresh_record is None:
        with _lock_for(token):
            _release_generation_claim(token, run_id)
        storage.release_queue_ticket(run_id)
        raise HTTPException(404, f"No run with id {run_id!r}. It was deleted just now.")

    job_id = fresh_record.get("generate_job_id")
    if not job_id:
        # The one Suno-billing network call in this whole function --
        # everything else here is metadata bookkeeping, safe to repeat.
        # Must stay outside the mutate() below: mutate_session_records may
        # retry its mutator against a fresher read under GCS contention, and
        # a retried Suno submission would mean a second, unwanted job.
        job_id = submit_song(fresh_record["prompt"])

        def _persist_job_id(current: list[dict]) -> list[dict]:
            # `or job_id` rather than an unconditional overwrite: defensive
            # against a job id somehow already having landed here between
            # our fresh read above and this write, so this write can never
            # clobber it with our own (redundant, but harmless) submission.
            return [
                {**r, "generate_job_id": r.get("generate_job_id") or job_id} if r["id"] == run_id else r
                for r in current
            ]

        with _lock_for(token):
            _update_local_cache(token, storage.mutate_session_records(token, _persist_job_id))

    with _lock_for(token):
        _release_generation_claim(token, run_id)

    return job_id


def _advance_generation(token: str, run_id: str, record: dict) -> dict:
    """Moves one run's generation forward by at most STATUS_CHECK_SECONDS of
    actual waiting, then returns -- replaces the old _run_generation, which
    held one HTTP request open for the entire multi-minute render. Safe to
    call repeatedly, from different requests and even different instances,
    for as long as a run's queue ticket stays active:

    - No generate_job_id yet: try to claim submission (_submit_if_needed).
      Losing that race isn't an error -- it just means someone else is
      already submitting, so this call reports the same "still generating"
      a not-yet-done job would.
    - A job id (just-submitted, or already on the record from an earlier
      call): one bounded, claim-free status check. Still pending ->
      "generating", with the queue ticket and everything else left
      untouched, since the run legitimately isn't done. COMPLETED -> claim
      just this finalize step, download+save the audio, persist tracks,
      release the claim and the queue ticket, return the finished result.
      FAILED -> persist the error and release the queue ticket (via
      _fail_generation), then raise, same as the old failure path.
    """
    job_id = _submit_if_needed(token, run_id, record)
    if job_id is None:
        return {"status": "generating"}

    image_source = record["image_source"]
    original_filename = (
        image_source["filename"] if image_source["kind"] == "upload" else image_source["name"]
    )

    try:
        job = _check_job_status(job_id, "song")
    except Exception as exc:
        # HTTPException carries a real user-facing message in .detail; any
        # other exception's str() is the best we have.
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        _fail_generation(token, run_id, detail)
        raise

    if job is None:
        return {"status": "generating"}

    with _lock_for(token):
        if not _claim_generation(token, run_id):
            # Someone else is already finalizing this same COMPLETED job --
            # nothing to do here but report the same "still generating" any
            # other not-yet-done poll would.
            return {"status": "generating"}

    try:
        tracks = save_audio(token, run_id, original_filename, job, record["algorithm"])
    except Exception as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        _fail_generation(token, run_id, detail)
        with _lock_for(token):
            _release_generation_claim(token, run_id)
        raise

    # By id, through mutate_session_records, same reasoning as delete_run:
    # this instance's `records` may be behind another instance's writes, so
    # the field is set against mutate_session_records' own fresh read
    # rather than the in-memory `record` this function was handed -- those
    # other fields (prompt/algorithm/image_source) don't change mid-
    # generate, only `tracks` does. Also clears generate_error and
    # generate_job_id: belt-and-suspenders alongside /generate's own
    # clear-on-retry above, so a successful run never leaves a previous
    # attempt's failure message or dead job id sitting in the record.
    def _fill_tracks(current: list[dict]) -> list[dict]:
        return [
            {**r, "tracks": tracks, "generate_error": None, "generate_job_id": None}
            if r["id"] == run_id else r
            for r in current
        ]

    with _lock_for(token):
        _update_local_cache(token, storage.mutate_session_records(token, _fill_tracks))
        _release_generation_claim(token, run_id)
    storage.release_queue_ticket(run_id)

    return _generation_response(run_id, record, job_id, tracks)


@app.post("/generate")
def generate(request: Request, run_id: str):
    """Renders a song for a run `/tag` already produced -- or, if apiframe's
    MAX_CONCURRENT_GENERATIONS slots are all taken, joins the global queue
    and hands back `{"status": "queued", "position": N}` instead of
    touching apiframe at all. The frontend polls GET .../status from there.

    Even when a slot is free immediately, this no longer necessarily
    finishes the render before responding: it advances the run by at most
    STATUS_CHECK_SECONDS (see _advance_generation) and can come back
    `{"status": "generating"}` too, in which case the frontend falls back to
    the same GET .../status polling the queued path always used. This is
    what keeps a single request from ever being held open for a whole
    multi-minute render (see PLAN-generate-bounded-wait.md).

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

    # `tracks` already set means this run was already rendered: hand that
    # back rather than ever touching the queue for a run that's done. Same
    # guard /generate has always had against a retry/double-click after
    # completion; see _advance_generation/_submit_if_needed for the
    # still-in-flight case, which now reports {"status": "generating"}
    # rather than a 409.
    if record["tracks"]:
        return _generation_response(run_id, record, None, record["tracks"])

    # A retry of a run whose previous attempt failed still carries that
    # attempt's generate_error -- clear it before this new attempt is ever
    # enqueued, not after it finishes. GET .../status checks generate_error
    # before it ever looks at the queue (see below), so a poll landing
    # between this POST and the retry's own completion would otherwise
    # report the *previous* failure instead of this attempt's real
    # queued/generating progress.
    if record.get("generate_error"):
        def _clear_error(current: list[dict]) -> list[dict]:
            return [{**r, "generate_error": None} if r["id"] == run_id else r for r in current]

        with _lock_for(token):
            _update_local_cache(token, storage.mutate_session_records(token, _clear_error))
        record = {**record, "generate_error": None}

    # Enqueues this run_id (idempotent -- a retry/double-click for the same
    # run_id just refreshes its ticket rather than creating a second one)
    # and, in the same atomic mutation, sweeps stale tickets and promotes
    # oldest-waiting ones while there's room. Whether *this* run_id came out
    # the other end "active" is what decides the fast path below.
    tickets = storage.mutate_queue(
        run_id, token, QUEUE_WAITING_STALE_AFTER, GENERATION_CLAIM_STALE_AFTER, MAX_CONCURRENT_GENERATIONS
    )
    ticket = next((t for t in tickets if t["run_id"] == run_id), None)
    if ticket is None or ticket["status"] == "waiting":
        return {"status": "queued", "position": storage.ticket_position(tickets, run_id) or 0}

    # A slot was free immediately -- proceed through the same bounded-wait
    # path GET .../status uses once a ticket's promoted (_advance_generation
    # claims only its own short submission/finalize windows, rather than
    # this route holding one claim for the whole render the way it used to).
    # A concurrent double-click/retry that loses the submission race no
    # longer 409s here -- it comes back {"status": "generating"} same as any
    # other not-yet-done case, and the run vanishing out from under a claim
    # (GCS mode) is still a 404, just raised from inside _advance_generation
    # now instead of this route.
    return _advance_generation(token, run_id, record)


@app.get("/generate/{run_id}/status")
def generate_status(request: Request, run_id: str):
    """Polled by the frontend (every ~3s) while a /generate call came back
    `queued` or `generating` -- reports live queue position, and once the
    ticket's active, advances the run by one more bounded status check each
    poll (_advance_generation), until it eventually reports "done" or
    "error".

    All queue-wait and apiframe-submission work happens inside this
    foreground request (or inside the original POST's fast path above) and
    nowhere else -- Cloud Run only allocates CPU to an instance while it's
    actively handling *some* request, so a detached background thread with
    no request behind it could silently stall.
    """
    token, records = _session_for(request)
    record = find_session(records, run_id)
    if record is None:
        raise HTTPException(404, f"No run with id {run_id!r}. It may have been deleted, or the backend restarted since it was tagged.")

    # Checked before ever touching the queue: a finished (or failed) run's
    # ticket was already removed by _advance_generation's own release, and
    # mutate_queue below would otherwise treat "no ticket found" as "not
    # enqueued yet" and silently resurrect a done run as a fresh waiting
    # ticket.
    if record["tracks"]:
        return {"status": "done", **_generation_response(run_id, record, None, record["tracks"])}
    if record.get("generate_error"):
        return {"status": "error", "detail": record["generate_error"]}

    # Still pending: refresh this ticket's heartbeat and re-run the same
    # sweep-then-promote pass POST /generate's own enqueue uses -- this is
    # what lets a promotion discovered here (rather than in some other
    # request) actually get acted on, per the foreground-only constraint
    # above.
    tickets = storage.mutate_queue(
        run_id, token, QUEUE_WAITING_STALE_AFTER, GENERATION_CLAIM_STALE_AFTER, MAX_CONCURRENT_GENERATIONS
    )
    ticket = next((t for t in tickets if t["run_id"] == run_id), None)
    if ticket is None or ticket["status"] == "waiting":
        return {"status": "queued", "position": storage.ticket_position(tickets, run_id) or 0}

    # Newly (or already) active -- advance it by at most one bounded status
    # check (_advance_generation claims its own short submission/finalize
    # windows internally; a concurrent poll for the same run_id that loses
    # one of those races just gets "generating" back, same as this one
    # would). Only wrap the result in "done" when it actually is one --
    # "generating" already carries its own status.
    result = _advance_generation(token, run_id, record)
    if result.get("status") == "generating":
        return result
    return {"status": "done", **result}


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
