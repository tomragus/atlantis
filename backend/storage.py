"""Persists per-visitor session records and their audio/image bytes.

Gated by the SESSION_BUCKET env var: unset, every function operates against a
per-token subdirectory under backend/history/, which is what keeps local dev
free of any GCP dependency -- the same no-op gating the sibling project's
session_store.py uses. Set, file bytes go to that bucket under <token>/... and
session records round-trip as one pickled blob per token under sessions/<token>.

A `ref` is an opaque string a caller stores and later passes back to
playback_url/delete_file -- locally it's a path relative to HISTORY_DIR; on
GCS it's the blob's object name (also <token>/...).

Two things in here are specifically about surviving more than one Cloud Run
instance touching the same visitor at once (unset SESSION_BUCKET never sees
more than one process, so neither needs to do anything there):
`mutate_session_records` (a GCS-optimistic-concurrency read+write for a
visitor's run list) and `claim_generation`/`release_generation_claim` (a
GCS-conditional-write claim so two instances can't both submit the same
run's Suno job). main.py's own `_lock_for` still guards same-instance
races on top of these -- it's what these two are extending, not replacing.
"""

import hashlib
import os
import pickle
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "history"

_BUCKET_ENV_VAR = "SESSION_BUCKET"
_SESSION_BLOB_PREFIX = "sessions/"
# How long a signed playback URL stays valid. Generated once, at the point a
# track/image is saved, and held as-is in the session record from then on --
# see main.py's Open Questions about what that means for a record read back
# well after this window closes.
_SIGNED_URL_EXPIRATION = timedelta(hours=12)


def _namespace(token: str) -> str:
    """Turns a session token into the path/object-name component every
    other function in this module uses.

    The token is whatever the client's session_id cookie says (see
    main.py's session_cookie_middleware) -- trusted for identity, but never
    for shape, since a client can send any cookie value it wants, including
    one containing "../". Hashing it here, at this module's one entry point
    for turning a token into a filesystem path or GCS object name, closes
    that off regardless of what the cookie contains, rather than relying on
    every call site (or every future one) to separately validate it.
    """
    return hashlib.sha256(token.encode()).hexdigest()


@lru_cache(maxsize=1)
def _gcs_client():
    """The GCS Client, or None when persistence is not configured.

    Built once and reused, same one-shot-cache idiom as main.py's
    known_sample_images(): a Client opens its own HTTP connection pool, and
    every route that touches storage was calling this per file/ref (up to
    ~60 times on one /history GET), rebuilding that pool each time for no
    reason.
    """
    if not os.environ.get(_BUCKET_ENV_VAR):
        return None
    from google.cloud import storage  # imported here: free when unconfigured

    return storage.Client()


def _bucket():
    """The configured bucket, or None when persistence is not configured."""
    client = _gcs_client()
    if client is None:
        return None
    return client.bucket(os.environ[_BUCKET_ENV_VAR])


def gcs_configured() -> bool:
    """Whether SESSION_BUCKET is set -- lets main.py pick between the
    cross-instance-safe GCS path and the process-local fallback (the
    in-memory _generating_runs set) without reaching into this module's
    private _bucket()."""
    return _bucket() is not None


def put_file(token: str, subpath: str, data: bytes) -> str:
    """Stores one file's bytes under this visitor's namespace, returning a ref
    to hand back to playback_url/delete_file later."""
    ref = f"{_namespace(token)}/{subpath}"
    bucket = _bucket()
    if bucket is None:
        path = HISTORY_DIR / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ref
    bucket.blob(ref).upload_from_string(data)
    return ref


@lru_cache(maxsize=1)
def _signing_credentials():
    """Credentials generate_signed_url can actually sign with.

    Cloud Run's attached service account carries only a short-lived OAuth
    token, never a private key, so the plain credentials google.auth.default()
    returns raise "you need a private key to sign credentials" the moment
    generate_signed_url tries to use them -- confirmed live, this is not
    theoretical. The fix is self-impersonation: ask the IAM Credentials API
    to sign on behalf of the same service account, which is exactly what the
    iam.serviceAccountTokenCreator binding granted to that account (on
    itself) during deployment is for.

    Built once and reused rather than re-derived on every playback_url call:
    the object itself renews its own token as needed, so there's nothing to
    gain by rebuilding it (and its own google.auth.default() + refresh()
    network round trip) on every one of the up-to-~60 calls one /history GET
    can make.
    """
    import google.auth
    from google.auth import impersonated_credentials
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default()
    credentials.refresh(Request())  # populates service_account_email
    return impersonated_credentials.Credentials(
        source_credentials=credentials,
        target_principal=credentials.service_account_email,
        target_scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
        lifetime=300,
    )


def playback_url(ref: str) -> str:
    """A URL the frontend can play or display directly."""
    bucket = _bucket()
    if bucket is None:
        return f"/files/{ref}"
    return bucket.blob(ref).generate_signed_url(
        version="v4",
        expiration=_SIGNED_URL_EXPIRATION,
        method="GET",
        credentials=_signing_credentials(),
    )


def delete_file(ref: str) -> int | None:
    """Deletes the file this ref points to and returns the bytes freed, or
    None if it was already gone."""
    bucket = _bucket()
    if bucket is None:
        path = HISTORY_DIR / ref
        if not path.is_file():
            return None
        size = path.stat().st_size
        path.unlink()
        return size
    # get_blob (rather than blob()+exists()) does one request that returns
    # None for a missing object and, for a real one, a Blob with metadata
    # (including .size) already populated -- exists() alone only fetches the
    # "name" field and would leave .size unset.
    blob = bucket.get_blob(ref)
    if blob is None:
        return None
    size = blob.size
    blob.delete()
    return size


def _session_path(token: str) -> Path:
    return HISTORY_DIR / _namespace(token) / "sessions.pkl"


def _session_blob_name(token: str) -> str:
    return _SESSION_BLOB_PREFIX + _namespace(token)


def save_session_records(token: str, records: list[dict]) -> None:
    """Writes one visitor's whole run list, replacing whatever was there.
    Left to raise on failure rather than swallowing it -- a save that
    silently didn't happen would make the next restart lose the visitor's
    history with no warning that the guarantee had lapsed."""
    data = pickle.dumps(records)
    bucket = _bucket()
    if bucket is None:
        path = _session_path(token)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    bucket.blob(_session_blob_name(token)).upload_from_string(data)


def load_session_records(token: str) -> list[dict] | None:
    """The last persisted run list for this token, or None if there isn't one."""
    bucket = _bucket()
    if bucket is None:
        path = _session_path(token)
        if not path.is_file():
            return None
        return pickle.loads(path.read_bytes())
    blob = bucket.blob(_session_blob_name(token))
    if not blob.exists():
        return None
    return pickle.loads(blob.download_as_bytes())


class SessionWriteConflict(RuntimeError):
    """mutate_session_records couldn't land a write after repeated GCS
    generation conflicts -- main.py maps this to a 503 (see its
    @app.exception_handler) rather than letting it surface as a bare 500,
    since this is a real, expected-under-contention outcome, not a bug."""


# 10 attempts with jittered backoff between them, not a bare retry loop:
# a visitor with several tabs open, each firing a request around the same
# moment, would otherwise burn through a small fixed attempt count fast
# (confirmed empirically -- 20 concurrent writers exhausted 5 attempts with
# no backoff). The jitter spreads out retries so simultaneous losers don't
# immediately collide again on their very next attempt.
_MAX_MUTATE_ATTEMPTS = 10
_MUTATE_BACKOFF_BASE_SECONDS = 0.02
_MUTATE_BACKOFF_CAP_SECONDS = 0.5

# Guards local-mode pickled-blob writes against real thread concurrency --
# FastAPI/uvicorn runs each sync `def` route handler in its own worker
# thread even in local dev (see main.py's routes), so two requests landing
# at once could otherwise race a plain read-modify-write. Session records
# never needed this: main.py's own per-token _lock_for already serializes
# same-token writes, and different tokens' records live in different files
# with no shared state to race over. The queue (below) has no such
# upstream lock -- it's one global blob every visitor's requests write to
# -- so this single lock is what keeps _mutate_pickled_blob's local branch
# safe for both use cases rather than asking main.py to invent a
# queue-specific lock for the same job. GCS mode doesn't need it: its
# branch is already safe via if_generation_match, the real cross-instance
# guarantee this is only standing in for locally.
_local_mutate_lock = threading.Lock()


def _mutate_pickled_blob(local_path: Path, blob_name: str, mutate):
    """The generic "read a blob's generation, apply a pure mutator, write
    back conditioned on that generation, retry with jittered backoff on a
    losing race" shape -- factored out of what was originally
    mutate_session_records's own body so the new global queue store
    (mutate_queue, below) can reuse the exact same cross-instance-safety
    logic instead of carrying a second copy of it. Every caller's `mutate`
    must be a pure function of the list it's given, for the same reason
    mutate_session_records' own docstring has always explained: it may run
    more than once per call, against whatever the freshest read turned up,
    if an earlier attempt lost a generation race.

    Local mode (SESSION_BUCKET unset) is a plain read-modify-write under
    _local_mutate_lock; GCS mode is the optimistic-concurrency loop --
    `if_generation_match` on the write means a second Cloud Run instance's
    concurrent write can't silently clobber this one, and a losing
    PreconditionFailed re-reads and retries `mutate` against the fresher
    data, up to _MAX_MUTATE_ATTEMPTS times before giving up.
    """
    bucket = _bucket()
    if bucket is None:
        with _local_mutate_lock:
            if not local_path.is_file():
                records = []
            else:
                records = pickle.loads(local_path.read_bytes())
            new_records = mutate(records)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(pickle.dumps(new_records))
            return new_records

    from google.api_core.exceptions import PreconditionFailed

    for attempt in range(_MAX_MUTATE_ATTEMPTS):
        if attempt > 0:
            # Exponential-ish jitter: spreads out retries so two writers who
            # just collided don't immediately collide again on their very
            # next attempt.
            backoff = min(_MUTATE_BACKOFF_CAP_SECONDS, _MUTATE_BACKOFF_BASE_SECONDS * (2 ** attempt))
            time.sleep(random.uniform(0, backoff))

        blob = bucket.get_blob(blob_name)
        if blob is None:
            generation, records = 0, []
        else:
            generation, records = blob.generation, pickle.loads(blob.download_as_bytes())

        new_records = mutate(records)
        data = pickle.dumps(new_records)
        try:
            bucket.blob(blob_name).upload_from_string(data, if_generation_match=generation)
            return new_records
        except PreconditionFailed:
            continue  # someone else wrote first -- re-read and retry against their version

    raise SessionWriteConflict(
        f"Could not save {blob_name!r} after {_MAX_MUTATE_ATTEMPTS} attempts "
        "-- repeated concurrent writes to the same blob."
    )


def mutate_session_records(token: str, mutate) -> list[dict]:
    """Reads this visitor's current run list, applies `mutate` (a function
    taking that list and returning the new list to persist), and writes the
    result back -- the one write path every caller that changes a visitor's
    session should go through instead of save_session_records directly.

    Thin wrapper around _mutate_pickled_blob -- see that function for the
    actual local-vs-GCS mechanics, and its docstring for why `mutate` must
    be pure (main.py's callers that also need to raise on some condition,
    e.g. the session cap, do so from inside `mutate` itself so that check
    is against the freshest data too, not whatever this process last
    cached).
    """
    return _mutate_pickled_blob(_session_path(token), _session_blob_name(token), mutate)


def delete_session_records(token: str) -> None:
    """Removes this token's persisted run list, if one exists."""
    bucket = _bucket()
    if bucket is None:
        path = _session_path(token)
        if path.is_file():
            path.unlink()
        return
    blob = bucket.blob(_session_blob_name(token))
    if blob.exists():
        blob.delete()


_CLAIM_BLOB_PREFIX = "generating/"


def _claim_blob_name(token: str, run_id: str) -> str:
    return f"{_CLAIM_BLOB_PREFIX}{_namespace(token)}/{run_id}"


def claim_generation(token: str, run_id: str, stale_after: timedelta) -> bool:
    """Attempts to claim `run_id` for this instance's Suno submission.
    True if the claim succeeded (no other instance is currently mid-
    /generate for this run); False if someone else already holds a live
    one. GCS mode only -- main.py falls back to the process-local
    `_generating_runs` set locally, where there's never more than one
    process to race against.

    A conditional create (`if_generation_match=0`, GCS's "only if this
    object doesn't exist yet") is what makes "does a claim already exist"
    atomic across instances, the same way the local set's check-then-add is
    atomic within one process under main.py's `_lock_for`.

    A claim older than `stale_after` (main.py passes GIVE_UP_AFTER_SECONDS
    plus a safety margin) is treated as abandoned -- the instance that made
    it was recycled mid-request (a Cloud Run kill, a crash) before its own
    try/finally cleanup ran -- and is overwritten rather than blocking that
    run_id from ever generating again.
    """
    from google.api_core.exceptions import PreconditionFailed

    bucket = _bucket()
    if bucket is None:
        raise RuntimeError("claim_generation requires SESSION_BUCKET (GCS mode) -- "
                            "callers should check storage.gcs_configured() first")
    name = _claim_blob_name(token, run_id)
    now = datetime.now(timezone.utc)
    payload = now.isoformat().encode()

    try:
        bucket.blob(name).upload_from_string(payload, if_generation_match=0)
        return True
    except PreconditionFailed:
        pass  # a claim already exists -- stale, or still live?

    existing = bucket.get_blob(name)
    if existing is None:
        # Whoever held it released it between our failed create and this
        # read -- one more attempt at a fresh create.
        try:
            bucket.blob(name).upload_from_string(payload, if_generation_match=0)
            return True
        except PreconditionFailed:
            return False

    if _claim_is_live(existing, now, stale_after):
        return False  # still live -- another instance is genuinely mid-/generate

    # Stale: reclaim it, but conditioned on the generation we just read, so
    # two instances racing to reclaim the same stale claim can't both win.
    try:
        bucket.blob(name).upload_from_string(payload, if_generation_match=existing.generation)
        return True
    except PreconditionFailed:
        return False


def _claim_is_live(claim_blob, now: datetime, stale_after: timedelta) -> bool:
    claimed_at = datetime.fromisoformat(claim_blob.download_as_text())
    return now - claimed_at <= stale_after


def generation_claimed(token: str, run_id: str, stale_after: timedelta) -> bool:
    """Read-only: True if a live (non-stale) generation claim currently
    exists for run_id -- doesn't create, reclaim, or otherwise modify
    anything, unlike claim_generation. For delete_run to refuse deleting a
    run that's genuinely mid-/generate on some instance right now, so that
    run's eventual Suno result doesn't end up paid for with no record left
    to attach it to."""
    bucket = _bucket()
    if bucket is None:
        raise RuntimeError("generation_claimed requires SESSION_BUCKET (GCS mode) -- "
                            "callers should check storage.gcs_configured() first")
    existing = bucket.get_blob(_claim_blob_name(token, run_id))
    if existing is None:
        return False
    return _claim_is_live(existing, datetime.now(timezone.utc), stale_after)


def release_generation_claim(token: str, run_id: str) -> None:
    """Clears a claim made by claim_generation, on both the success and
    failure paths -- best-effort, mirroring _delete_file_safe in main.py: a
    claim that can't be deleted (a transient GCS error, or the process
    being killed before this runs) just sits until claim_generation's
    staleness check reclaims it later, rather than raising here and
    replacing the caller's own real result/error.
    """
    bucket = _bucket()
    if bucket is None:
        return
    try:
        bucket.blob(_claim_blob_name(token, run_id)).delete()
    except Exception as exc:
        print(f"could not release generation claim for {run_id!r}: {exc}")


# --- Global generation queue ---------------------------------------------
#
# apiframe's account only supports a handful of concurrent Suno jobs at
# once (main.py's MAX_CONCURRENT_GENERATIONS), and that cap is account-wide,
# not per-instance -- so gating it needs one global ticket list every
# visitor's /generate request shares, unlike everything else in this module
# (which is scoped per visitor token). Unlike claim_generation, there's no
# local-dev-only shortcut here: even a single local process can have more
# than a handful of requests concurrently mid-/generate (uvicorn runs each
# sync route handler in its own thread), so the same storage-backed ticket
# list is used in both modes -- see _mutate_pickled_blob above for the
# local-file-vs-GCS split every function below goes through.
#
# One dict per ticket: run_id, token (who it belongs to, for bookkeeping --
# nothing here reads it back), joined_at (when it was first enqueued,
# what waiting position is ranked by), last_seen (heartbeat, refreshed by
# every status poll -- what the waiting-staleness sweep checks), status
# ("waiting" or "active"), claimed_at (when it was promoted to active --
# what the active-staleness sweep checks, None while still waiting).

_QUEUE_BLOB_NAME = "queue/state"
_QUEUE_LOCAL_PATH = HISTORY_DIR / "_queue.pkl"


def _load_queue() -> list[dict]:
    """The current ticket list, read-only. Never use this as the basis for
    a write -- that's what mutate_queue's own fresh read-under-retry is
    for; a write built off this instead could clobber a concurrent change."""
    bucket = _bucket()
    if bucket is None:
        if not _QUEUE_LOCAL_PATH.is_file():
            return []
        return pickle.loads(_QUEUE_LOCAL_PATH.read_bytes())
    blob = bucket.blob(_QUEUE_BLOB_NAME)
    if not blob.exists():
        return []
    return pickle.loads(blob.download_as_bytes())


def _sweep_and_promote(
    tickets: list[dict],
    now: datetime,
    waiting_stale_after: timedelta,
    active_stale_after: timedelta,
    max_concurrent: int,
) -> list[dict]:
    """Pure: drops tickets stale by either rule, then promotes
    oldest-waiting tickets to active while there's room under
    max_concurrent. Split out from mutate_queue below so its enqueue path
    and its touch-only path both run the exact same sweep-then-promote
    rule instead of two near-copies of it.

    A waiting ticket goes stale if it hasn't been polled (last_seen
    refreshed) within waiting_stale_after -- the tab-closed/network-dropped
    case. An active ticket goes stale if it's been claimed_at longer than
    active_stale_after without finishing -- the crash-recovery case, the
    same reasoning as main.py's GENERATION_CLAIM_STALE_AFTER for
    claim_generation: a real generation should never legitimately run
    longer than that, so anything still active past it was left behind by
    an instance that got recycled mid-request rather than one still
    genuinely working.
    """
    survivors = [
        ticket
        for ticket in tickets
        if (
            now - ticket["last_seen"] <= waiting_stale_after
            if ticket["status"] == "waiting"
            else now - ticket["claimed_at"] <= active_stale_after
        )
    ]

    active_count = sum(1 for ticket in survivors if ticket["status"] == "active")
    room = max(0, max_concurrent - active_count)
    waiting = sorted(
        (ticket for ticket in survivors if ticket["status"] == "waiting"),
        key=lambda ticket: ticket["joined_at"],
    )
    promoted_ids = {ticket["run_id"] for ticket in waiting[:room]}

    return [
        {**ticket, "status": "active", "claimed_at": now}
        if ticket["run_id"] in promoted_ids
        else ticket
        for ticket in survivors
    ]


def mutate_queue(
    run_id: str,
    token: str,
    waiting_stale_after: timedelta,
    active_stale_after: timedelta,
    max_concurrent: int,
) -> list[dict]:
    """The one entry point for advancing the global generation queue --
    main.py calls this both to enqueue a brand-new run_id (POST /generate)
    and to refresh an existing ticket's heartbeat while checking on it
    (GET /generate/{run_id}/status). Both cases share one code path
    because "touch this ticket, creating it if it isn't there yet" is the
    same operation either way, and both need the same sweep-then-promote
    pass to run immediately afterward, atomically, so nobody can observe a
    state where a slot freed by the sweep hasn't yet been offered to the
    oldest waiter.

    Enqueueing is idempotent: a retry or double-click for a run_id that
    already has a ticket just refreshes its last_seen rather than creating
    a second one. Callers must not call this for a run_id whose generation
    has already finished (main.py checks the session record's
    tracks/generate_error fields before ever reaching here) -- a finished
    run's ticket is deliberately removed by release_queue_ticket, and this
    function has no way to distinguish "never enqueued" from "already
    finished and released," so calling it after release would silently
    re-enqueue a done run as a fresh waiting ticket.

    Goes through _mutate_pickled_blob, so this is atomic the same way
    mutate_session_records is: two callers racing each other never both
    see room for the same slot, because the second one's write is
    conditioned on the generation the first one's write just advanced
    past.
    """

    def _mutate(tickets: list[dict]) -> list[dict]:
        now = datetime.now(timezone.utc)
        found = False
        touched = []
        for ticket in tickets:
            if ticket["run_id"] == run_id:
                found = True
                ticket = {**ticket, "last_seen": now}
            touched.append(ticket)
        if not found:
            touched.append(
                {
                    "run_id": run_id,
                    "token": token,
                    "joined_at": now,
                    "last_seen": now,
                    "status": "waiting",
                    "claimed_at": None,
                }
            )
        return _sweep_and_promote(touched, now, waiting_stale_after, active_stale_after, max_concurrent)

    return _mutate_pickled_blob(_QUEUE_LOCAL_PATH, _QUEUE_BLOB_NAME, _mutate)


def release_queue_ticket(run_id: str) -> None:
    """Removes run_id's ticket, freeing its slot -- called once
    _run_generation stops actually running (success or failure) so the
    slot doesn't sit reserved until some future poll's staleness sweep
    notices it. Doesn't itself promote the next-in-line: the next
    mutate_queue call (that visitor's own next poll, or any other
    request's queue touch) does its own sweep-and-promote pass and picks
    up the now-free slot then -- consistent with every queue-wait and
    apiframe-submission step in this feature only ever running inside a
    foreground request, never a background task with nothing driving it.
    """

    def _mutate(tickets: list[dict]) -> list[dict]:
        return [ticket for ticket in tickets if ticket["run_id"] != run_id]

    _mutate_pickled_blob(_QUEUE_LOCAL_PATH, _QUEUE_BLOB_NAME, _mutate)


def ticket_position(tickets: list[dict], run_id: str) -> int | None:
    """Where run_id ranks among the currently-waiting tickets in this list
    (0 = next up), or None if it isn't currently waiting -- already
    active, or has no ticket at all. Takes a ticket list rather than
    reading storage itself so a caller that just got a fresh list back
    from mutate_queue can rank against that exact state instead of paying
    for (and risking a race against) a second read."""
    waiting = sorted(
        (ticket for ticket in tickets if ticket["status"] == "waiting"),
        key=lambda ticket: ticket["joined_at"],
    )
    for index, ticket in enumerate(waiting):
        if ticket["run_id"] == run_id:
            return index
    return None


def queue_position(run_id: str) -> int | None:
    """Convenience wrapper around ticket_position for a caller that
    doesn't already have a fresh ticket list in hand."""
    return ticket_position(_load_queue(), run_id)
