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


def mutate_session_records(token: str, mutate) -> list[dict]:
    """Reads this visitor's current run list, applies `mutate` (a function
    taking that list and returning the new list to persist), and writes the
    result back -- the one write path every caller that changes a visitor's
    session should go through instead of save_session_records directly.

    Locally (SESSION_BUCKET unset) that's a plain read-modify-write: always
    one process, so nothing else can have changed the file in between. In
    GCS mode it's optimistic concurrency instead: the blob's generation is
    read alongside its content, and the write is conditioned on that
    generation still being current (`if_generation_match`) so a second Cloud
    Run instance's concurrent write can't silently clobber this one. A
    conflict re-reads the now-current data and retries `mutate` against it,
    up to `_MAX_MUTATE_ATTEMPTS` times before giving up -- so `mutate` may
    run more than once and must be a pure function of the list it's given,
    with no side effects beyond building the new list (main.py's callers
    that also need to raise on some condition, e.g. the session cap, do so
    from inside `mutate` itself so that check is against the freshest data
    too, not whatever this process last cached).
    """
    bucket = _bucket()
    if bucket is None:
        records = load_session_records(token) or []
        new_records = mutate(records)
        save_session_records(token, new_records)
        return new_records

    from google.api_core.exceptions import PreconditionFailed

    name = _session_blob_name(token)
    for attempt in range(_MAX_MUTATE_ATTEMPTS):
        if attempt > 0:
            # Exponential-ish jitter: spreads out retries so two writers who
            # just collided don't immediately collide again on their very
            # next attempt.
            backoff = min(_MUTATE_BACKOFF_CAP_SECONDS, _MUTATE_BACKOFF_BASE_SECONDS * (2 ** attempt))
            time.sleep(random.uniform(0, backoff))

        blob = bucket.get_blob(name)
        if blob is None:
            generation, records = 0, []
        else:
            generation, records = blob.generation, pickle.loads(blob.download_as_bytes())

        new_records = mutate(records)
        data = pickle.dumps(new_records)
        try:
            bucket.blob(name).upload_from_string(data, if_generation_match=generation)
            return new_records
        except PreconditionFailed:
            continue  # someone else wrote first -- re-read and retry against their version

    raise SessionWriteConflict(
        f"Could not save session records after {_MAX_MUTATE_ATTEMPTS} attempts "
        "-- repeated concurrent writes to the same visitor's session."
    )


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
