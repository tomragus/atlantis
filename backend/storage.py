"""Persists per-visitor session records and their audio/image bytes.

Gated by the SESSION_BUCKET env var: unset, every function operates against a
per-token subdirectory under backend/history/, which is what keeps local dev
free of any GCP dependency -- the same no-op gating the sibling project's
session_store.py uses. Set, file bytes go to that bucket under <token>/... and
session records round-trip as one pickled blob per token under sessions/<token>.

A `ref` is an opaque string a caller stores and later passes back to
playback_url/delete_file -- locally it's a path relative to HISTORY_DIR; on
GCS it's the blob's object name (also <token>/...).
"""

import hashlib
import os
import pickle
from datetime import timedelta
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


def _bucket():
    """The configured bucket, or None when persistence is not configured."""
    name = os.environ.get(_BUCKET_ENV_VAR)
    if not name:
        return None
    from google.cloud import storage  # imported here: free when unconfigured

    return storage.Client().bucket(name)


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


def save_session_records(token: str, records: list[dict]) -> None:
    """Writes one visitor's whole run list, replacing whatever was there.
    Left to raise on failure rather than swallowing it -- a save that
    silently didn't happen would make the next restart lose the visitor's
    history with no warning that the guarantee had lapsed."""
    data = pickle.dumps(records)
    bucket = _bucket()
    if bucket is None:
        path = HISTORY_DIR / _namespace(token) / "sessions.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    bucket.blob(_SESSION_BLOB_PREFIX + _namespace(token)).upload_from_string(data)


def load_session_records(token: str) -> list[dict] | None:
    """The last persisted run list for this token, or None if there isn't one."""
    bucket = _bucket()
    if bucket is None:
        path = HISTORY_DIR / _namespace(token) / "sessions.pkl"
        if not path.is_file():
            return None
        return pickle.loads(path.read_bytes())
    blob = bucket.blob(_SESSION_BLOB_PREFIX + _namespace(token))
    if not blob.exists():
        return None
    return pickle.loads(blob.download_as_bytes())


def delete_session_records(token: str) -> None:
    """Removes this token's persisted run list, if one exists."""
    bucket = _bucket()
    if bucket is None:
        path = HISTORY_DIR / _namespace(token) / "sessions.pkl"
        if path.is_file():
            path.unlink()
        return
    blob = bucket.blob(_SESSION_BLOB_PREFIX + _namespace(token))
    if blob.exists():
        blob.delete()
