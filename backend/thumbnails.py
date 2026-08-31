"""Generates and caches small preview thumbnails for the sample-image picker.

test-images/*.png are raw, full-camera-resolution PNGs (2-11MB each) meant
only for the actual tagging step -- the picker grid displays them at ~166px
and RunCard's thumb reuses the same image at 42px (see main.py/lib.ts), so
serving the original to the browser at either size wastes bandwidth for no
visible gain. Mirrors embedding_tagger.py's disk-cache pattern
(backend/vocab_cache/): compute once, cache to disk, never touch the
read-only source.

Generated lazily, on first request for each name, rather than eagerly at
startup -- unlike the model/vocabulary preload this pairs with, a Pillow
resize+encode is cheap enough (tens of milliseconds) that paying for it on
whichever images a visitor actually views beats spending cold-start time on
all 36 up front.
"""

import uuid
from pathlib import Path

from PIL import Image

THUMBNAIL_CACHE_DIR = Path(__file__).parent / "thumbnail_cache"
THUMBNAIL_MAX_EDGE = 640
THUMBNAIL_QUALITY = 85


def thumbnail_path(source_path: Path) -> Path:
    """The cached thumbnail for one sample image, generating it first if
    this is the first request for it. Always JPEG regardless of the
    source's own format -- smaller than PNG at this quality/size, and the
    picker never needs the source's format back.

    Written to a per-call temp file and renamed into place (Path.replace is
    atomic on the same filesystem) so two concurrent requests racing to
    generate the same not-yet-cached thumbnail can't leave a half-written
    file behind for a third request to read -- worst case both do the
    (cheap) resize work and the second rename just overwrites the first.
    """
    cache_path = THUMBNAIL_CACHE_DIR / f"{source_path.stem}.jpg"
    if cache_path.is_file():
        return cache_path

    THUMBNAIL_CACHE_DIR.mkdir(exist_ok=True)

    # convert("RGB") drops any alpha/palette channel -- same reasoning as
    # embedding_tagger._open_image -- and JPEG can't encode one anyway.
    image = Image.open(source_path).convert("RGB")
    image.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE))

    tmp_path = THUMBNAIL_CACHE_DIR / f"{source_path.stem}.{uuid.uuid4().hex}.tmp"
    image.save(tmp_path, format="JPEG", quality=THUMBNAIL_QUALITY)
    tmp_path.replace(cache_path)

    return cache_path
