"""Cache slow model results on disk ("stubs") so reruns skip inference."""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def stub_key(video_path: str | Path, *parts: Any) -> str:
    """Build a cache file name from the video and the settings that affect results."""
    video_path = Path(video_path)
    stat = video_path.stat()
    raw = "|".join([video_path.name, str(stat.st_size), str(int(stat.st_mtime))] + [str(p) for p in parts])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{video_path.stem}_{digest}.pkl"


def save_stub(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def read_stub(path: str | Path, expected_length: Optional[int] = None) -> Optional[Any]:
    """Return the cached object, or None when missing, unreadable, or the wrong length.

    Only read stubs you created yourself: pickle files can run code when loaded.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
    except Exception as error:  # corrupted or incompatible cache
        logger.warning("Ignoring unreadable stub %s: %s", path, error)
        return None
    if expected_length is not None and hasattr(obj, "__len__") and len(obj) != expected_length:
        logger.warning("Ignoring stub %s: length %d != %d", path, len(obj), expected_length)
        return None
    return obj
