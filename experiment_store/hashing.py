"""Content-hash helpers for S3 upload verification (used only by s3_layout)."""
from __future__ import annotations

import base64
import hashlib

_CHUNK_BYTES = 1 << 20


def sha256_hex_and_b64_of_file(path: str) -> tuple[str, str]:
    """Return ``(hex, base64)`` of the file's sha256 computed in a SINGLE pass.

    hex is stored as object metadata (audit / whole-file verification of a multipart object whose
    server-side ``ChecksumSHA256`` is composite, not whole-file); base64 is the form S3 returns in
    ``ChecksumSHA256`` for a single-part object, compared directly to the local file. Deriving both
    from one digest avoids two full reads of a multi-GB artifact per upload.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            h.update(chunk)
    digest = h.digest()
    return digest.hex(), base64.b64encode(digest).decode()
