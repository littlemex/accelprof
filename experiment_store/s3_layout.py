"""S3 layout + crash-safe, content-verified transfer for a run's artifacts.

Layout (the fixed part of the storage contract): every run's blobs live under

    s3://<bucket>/<alias>/<run_id>/<relative-path>

so a run is a self-contained prefix. GB profiler artifacts (`.nsys-rep`, `.ncu-rep`,
`.neff`/`.ntff`, `.pb` sets) are NOT MLflow artifacts — SageMaker-managed MLflow caps artifact
download at 200 MB (https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html)
— they live here and are referenced from the MLflow run by the ``artifacts_uri`` tag.

Upload integrity: ``upload_file`` runs with ``ChecksumAlgorithm=SHA256``, so boto3 computes a
per-part SHA256 and S3 validates it server-side, rejecting a corrupted part (the client raises on
that rejection) — for single-part AND multipart (GB) uploads. We then confirm the object landed:
for a
single-part object S3 returns the whole-file ``ChecksumSHA256`` and we compare it to the local
file; for a MULTIPART object S3 returns a *composite* checksum (``<base64>-<partcount>``) that is
deliberately not the whole-file hash, so we confirm by size and rely on the per-part validation
already done (a whole-file compare here would falsely fail every GB upload). The whole-file
sha256 (hex) is stored as object metadata for downstream/audit verification.

Reads: consumers mount the bucket (AWS S3 Files, full-POSIX NFS) and read artifacts in place —
see store.locate(). ``download_prefix`` is a fallback for non-mounted / cross-region contexts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .hashing import sha256_hex_and_b64_of_file

# boto3 error codes that will never succeed on retry — retrying just re-sends the bytes.
_NON_RETRYABLE = {"AccessDenied", "403", "InvalidAccessKeyId", "SignatureDoesNotMatch", "NoSuchBucket"}
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 0.5
_DELETE_BATCH = 1000  # S3 DeleteObjects hard cap per request
_SHA256_META_KEY = "sha256-hex"  # object metadata key carrying the whole-file digest
# A run's blobs are held from GC by a sidecar marker object at "<prefix><name>" (NOT an MLflow tag:
# a soft-deleted run's tags are gone, but its blobs — and this marker — persist). Written by
# store.hold(); honoured by the janitor's retention check.
RETENTION_MARKER = ".retention-keep"


@dataclass(frozen=True)
class ParsedS3Uri:
    bucket: str
    key: str  # prefix or object key; always without a leading slash


class UploadVerificationError(Exception):
    """A deterministic verification failure (checksum/size mismatch) — NOT retryable."""


def run_prefix(alias: str, run_id: str) -> str:
    """The S3 key prefix (no bucket, trailing slash) for one run's artifacts."""
    return f"{alias}/{run_id}/"


def artifacts_uri(bucket: str, alias: str, run_id: str) -> str:
    return f"s3://{bucket}/{run_prefix(alias, run_id)}"


def parse_s3_uri(uri: str) -> ParsedS3Uri:
    """Parse ``s3://bucket/key`` strictly. Rejects a query/fragment (a ``?``/``#`` is a legal S3
    key char but would be dropped by urlparse, silently pointing at a different object)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"not an s3:// URI: {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"s3 URI must not contain '?' or '#': {uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(f"malformed s3 URI (no bucket): {uri!r}")
    return ParsedS3Uri(bucket=bucket, key=key)


def _error_code(exc: Exception) -> str | None:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        return resp.get("Error", {}).get("Code")
    return None


def upload_verified(s3_client: Any, local_path: str, bucket: str, key: str, _sleep=None) -> str:
    """Upload one file with per-part checksum validation, then confirm it landed intact.

    Returns the whole-file sha256 (hex). Fails closed and NON-retryably on a deterministic
    verification mismatch (so a GB file is not re-sent 5x); retries only transient transport
    errors; fails fast on non-retryable AWS errors (AccessDenied, missing file, …).
    """
    import time as _time

    sleep = _sleep or _time.sleep
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)  # deterministic — never retry
    local_hex, local_b64 = sha256_hex_and_b64_of_file(local_path)  # single pass over the file
    local_size = os.path.getsize(local_path)

    last_err: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            s3_client.upload_file(
                local_path, bucket, key,
                ExtraArgs={"ChecksumAlgorithm": "SHA256", "Metadata": {_SHA256_META_KEY: local_hex}},
            )
            head = s3_client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
            remote = head.get("ChecksumSHA256")
            # Single-part object => whole-file checksum; compare exactly. Multipart => composite
            # "<b64>-<N>" (not the whole-file hash) => size-confirm (parts were validated on upload).
            if remote is not None and "-" not in remote:
                if remote != local_b64:
                    raise UploadVerificationError(f"checksum mismatch for {key}: {remote} != {local_b64}")
            elif head["ContentLength"] != local_size:
                raise UploadVerificationError(
                    f"size mismatch for {key}: {head['ContentLength']} != {local_size}")
            return local_hex
        except UploadVerificationError:
            raise  # deterministic; re-uploading won't help
        except Exception as e:  # noqa: BLE001
            if _error_code(e) in _NON_RETRYABLE:
                raise
            last_err = e
            if attempt < _MAX_ATTEMPTS:
                sleep(_BASE_DELAY_S * (2 ** (attempt - 1)))
    raise RuntimeError(f"upload failed after {_MAX_ATTEMPTS} attempts: {last_err}") from last_err


def _iter_files(path: str):
    """Yield (local_file, relative_path) for a file (relpath=basename) or a directory (recursive,
    relpaths preserved)."""
    if os.path.isdir(path):
        base = path.rstrip("/") or "/"
        for root, _dirs, files in os.walk(base):
            for name in files:
                fp = os.path.join(root, name)
                yield fp, os.path.relpath(fp, base)
    elif os.path.exists(path):
        yield path, os.path.basename(path)
    else:
        raise FileNotFoundError(path)


def upload_artifacts(s3_client: Any, bucket: str, alias: str, run_id: str,
                     artifacts: list[str] | None) -> dict[str, Any]:
    """Upload a run's artifacts (files and/or directories) under its prefix, preserving structure.

    Rejects two artifacts that map to the same destination key (a silent overwrite would report
    success while losing a file). An empty/None list is valid (accuracy-only run) → zero objects.
    Returns ``{"uri", "objects", "count"}``.
    """
    prefix = run_prefix(alias, run_id)
    planned: list[tuple[str, str]] = []  # (local_file, key)
    seen: dict[str, str] = {}
    for path in artifacts or []:
        for local_file, rel in _iter_files(path):
            key = prefix + rel.replace(os.sep, "/")
            if key in seen:
                raise ValueError(
                    f"artifact key collision: {seen[key]!r} and {local_file!r} both map to {key!r}")
            seen[key] = local_file
            planned.append((local_file, key))
    objects: list[str] = []
    for local_file, key in planned:
        upload_verified(s3_client, local_file, bucket, key)
        objects.append(key)
    return {"uri": artifacts_uri(bucket, alias, run_id), "objects": objects, "count": len(objects)}


def list_prefix(s3_client: Any, uri: str) -> list[str]:
    """List all object keys under an artifacts_uri prefix (handles pagination)."""
    parsed = parse_s3_uri(uri)
    keys: list[str] = []
    token = None
    while True:
        kw = {"Bucket": parsed.bucket, "Prefix": parsed.key}
        if token:
            kw["ContinuationToken"] = token
        resp = s3_client.list_objects_v2(**kw)
        keys.extend(o["Key"] for o in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def download_prefix(s3_client: Any, uri: str, dest_dir: str) -> list[str]:
    """FALLBACK (non-mounted / cross-region): copy a run's whole prefix to a local dir, preserving
    structure. Skips directory-placeholder keys (ending in ``/``) and guards against keys that
    would escape ``dest_dir`` via ``..`` (keys are effectively producer-controlled)."""
    parsed = parse_s3_uri(uri)
    dest_root = os.path.realpath(dest_dir)
    written: list[str] = []
    for key in list_prefix(s3_client, uri):
        rel = key[len(parsed.key):]
        if not rel or rel.endswith("/"):
            continue  # the prefix itself, or a nested directory placeholder
        local = os.path.realpath(os.path.join(dest_root, rel))
        if local != dest_root and not local.startswith(dest_root + os.sep):
            raise ValueError(f"refusing to write outside dest_dir (key {key!r} escapes via '..')")
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        s3_client.download_file(parsed.bucket, key, local)
        written.append(local)
    return written


def local_dir_for_mount(uri: str, mount_base: str, expected_bucket: str | None = None) -> str:
    """Map an artifacts_uri to the local directory where that bucket is mounted (S3 Files),
    WITHOUT copying: ``<mount_base>/<key-prefix>``. If ``expected_bucket`` is given, the URI's
    bucket must match it (the mount serves exactly one bucket), else raise — a run whose artifacts
    live in another region's bucket must NOT resolve to a plausible-but-wrong local path."""
    parsed = parse_s3_uri(uri)
    if expected_bucket is not None and parsed.bucket != expected_bucket:
        raise ValueError(
            f"artifacts are in bucket {parsed.bucket!r} but this mount serves {expected_bucket!r}; "
            f"use download() for cross-bucket/region access")
    return os.path.join(mount_base, parsed.key)


def _marker_key(uri: str) -> tuple[str, str]:
    parsed = parse_s3_uri(uri)
    if not parsed.key:
        raise ValueError(f"refusing a retention marker on an empty prefix: {uri!r}")
    return parsed.bucket, parsed.key.rstrip("/") + "/" + RETENTION_MARKER


def write_retention_marker(s3_client: Any, uri: str) -> str:
    """Place the GC hold marker under a run's prefix (idempotent). Returns the marker key."""
    bucket, key = _marker_key(uri)
    s3_client.put_object(Bucket=bucket, Key=key, Body=b"")
    return key


def has_retention_marker(s3_client: Any, uri: str) -> bool:
    """True if a run's prefix carries the GC hold marker."""
    bucket, key = _marker_key(uri)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:  # noqa: BLE001
        if _error_code(e) in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def delete_prefix(s3_client: Any, uri: str) -> int:
    """Delete every object under a run's prefix (used by the janitor / purge). Returns count.
    Requires s3:DeleteObject — deliberately NOT granted to the mcp-reader role. Raises if S3
    reports per-key errors (DeleteObjects returns 200 with a per-key ``Errors`` list), so a
    partial delete is never reported as success. Refuses an empty key (whole-bucket wipe guard)."""
    parsed = parse_s3_uri(uri)
    if not parsed.key:
        raise ValueError(f"refusing to delete an empty prefix (whole bucket): {uri!r}")
    keys = list_prefix(s3_client, uri)
    deleted = 0
    for i in range(0, len(keys), _DELETE_BATCH):
        batch = keys[i:i + _DELETE_BATCH]
        resp = s3_client.delete_objects(Bucket=parsed.bucket,
                                        Delete={"Objects": [{"Key": k} for k in batch]})
        errors = resp.get("Errors") or []
        if errors:
            raise RuntimeError(f"delete_objects reported {len(errors)} error(s), first: {errors[0]}")
        deleted += len(resp.get("Deleted", batch))
    return deleted
