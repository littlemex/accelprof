"""Unit tests for the S3 layout + verified transfer, using moto (real boto3 calls, mocked S3)."""
from __future__ import annotations

import os

import boto3
import pytest

moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

from . import s3_layout as L  # noqa: E402

BUCKET = "mcp-traces-test"
REGION = "us-east-1"


def _bucket():
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    return s3


def test_run_prefix_and_uri():
    assert L.run_prefix("myalias", "run123") == "myalias/run123/"
    assert L.artifacts_uri(BUCKET, "myalias", "run123") == f"s3://{BUCKET}/myalias/run123/"


@pytest.mark.parametrize("uri", ["http://x/y", "s3:///k", "s3://b/k?q=1", "s3://b/k#f"])
def test_parse_s3_uri_rejects_bad(uri):
    with pytest.raises(ValueError):
        L.parse_s3_uri(uri)


@mock_aws
def test_upload_artifacts_files_and_dirs_preserve_structure(tmp_path):
    s3 = _bucket()
    (tmp_path / "trace.nsys-rep").write_bytes(b"NSYS" * 100)
    sub = tmp_path / "profile_output"
    sub.mkdir()
    (sub / "model.neff").write_bytes(b"NEFF")
    (sub / "run.ntff").write_bytes(b"NTFF")

    out = L.upload_artifacts(s3, BUCKET, "alias1", "runX",
                             [str(tmp_path / "trace.nsys-rep"), str(sub)])
    assert out["uri"] == f"s3://{BUCKET}/alias1/runX/"
    assert out["count"] == 3
    keys = set(L.list_prefix(s3, out["uri"]))
    assert keys == {
        "alias1/runX/trace.nsys-rep",
        "alias1/runX/model.neff",
        "alias1/runX/run.ntff",
    }


@mock_aws
def test_upload_artifacts_empty_is_valid(tmp_path):
    s3 = _bucket()
    out = L.upload_artifacts(s3, BUCKET, "alias1", "runX", None)
    assert out["count"] == 0
    assert out["uri"] == f"s3://{BUCKET}/alias1/runX/"


@mock_aws
def test_download_prefix_roundtrip(tmp_path):
    s3 = _bucket()
    sub = tmp_path / "in" / "profile_output"
    sub.mkdir(parents=True)
    (sub / "model.neff").write_bytes(b"NEFF-bytes")
    out = L.upload_artifacts(s3, BUCKET, "a", "r", [str(sub)])
    dest = tmp_path / "out"
    written = L.download_prefix(s3, out["uri"], str(dest))
    assert (dest / "model.neff").read_bytes() == b"NEFF-bytes"
    assert len(written) == 1


def test_local_dir_for_mount():
    d = L.local_dir_for_mount(f"s3://{BUCKET}/alias1/runX/", "/mnt/traces")
    assert d == "/mnt/traces/alias1/runX/"


@mock_aws
def test_delete_prefix(tmp_path):
    s3 = _bucket()
    (tmp_path / "f1").write_bytes(b"a")
    (tmp_path / "f2").write_bytes(b"b")
    out = L.upload_artifacts(s3, BUCKET, "a", "r", [str(tmp_path / "f1"), str(tmp_path / "f2")])
    assert L.delete_prefix(s3, out["uri"]) == 2
    assert L.list_prefix(s3, out["uri"]) == []


@mock_aws
def test_upload_verified_fails_closed_on_access_denied(tmp_path, monkeypatch):
    s3 = _bucket()
    f = tmp_path / "x"
    f.write_bytes(b"data")

    class Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    def boom(*a, **k):
        raise Denied()

    monkeypatch.setattr(s3, "upload_file", boom)
    with pytest.raises(Denied):
        L.upload_verified(s3, str(f), BUCKET, "k", _sleep=lambda s: None)


@mock_aws
def test_upload_verified_accepts_multipart_composite_checksum(tmp_path, monkeypatch):
    """A multipart object's ChecksumSHA256 is '<b64>-<N>' (NOT the whole-file hash). We must
    accept it via size-confirm and NOT retry (the GB-file blocker) — regression for B1/H1."""
    s3 = _bucket()
    f = tmp_path / "big"
    f.write_bytes(b"x" * 4096)

    calls = {"upload": 0, "sleep": 0}
    real_upload = s3.upload_file

    def counting_upload(*a, **k):
        calls["upload"] += 1
        return real_upload(*a, **k)

    def head_multipart(Bucket, Key, **k):
        return {"ChecksumSHA256": "abcDEF123==-7", "ContentLength": 4096}

    monkeypatch.setattr(s3, "upload_file", counting_upload)
    monkeypatch.setattr(s3, "head_object", head_multipart)
    digest = L.upload_verified(s3, str(f), BUCKET, "k", _sleep=lambda s: calls.__setitem__("sleep", calls["sleep"] + 1))
    assert len(digest) == 64  # hex sha256
    assert calls["upload"] == 1  # accepted on first try, not re-sent
    assert calls["sleep"] == 0


@mock_aws
def test_upload_verified_mismatch_is_not_retried(tmp_path, monkeypatch):
    s3 = _bucket()
    f = tmp_path / "s"
    f.write_bytes(b"data")
    calls = {"upload": 0}
    real_upload = s3.upload_file
    monkeypatch.setattr(s3, "upload_file",
                        lambda *a, **k: (calls.__setitem__("upload", calls["upload"] + 1), real_upload(*a, **k))[1])
    # single-part checksum that will never match the local file -> deterministic, must not retry
    monkeypatch.setattr(s3, "head_object", lambda **k: {"ChecksumSHA256": "WRONG==", "ContentLength": 4})
    with pytest.raises(L.UploadVerificationError):
        L.upload_verified(s3, str(f), BUCKET, "k", _sleep=lambda s: None)
    assert calls["upload"] == 1


@mock_aws
def test_upload_artifacts_rejects_key_collision(tmp_path):
    s3 = _bucket()
    a = tmp_path / "a"; a.mkdir(); (a / "model.pb").write_bytes(b"1")
    b = tmp_path / "b"; b.mkdir(); (b / "model.pb").write_bytes(b"2")
    with pytest.raises(ValueError):
        L.upload_artifacts(s3, BUCKET, "al", "r", [str(a / "model.pb"), str(b / "model.pb")])


@mock_aws
def test_download_prefix_skips_dir_placeholder(tmp_path):
    s3 = _bucket()
    (tmp_path / "f").write_bytes(b"data")
    out = L.upload_artifacts(s3, BUCKET, "al", "r", [str(tmp_path / "f")])
    # a console/S3-Files-created nested directory placeholder key ending in '/'
    s3.put_object(Bucket=BUCKET, Key="al/r/subdir/", Body=b"")
    dest = tmp_path / "out"
    written = L.download_prefix(s3, out["uri"], str(dest))
    assert [os.path.basename(w) for w in written] == ["f"]  # placeholder skipped, no IsADirectoryError


def test_local_dir_for_mount_rejects_bucket_mismatch():
    with pytest.raises(ValueError):
        L.local_dir_for_mount(f"s3://{BUCKET}/a/r/", "/mnt/traces", expected_bucket="other-bucket")


@mock_aws
def test_delete_prefix_raises_on_per_key_errors(tmp_path, monkeypatch):
    s3 = _bucket()
    (tmp_path / "f").write_bytes(b"data")
    out = L.upload_artifacts(s3, BUCKET, "al", "r", [str(tmp_path / "f")])
    monkeypatch.setattr(s3, "delete_objects",
                        lambda **k: {"Deleted": [], "Errors": [{"Key": "al/r/f", "Code": "AccessDenied"}]})
    with pytest.raises(RuntimeError):
        L.delete_prefix(s3, out["uri"])


@mock_aws
def test_delete_prefix_refuses_empty_key(tmp_path):
    s3 = _bucket()
    with pytest.raises(ValueError):
        L.delete_prefix(s3, f"s3://{BUCKET}/")
