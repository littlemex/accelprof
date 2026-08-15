"""End-to-end library test: moto S3 + file-store MLflow wired through ExperimentStore.

Run with MLFLOW_ALLOW_FILE_STORE=true.
"""
from __future__ import annotations

import boto3
import pytest

mlflow = pytest.importorskip("mlflow")
moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

from .mlflow_io import MlflowIO  # noqa: E402
from .store import ExperimentStore  # noqa: E402

BUCKET = "mcp-traces-test"
REGION = "us-east-1"


def _store(tmp_path, mount_base=None):
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    io = MlflowIO(tracking_uri=f"file://{tmp_path}/mlruns")
    return ExperimentStore(trace_bucket=BUCKET, s3_client=s3, mlflow=io, mount_base=mount_base)


@mock_aws
def test_log_resolve_download_roundtrip(tmp_path):
    store = _store(tmp_path)
    sub = tmp_path / "profile_output"
    sub.mkdir()
    (sub / "model.neff").write_bytes(b"NEFF")
    (sub / "run.ntff").write_bytes(b"NTFF")

    run_id = store.log("cmp", chip="neuron", region="us-east-1", workload_id="w1",
                       metrics={"cosine": 0.999}, params={"impl": "nxdi"},
                       tags={"note": "hi"}, artifacts=[str(sub)])

    runs = store.resolve("cmp")
    assert len(runs) == 1
    r = runs[0]
    assert r.run_id == run_id
    assert r.chip == "neuron"
    assert r.artifacts_uri == f"s3://{BUCKET}/cmp/{run_id}/"

    dest = tmp_path / "dl"
    written = store.download(r, str(dest))
    assert (dest / "model.neff").read_bytes() == b"NEFF"
    assert len(written) == 2


@mock_aws
def test_log_rejects_bad_identity(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.log("bad alias", chip="gpu", region="us-east-1", workload_id="w")


@mock_aws
def test_locate_needs_mount_base(tmp_path):
    store = _store(tmp_path)  # no mount_base
    run_id = store.log("a", chip="gpu", region="us-east-1", workload_id="w")
    r = store.resolve(run_id, by="id")[0]
    with pytest.raises(ValueError):
        store.locate(r)


@mock_aws
def test_locate_returns_mount_path(tmp_path):
    store = _store(tmp_path, mount_base="/mnt/traces")
    run_id = store.log("a", chip="gpu", region="us-east-1", workload_id="w")
    r = store.resolve(run_id, by="id")[0]
    assert store.locate(r) == f"/mnt/traces/a/{run_id}/"


@mock_aws
def test_locate_rejects_cross_bucket_uri(tmp_path):
    store = _store(tmp_path, mount_base="/mnt/traces")
    run_id = store.log("a", chip="gpu", region="us-east-1", workload_id="w")
    r = store.resolve(run_id, by="id")[0]
    # simulate a run whose artifacts live in another region's bucket
    r.artifacts_uri = f"s3://other-region-bucket/a/{run_id}/"
    with pytest.raises(ValueError):
        store.locate(r)


@mock_aws
def test_resolve_excludes_unfinished_by_default(tmp_path):
    store = _store(tmp_path)
    run_id = store.log("a", chip="gpu", region="us-east-1", workload_id="w")
    # fabricate a second, unfinished run under the same alias
    bad = store.mlflow.create_run(alias="a", chip="neuron", region="us-east-1", workload_id="w")
    assert {r.run_id for r in store.resolve("a")} == {run_id}                    # FINISHED only
    assert {r.run_id for r in store.resolve("a", include_unfinished=True)} == {run_id, bad}


@mock_aws
def test_purge_deletes_blobs_and_run(tmp_path):
    store = _store(tmp_path)
    f = tmp_path / "trace.nsys-rep"
    f.write_bytes(b"x" * 10)
    run_id = store.log("a", chip="gpu", region="us-east-1", workload_id="w", artifacts=[str(f)])
    r = store.resolve(run_id, by="id")[0]
    deleted = store.purge(r)
    assert deleted == 1
    # MLflow run soft-deleted -> the experiment still exists but has no active runs
    assert store.resolve("a") == []


@mock_aws
def test_namespace_tag_auto_injected(tmp_path):
    """namespace is stamped on every run from the store's configured value; a caller's explicit
    namespace tag overrides it."""
    s3 = boto3.client("s3", region_name=REGION); s3.create_bucket(Bucket=BUCKET)
    io = MlflowIO(tracking_uri=f"file://{tmp_path}/mlruns")
    store = ExperimentStore(trace_bucket=BUCKET, s3_client=s3, mlflow=io, namespace="ddp")
    rid = store.log("a", chip="gpu", region="us-east-1", workload_id="w")
    assert store.resolve(rid, by="id")[0].tags["namespace"] == "ddp"
    rid2 = store.log("a", chip="gpu", region="us-east-1", workload_id="w", tags={"namespace": "custom"})
    assert store.resolve(rid2, by="id")[0].tags["namespace"] == "custom"  # caller wins
    # searchable
    assert len(store.search(filter_string="tags.namespace = 'ddp'")) == 1


@mock_aws
def test_search_by_open_tag(tmp_path):
    """The tag / experiment-number query surface: runs tagged with an open 'run_no' are searchable
    by MLflow filter without the platform fixing that tag's meaning."""
    store = _store(tmp_path)
    for no in (1, 2, 3):
        store.log("sweep-a", chip="gpu", region="us-east-1", workload_id="w",
                  metrics={"tpot_ms": 10.0 + no}, tags={"run_no": str(no), "framework": "vllm"})
    only2 = store.search(alias="sweep-a", filter_string="tags.run_no = '2'")
    assert len(only2) == 1 and only2[0].tags["run_no"] == "2"
    allv = store.search(filter_string="tags.framework = 'vllm'")
    assert len(allv) == 3


@mock_aws
def test_resolve_excludes_soft_deleted_by_id(tmp_path):
    """A soft-deleted run keeps status FINISHED; resolve(by='id') must NOT return it as usable
    (else a consumer resolves a run the janitor has purged and locate()s an empty dir)."""
    store = _store(tmp_path)
    run_id = store.log("a", chip="gpu", region="us-east-1", workload_id="w")
    store.mlflow.delete_run(run_id)  # soft-delete
    assert store.resolve(run_id, by="id") == []                       # excluded from usable
    got = store.resolve(run_id, by="id", include_unfinished=True)      # visible for diagnostics/GC
    assert len(got) == 1 and got[0].is_deleted


@mock_aws
def test_construct_rejects_non_sigv4_client(tmp_path):
    """C6: an explicit non-SigV4 signer would 400 against SSE-KMS trace buckets — fail at boot."""
    from botocore.config import Config
    bad = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3"))
    io = MlflowIO(tracking_uri=f"file://{tmp_path}/mlruns")
    with pytest.raises(ValueError):
        ExperimentStore(trace_bucket=BUCKET, s3_client=bad, mlflow=io)


@mock_aws
def test_construct_accepts_v4_and_none(tmp_path):
    from botocore.config import Config
    io = MlflowIO(tracking_uri=f"file://{tmp_path}/mlruns")
    for sig in ("s3v4", "v4"):
        c = boto3.client("s3", region_name=REGION, config=Config(signature_version=sig))
        ExperimentStore(trace_bucket=BUCKET, s3_client=c, mlflow=io)  # no raise

    class _Fake:  # injected test double: no meta.config => signature_version None => allowed
        pass
    ExperimentStore(trace_bucket=BUCKET, s3_client=_Fake(), mlflow=io)


@mock_aws
def test_hold_writes_retention_marker(tmp_path):
    from . import s3_layout
    store = _store(tmp_path)
    f = tmp_path / "t.nsys-rep"; f.write_bytes(b"x")
    run_id = store.log("a", chip="gpu", region="us-east-1", workload_id="w", artifacts=[str(f)])
    r = store.resolve(run_id, by="id")[0]
    store.hold(r)
    assert s3_layout.has_retention_marker(store.s3_client, r.artifacts_uri)


@mock_aws
def test_failed_upload_marks_run_failed_and_raises(tmp_path, monkeypatch):
    store = _store(tmp_path)

    class Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    monkeypatch.setattr(store.s3_client, "upload_file", lambda *a, **k: (_ for _ in ()).throw(Denied()))
    f = tmp_path / "t"
    f.write_bytes(b"z")
    with pytest.raises(Denied):
        store.log("a", chip="gpu", region="us-east-1", workload_id="w", artifacts=[str(f)])
    # the FAILED run is excluded from the default resolve; visible only with include_unfinished
    assert store.resolve("a") == []
    failed = store.resolve("a", include_unfinished=True)
    assert len(failed) == 1
    assert failed[0].status == "FAILED"
    assert failed[0].artifacts_uri is None
