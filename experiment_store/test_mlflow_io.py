"""MLflow read/write against a real file-store MLflow (no server, no network).

Run with MLFLOW_ALLOW_FILE_STORE=true (file store is in maintenance mode in recent MLflow).
"""
from __future__ import annotations

import pytest

mlflow = pytest.importorskip("mlflow")

from . import ids  # noqa: E402
from .mlflow_io import MlflowIO  # noqa: E402


def _io(tmp_path):
    return MlflowIO(tracking_uri=f"file://{tmp_path}/mlruns")


def test_create_finalize_resolve_roundtrip(tmp_path):
    io = _io(tmp_path)
    run_id = io.create_run(alias="alias1", chip="gpu", region="ap-northeast-1", workload_id="w1")
    io.finalize_run(run_id, artifacts_uri="s3://b/alias1/%s/" % run_id,
                    metrics={"cosine": 0.9997, "latency_p50_ms": 110.0},
                    params={"impl": "vllm-0.20"}, tags={"note": "smoke"})

    # resolve by alias
    runs = io.resolve_alias("alias1")
    assert len(runs) == 1
    r = runs[0]
    assert r.run_id == run_id
    assert r.chip == "gpu"
    assert r.region == "ap-northeast-1"
    assert r.workload_id == "w1"
    assert r.artifacts_uri.endswith("%s/" % run_id)
    assert r.metrics["cosine"] == 0.9997
    assert r.params["impl"] == "vllm-0.20"
    assert r.tags["note"] == "smoke"

    # resolve by id
    r2 = io.resolve_run(run_id)
    assert r2.run_id == run_id


def test_resolve_alias_unknown_raises(tmp_path):
    io = _io(tmp_path)
    with pytest.raises(LookupError):
        io.resolve_alias("nope")


def test_multiple_runs_same_alias(tmp_path):
    io = _io(tmp_path)
    for chip in ("gpu", "neuron"):
        rid = io.create_run(alias="cmp", chip=chip, region="ap-northeast-1", workload_id="w")
        io.finalize_run(rid, artifacts_uri="s3://b/cmp/%s/" % rid, metrics={}, params={}, tags={})
    runs = io.resolve_alias("cmp")
    assert {r.chip for r in runs} == {"gpu", "neuron"}


def test_finalize_rejects_non_numeric_metric(tmp_path):
    io = _io(tmp_path)
    rid = io.create_run(alias="a", chip="gpu", region="ap-northeast-1", workload_id="w")
    with pytest.raises(ValueError):
        io.finalize_run(rid, artifacts_uri="s3://b/a/r/", metrics={"bad": "x"}, params={}, tags={})


def test_finalize_rejects_nan_metric(tmp_path):
    io = _io(tmp_path)
    rid = io.create_run(alias="a", chip="gpu", region="ap-northeast-1", workload_id="w")
    with pytest.raises(ValueError):
        io.finalize_run(rid, artifacts_uri="s3://b/a/r/", metrics={"bad": float("nan")},
                        params={}, tags={})


def test_finalize_rejects_reserved_tag_collision(tmp_path):
    io = _io(tmp_path)
    rid = io.create_run(alias="a", chip="gpu", region="ap-northeast-1", workload_id="w")
    with pytest.raises(ValueError):
        io.finalize_run(rid, artifacts_uri="s3://b/a/r/", metrics={},
                        params={}, tags={ids.CHIP_TAG: "neuron"})
