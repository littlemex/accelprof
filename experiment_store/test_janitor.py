"""Unit tests for the GC janitor (fake S3 + fake MLflow)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest

from .janitor import Janitor, JanitorAbortError

BUCKET = "mcp-traces-test"


def _rid(tag: str) -> str:
    """Deterministic 32-hex MLflow-shaped run id from a short label (real ids are 32 hex; the
    janitor now skips non-conforming prefixes, so tests must use conforming ids)."""
    return hashlib.md5(tag.encode()).hexdigest()


class _NotFound(Exception):
    """Mimics an MLflow RESOURCE_DOES_NOT_EXIST (carries the error_code the janitor keys on)."""
    error_code = "RESOURCE_DOES_NOT_EXIST"


class _Transient(Exception):
    """Mimics a throttle/outage — NO not-found code, so the janitor must abort."""


@dataclass
class FakeS3:
    # objects: key -> (bytes, mtime|None); supports list_objects_v2 (Prefix[/Delimiter]) + delete
    objects: dict = field(default_factory=dict)
    deleted: list = field(default_factory=list)

    def _lm(self, mtime):
        if mtime is None:
            return None
        return type("LM", (), {"timestamp": lambda self, _m=mtime: _m})()

    def list_objects_v2(self, Bucket, Prefix="", Delimiter=None, MaxKeys=None, ContinuationToken=None):
        keys = [k for k in self.objects if k.startswith(Prefix)]
        if Delimiter:
            cps, contents = set(), []
            for k in keys:
                rest = k[len(Prefix):]
                if Delimiter in rest:
                    cps.add(Prefix + rest.split(Delimiter)[0] + Delimiter)
                else:
                    contents.append({"Key": k})
            return {"CommonPrefixes": [{"Prefix": p} for p in sorted(cps)],
                    "Contents": contents, "IsTruncated": False}
        contents = [{"Key": k, "LastModified": self._lm(self.objects[k][1])} for k in keys]
        if MaxKeys:
            contents = contents[:MaxKeys]
        return {"Contents": contents, "KeyCount": len(contents), "IsTruncated": False}

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)
            self.deleted.append(o["Key"])
        return {"Deleted": Delete["Objects"]}


@dataclass
class FakeRef:
    run_id: str
    status: str
    lifecycle_stage: str = "active"

    @property
    def is_finished(self):
        return self.status == "FINISHED"

    @property
    def is_deleted(self):
        return self.lifecycle_stage == "deleted"


@dataclass
class FakeMlflow:
    runs: dict = field(default_factory=dict)   # run_id -> status | (status, lifecycle_stage)
    transient: set = field(default_factory=set)  # run_ids that raise a transient error
    server_down: bool = False                   # every call (incl. the canary) raises transient

    def resolve_run(self, run_id):
        if self.server_down or run_id in self.transient:
            raise _Transient(run_id)
        if run_id not in self.runs:
            raise _NotFound(run_id)
        val = self.runs[run_id]
        status, stage = val if isinstance(val, tuple) else (val, "active")
        return FakeRef(run_id, status, stage)


def _s3_with(*runs, mtime=None):
    # each run: (alias, run_id, [files]); mtime applied to every object
    objs = {}
    for alias, rid, files in runs:
        for f in files:
            objs[f"{alias}/{rid}/{f}"] = (b"x", mtime)
    return FakeS3(objects=objs)


def _j(s3, mlflow, **kw):
    # now far in the future so default-mtime (None) prefixes are never "recent"
    return Janitor(s3, BUCKET, mlflow, now=10**12, **kw)


def test_finds_orphan_when_no_mlflow_run():
    live, orphan = _rid("live1"), _rid("orphan1")
    s3 = _s3_with(("a", live, ["model.neff"]), ("a", orphan, ["trace"]))
    mlflow = FakeMlflow(runs={live: "FINISHED"})  # orphan has no run
    rep = _j(s3, mlflow).find_orphans()
    assert rep.scanned == 2
    assert rep.orphans == [f"s3://{BUCKET}/a/{orphan}/"]


def test_failed_run_is_orphan():
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]))
    rep = _j(s3, FakeMlflow(runs={r: "FAILED"})).find_orphans()  # exists but not FINISHED
    assert rep.orphans == [f"s3://{BUCKET}/a/{r}/"]


def test_live_finished_run_kept():
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]))
    rep = _j(s3, FakeMlflow(runs={r: "FINISHED"})).find_orphans()
    assert rep.orphans == []


def test_running_run_kept():
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]))
    rep = _j(s3, FakeMlflow(runs={r: "RUNNING"})).find_orphans()
    assert rep.orphans == [] and rep.skipped_recent == []


def test_abandoned_running_is_reported_not_purged():
    """A3: a RUNNING run whose blobs are older than the stale window (crashed after create_run)
    is SURFACED as stale_running — but never auto-purged (a live long run must not be reclaimed)."""
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]), mtime=0)   # written far before now (10**12) => way past stale
    rep = _j(s3, FakeMlflow(runs={r: "RUNNING"})).sweep(apply=True)
    assert rep.stale_running == [f"s3://{BUCKET}/a/{r}/"]
    assert rep.orphans == [] and rep.purged == []   # report-only, not deleted even in apply mode


def test_running_past_grace_but_not_abandoned_is_kept():
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]), mtime=10**12 - 86400)   # 1 day old: past 6h grace, under 7d stale
    rep = _j(s3, FakeMlflow(runs={r: "RUNNING"})).find_orphans()
    assert rep.stale_running == [] and rep.orphans == [] and rep.skipped_recent == []


def test_scheduled_run_kept():
    """M5: a SCHEDULED (queued, not-yet-terminal) run is live — not an orphan."""
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]))
    rep = _j(s3, FakeMlflow(runs={r: "SCHEDULED"})).find_orphans()
    assert rep.orphans == []


def test_foreign_prefix_is_skipped_not_purged():
    """N4: a prefix whose id segment is not a 32-hex run id was not written by us — never purge."""
    r = _rid("real")
    s3 = _s3_with(("manual-backup", "important-neffs", ["x.neff"]), ("a", r, ["t"]))
    mlflow = FakeMlflow(runs={})  # even with no live runs, the foreign prefix must be untouched
    rep = _j(s3, mlflow).sweep(apply=True)
    assert f"s3://{BUCKET}/manual-backup/important-neffs/" in rep.skipped_foreign
    assert s3.objects.get("manual-backup/important-neffs/x.neff") is not None
    assert rep.purged == [f"s3://{BUCKET}/a/{r}/"]


def test_tracking_server_down_aborts_before_scanning():
    """B2: the App is ephemeral; a down/absent tracking server must abort the sweep (canary)."""
    s3 = _s3_with(("a", _rid("r1"), ["t1", "t2"]))
    with pytest.raises(JanitorAbortError):
        _j(s3, FakeMlflow(server_down=True)).sweep(apply=True)
    assert s3.deleted == []


def test_circuit_breaker_aborts_on_high_orphan_ratio():
    """N3: canary passes (server alive) but the App is empty/recreated -> every run resolves
    not-found -> the orphan-ratio breaker must abort before deleting."""
    runs = [("a", _rid(f"run{i}"), ["t"]) for i in range(8)]
    s3 = _s3_with(*runs)
    mlflow = FakeMlflow(runs={})  # server answers not-found for all (canary still passes)
    with pytest.raises(JanitorAbortError):
        _j(s3, mlflow).sweep(apply=True)
    assert s3.deleted == []


def test_circuit_breaker_override_allows_purge():
    runs = [("a", _rid(f"run{i}"), ["t"]) for i in range(8)]
    s3 = _s3_with(*runs)
    rep = _j(s3, FakeMlflow(runs={}), max_orphan_fraction=1.0).sweep(apply=True)
    assert len(rep.purged) == 8


def test_soft_deleted_run_is_orphan():
    """N3(earlier): a soft-deleted run keeps status FINISHED; its blobs must still be collected."""
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]))
    mlflow = FakeMlflow(runs={r: ("FINISHED", "deleted")})
    rep = _j(s3, mlflow).find_orphans()
    assert rep.orphans == [f"s3://{BUCKET}/a/{r}/"]


def test_transient_mlflow_error_aborts_sweep():
    """N1/M1 blocker: an MLflow outage must NOT be read as 'orphan' — abort without deleting."""
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t1", "t2"]))
    mlflow = FakeMlflow(runs={}, transient={r})
    with pytest.raises(JanitorAbortError):
        _j(s3, mlflow).sweep(apply=True)
    assert s3.deleted == []


def test_grace_period_skips_recent_prefix():
    """N2/M2 blocker: a prefix whose newest object is younger than the grace period is skipped."""
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]), mtime=1000.0)
    mlflow = FakeMlflow(runs={})
    j = Janitor(s3, BUCKET, mlflow, min_age_seconds=3600, now=1000.0 + 60)  # 60s old < 1h
    rep = j.sweep(apply=True)
    assert rep.orphans == [] and rep.skipped_recent == [f"s3://{BUCKET}/a/{r}/"]
    assert s3.deleted == []


def test_grace_period_elapsed_prefix_is_orphan():
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]), mtime=1000.0)
    j = Janitor(s3, BUCKET, FakeMlflow(runs={}), min_age_seconds=3600, now=1000.0 + 7200)  # 2h old
    rep = j.find_orphans()
    assert rep.orphans == [f"s3://{BUCKET}/a/{r}/"]


def test_retention_keep_marker_holds_orphan():
    r = _rid("r1")
    s3 = _s3_with(("a", r, ["t"]))
    s3.objects[f"a/{r}/.retention-keep"] = (b"", None)   # explicit hold
    mlflow = FakeMlflow(runs={})                          # no live run -> would be orphan
    rep = _j(s3, mlflow).find_orphans()
    assert rep.orphans == []
    assert rep.kept == [f"s3://{BUCKET}/a/{r}/"]


def test_sweep_dry_run_deletes_nothing():
    s3 = _s3_with(("a", _rid("orphan"), ["t1", "t2"]))
    rep = _j(s3, FakeMlflow()).sweep(apply=False)
    assert rep.orphans and rep.purged == [] and s3.deleted == []


def test_sweep_apply_purges_orphans():
    live, orphan = _rid("live"), _rid("orphan")
    s3 = _s3_with(("a", live, ["k"]), ("b", orphan, ["t1", "t2"]))
    mlflow = FakeMlflow(runs={live: "FINISHED"})
    rep = _j(s3, mlflow).sweep(apply=True)
    assert rep.purged == [f"s3://{BUCKET}/b/{orphan}/"]
    assert rep.deleted_objects == 2
    assert set(s3.deleted) == {f"b/{orphan}/t1", f"b/{orphan}/t2"}
    assert any(k.startswith(f"a/{live}/") for k in s3.objects)  # live run untouched
