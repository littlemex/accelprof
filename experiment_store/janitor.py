"""GC janitor — delete trace-bucket blobs that no live MLflow run references.

S3 lifecycle (data-layer) handles time-based expiry. What it cannot do is remove ORPHANS: a
run's ``<alias>/<run_id>/`` prefix whose MLflow run was never finalized (a crashed producer), was
marked FAILED, or was soft-deleted — the GB blobs would otherwise sit forever. This janitor walks
the bucket's run prefixes and purges the ones with no live run behind them.

It needs ``s3:DeleteObject`` (the dedicated janitor role — NOT mcp-reader, which is Delete-less).
Default is dry-run: it reports what it WOULD delete; deletion is opt-in via ``apply=True``.

Two invariants make apply-mode safe to run against a live bucket (both learned the hard way — see
the adversarial review):

  * Fail CLOSED on uncertainty. A prefix is an orphan ONLY when MLflow *authoritatively* has no
    such run (RESOURCE_DOES_NOT_EXIST) or reports it deleted/failed. ANY other MLflow error
    (outage, throttle, expired credential) ABORTS the whole sweep — never "can't ask" ⇒ "delete".
  * Grace period. A prefix whose newest object was written within ``min_age_seconds`` is skipped
    entirely, so an in-flight producer (run still RUNNING, uploading GB of blobs for minutes) is
    never purged out from under itself. An active upload keeps refreshing its newest LastModified,
    so it stays protected; a crashed producer stops writing and ages out.

A prefix carrying the ``s3_layout.RETENTION_MARKER`` sidecar object (written by ``store.hold``) is
never purged even when orphaned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import env, s3_layout

import re

_DEFAULT_MIN_AGE_SECONDS = 6 * 3600  # in-flight-upload grace period
# ONLY MLflow's authoritative "this run does not exist" code. Deliberately NOT the generic "404"/
# "NOT_FOUND": those also fire when the tracking ENDPOINT is absent (the App is opt-in/ephemeral —
# stopped between campaigns), and reading endpoint-absent as run-absent would mass-delete (B2).
_NOT_FOUND_CODES = {"RESOURCE_DOES_NOT_EXIST"}
# MLflow run ids are 32 lowercase hex. A prefix whose id segment is NOT this shape is foreign data
# (a hand-placed backup/, an unrelated upload) that the platform never wrote — never classify it as
# an orphan (N4). The canary id below is also this shape so a run_id-format-validating server
# returns NOT-FOUND (not INVALID_PARAMETER_VALUE, which would abort every sweep) (N5).
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CANARY_RUN_ID = "f" * 32
# Above this fraction of scanned prefixes classified as orphans, the sweep aborts (unless raised):
# the canary proves the tracking server is alive, but NOT that it is the SAME server that owns
# these runs — a recreated/empty/mis-pointed App answers not-found for everything and would
# mass-delete. A high orphan ratio is that signature; fail closed (N3).
_DEFAULT_MAX_ORPHAN_FRACTION = 0.5
_MIN_SCANNED_FOR_FRACTION = 5  # don't trip the breaker on a tiny bucket
# Non-terminal MLflow statuses: a run in one of these is live and must never be purged (M5). A
# terminal-but-not-FINISHED run (FAILED/KILLED) is an orphan; FINISHED is a keeper.
_LIVE_STATUSES = {"RUNNING", "SCHEDULED"}
# A RUNNING run whose newest blob is older than this is treated as ABANDONED (crashed after
# create_run) and surfaced in the report — generous by default so a genuinely long run is never
# flagged. Report-only: never auto-purged (reclaiming a live run's data is the one thing the
# janitor exists to prevent).
_DEFAULT_STALE_RUNNING_SECONDS = 7 * 24 * 3600


class JanitorAbortError(RuntimeError):
    """The sweep was aborted because MLflow could not be queried authoritatively — refusing to
    classify anything as an orphan (fail-closed). No deletions were performed."""


@dataclass
class SweepReport:
    scanned: int = 0
    orphans: list[str] = field(default_factory=list)      # artifacts_uris with no live run
    purged: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)         # orphaned but retention-held
    skipped_recent: list[str] = field(default_factory=list)   # within the grace period
    skipped_foreign: list[str] = field(default_factory=list)  # id not a run_id => not ours (N4)
    stale_running: list[str] = field(default_factory=list)    # RUNNING but abandoned; leak to purge
    deleted_objects: int = 0


@dataclass
class _PrefixScan:
    newest_mtime: float | None   # epoch seconds of the newest object, None if empty
    has_marker: bool
    empty: bool


def _is_not_found(exc: Exception) -> bool:
    code = getattr(exc, "error_code", None)
    if code in _NOT_FOUND_CODES:
        return True
    # boto/other clients surface a code under response.Error.Code
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict) and resp.get("Error", {}).get("Code") in _NOT_FOUND_CODES:
        return True
    return False


class Janitor:
    def __init__(self, s3_client: Any, trace_bucket: str, mlflow_io: Any,
                 min_age_seconds: int = _DEFAULT_MIN_AGE_SECONDS, now: float | None = None,
                 max_orphan_fraction: float = _DEFAULT_MAX_ORPHAN_FRACTION,
                 stale_running_seconds: int = _DEFAULT_STALE_RUNNING_SECONDS):
        self._s3 = s3_client
        self._bucket = trace_bucket
        self._mlflow = mlflow_io
        self._max_orphan_fraction = max_orphan_fraction
        self._min_age = min_age_seconds
        self._stale_running = stale_running_seconds
        self._now = now  # injectable clock (epoch seconds) for tests; None => time.time()

    def _clock(self) -> float:
        if self._now is not None:
            return self._now
        import time
        return time.time()

    def _list_run_prefixes(self) -> list[str]:
        """Enumerate ``<alias>/<run_id>/`` prefixes via two delimiter-scoped listings."""
        prefixes: list[str] = []
        for alias_prefix in self._common_prefixes(""):
            prefixes.extend(self._common_prefixes(alias_prefix))
        return prefixes

    def _common_prefixes(self, prefix: str) -> list[str]:
        out: list[str] = []
        token = None
        while True:
            kw: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix, "Delimiter": "/"}
            if token:
                kw["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kw)
            out.extend(cp["Prefix"] for cp in resp.get("CommonPrefixes", []))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return out

    def _scan_prefix(self, run_prefix: str) -> _PrefixScan:
        """One paginated listing of a run's objects: newest LastModified + retention marker + empty.
        Reused for the grace-period and retention checks (no second list)."""
        marker_key = run_prefix.rstrip("/") + "/" + s3_layout.RETENTION_MARKER
        newest: float | None = None
        has_marker = False
        empty = True
        token = None
        while True:
            kw: dict[str, Any] = {"Bucket": self._bucket, "Prefix": run_prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                empty = False
                if obj["Key"] == marker_key:
                    has_marker = True
                lm = obj.get("LastModified")
                ts = lm.timestamp() if hasattr(lm, "timestamp") else lm
                if ts is not None and (newest is None or ts > newest):
                    newest = ts
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return _PrefixScan(newest_mtime=newest, has_marker=has_marker, empty=empty)

    def _run_id_of(self, run_prefix: str) -> str:
        # "<alias>/<run_id>/" -> "<run_id>"
        return run_prefix.rstrip("/").split("/")[-1]

    def _classify(self, run_id: str) -> str:
        """Authoritative disposition of a prefix from MLflow: 'orphan' (run gone / soft-deleted /
        terminal non-FINISHED), 'running' (not-yet-terminal RUNNING/SCHEDULED — kept, but see the
        stale-running check in find_orphans), or 'keep' (FINISHED). Any non-not-found MLflow error
        raises JanitorAbortError (fail-closed)."""
        try:
            ref = self._mlflow.resolve_run(run_id)
        except Exception as e:  # noqa: BLE001
            if _is_not_found(e):
                return "orphan"
            raise JanitorAbortError(
                f"MLflow query for run {run_id!r} failed non-authoritatively ({e!r}); aborting "
                f"sweep without deleting anything") from e
        if getattr(ref, "is_deleted", False):
            return "orphan"      # soft-deleted run: its blobs are garbage (N3)
        if ref.status in _LIVE_STATUSES:
            return "running"     # not-yet-terminal (RUNNING/SCHEDULED) — keep (M5)
        return "keep" if ref.is_finished else "orphan"  # FINISHED keep; FAILED/KILLED => orphan

    def _is_orphan(self, run_id: str) -> bool:
        """Back-compat boolean view of _classify (True == GC-eligible)."""
        return self._classify(run_id) == "orphan"

    def _assert_tracking_reachable(self) -> None:
        """Prove the MLflow tracking server ANSWERS before any 'no run' is treated as authoritative.
        The App is opt-in/ephemeral, so a stopped/absent server must ABORT the sweep, never be read
        as 'every run is absent' (B2). A healthy server returns not-found for a bogus id; a
        down/absent one raises a transient/connection error instead."""
        try:
            self._mlflow.resolve_run(_CANARY_RUN_ID)
        except Exception as e:  # noqa: BLE001
            if _is_not_found(e):
                return  # healthy: the server answered "no such run"
            raise JanitorAbortError(
                f"MLflow tracking server is not answering authoritatively ({e!r}); aborting sweep "
                f"— the App may be stopped/absent, and prefixes must NOT be classified as orphans "
                f"in that state") from e

    def find_orphans(self) -> SweepReport:
        self._assert_tracking_reachable()  # fail-closed before any classification (B2)
        report = SweepReport()
        now = self._clock()
        for run_prefix in self._list_run_prefixes():
            report.scanned += 1
            uri = f"s3://{self._bucket}/{run_prefix}"
            scan = self._scan_prefix(run_prefix)
            if scan.empty:
                continue  # nothing to delete
            if scan.newest_mtime is not None and (now - scan.newest_mtime) < self._min_age:
                report.skipped_recent.append(uri)  # possibly in-flight — never touch (N2)
                continue
            run_id = self._run_id_of(run_prefix)
            if not _RUN_ID_RE.match(run_id):
                report.skipped_foreign.append(uri)  # not a run_id => not written by us (N4)
                continue
            disposition = self._classify(run_id)
            if disposition == "running":
                # A RUNNING run is normally kept, but one that crashed after create_run leaves its
                # (possibly GB) blobs guarded by a RUNNING status forever. If nothing has been
                # written for the stale window it is abandoned — SURFACE it (report-only, never
                # auto-deleted: a still-live long run must not be reclaimed) so an operator can purge.
                if scan.newest_mtime is not None and (now - scan.newest_mtime) > self._stale_running:
                    report.stale_running.append(uri)
                continue
            if disposition == "keep":
                continue  # a FINISHED run owns this prefix
            if scan.has_marker:
                report.kept.append(uri)  # explicit retention hold
                continue
            report.orphans.append(uri)
        return report

    def sweep(self, *, apply: bool = False) -> SweepReport:
        """Find orphans and, when ``apply=True``, purge their S3 prefixes. Default dry-run.
        Raises JanitorAbortError (before any deletion) if MLflow could not be queried reliably, or
        if the orphan ratio trips the circuit breaker (a recreated/empty/mis-pointed App would
        answer not-found for everything — N3)."""
        report = self.find_orphans()
        if apply:
            considered = report.scanned - len(report.skipped_recent) - len(report.skipped_foreign)
            if (considered >= _MIN_SCANNED_FOR_FRACTION
                    and len(report.orphans) > self._max_orphan_fraction * considered):
                raise JanitorAbortError(
                    f"orphan ratio {len(report.orphans)}/{considered} exceeds "
                    f"{self._max_orphan_fraction:.0%}; refusing to purge — the tracking server is "
                    f"answering but may be empty/recreated/mis-pointed (canary can't catch that). "
                    f"Verify MCP_MLFLOW_TRACKING_URI, or raise max_orphan_fraction to override.")
            for uri in report.orphans:
                report.deleted_objects += s3_layout.delete_prefix(self._s3, uri)
                report.purged.append(uri)
        return report


def main() -> None:
    """CronJob/Lambda entrypoint. Env: MCP_TRACE_BUCKET, MCP_MLFLOW_TRACKING_URI, MCP_AWS_REGION
    (shared loader in env.py). JANITOR_APPLY=1 to actually delete (default dry-run, prints plan);
    JANITOR_MIN_AGE_HOURS overrides the in-flight grace period (default 6h);
    JANITOR_MAX_ORPHAN_PCT overrides the circuit-breaker orphan ratio (default 50);
    JANITOR_STALE_RUNNING_HOURS overrides the abandoned-RUNNING report threshold (default 168h)."""
    import json

    from .mlflow_io import MlflowIO
    from .store import default_s3_client

    e = env.os_environ()
    bucket = env.require(e, env.TRACE_BUCKET_ENV, "the trace bucket to GC")
    region = env.region(e)
    tracking = env.require(e, env.TRACKING_ENV, "the MLflow tracking URI / app ARN")
    apply = e.get("JANITOR_APPLY") == "1"
    # Defaults derived from the library constants so the CLI and Janitor() can't drift.
    min_age = env.optional_int(e, "JANITOR_MIN_AGE_HOURS", _DEFAULT_MIN_AGE_SECONDS // 3600) * 3600
    max_frac = env.optional_int(e, "JANITOR_MAX_ORPHAN_PCT",
                                round(_DEFAULT_MAX_ORPHAN_FRACTION * 100)) / 100.0
    stale = env.optional_int(e, "JANITOR_STALE_RUNNING_HOURS",
                             _DEFAULT_STALE_RUNNING_SECONDS // 3600) * 3600
    j = Janitor(default_s3_client(region), bucket, MlflowIO(tracking_uri=tracking),
                min_age_seconds=min_age, max_orphan_fraction=max_frac, stale_running_seconds=stale)
    rep = j.sweep(apply=apply)
    print(json.dumps({"apply": apply, "scanned": rep.scanned, "orphans": rep.orphans,
                      "kept": rep.kept, "purged": rep.purged,
                      "skipped_recent": rep.skipped_recent,
                      "skipped_foreign": rep.skipped_foreign,
                      "stale_running": rep.stale_running,  # abandoned RUNNING: inspect + purge manually
                      "deleted_objects": rep.deleted_objects}, indent=2))


if __name__ == "__main__":
    main()
