"""Thin MLflow read/write with fixed reserved tags and fully open content.

Write side: create a run under the experiment named by the alias, set the reserved identity tags
(ids.creation_tags), then log the caller's metrics/params/tags VERBATIM. We never enumerate or
normalize content keys — MLflow's schemaless metrics/params/tags are the point (no accuracy-metric
set is baked into the platform).

Completeness contract: ``finalize_run`` logs content FIRST, sets ``artifacts_uri`` next, and marks
the run FINISHED LAST. So a run with status FINISHED always has artifacts_uri + its content; a
run that failed part-way is left non-FINISHED and the store marks it FAILED. Readers should treat
only FINISHED runs as usable (store.resolve filters them by default) — ``RunRef.status`` exposes
this.

Read side: resolve an alias to its runs (experiment name == alias, so a single
get_experiment_by_name, avoiding the search_runs(experiment_ids=[]) == "zero runs" pitfall, and
paginating so an alias with >1000 runs is not silently truncated), or a run_id to one run.

Content value rules (the only constraints, imposed by MLflow, surfaced early and clearly):
  metrics -> finite real numbers (int/float; NaN/Inf/bool/overflowing/non-numeric raise);
  params/tags -> stringified; a caller tag may not collide with a reserved key.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import ids


@dataclass
class RunRef:
    run_id: str
    status: str | None
    alias: str | None
    chip: str | None
    region: str | None
    workload_id: str | None
    artifacts_uri: str | None
    start_time: int | None = None
    # MLflow's active|deleted flag. A soft-deleted run keeps status FINISHED (delete_run does NOT
    # change status), so lifecycle_stage is the ONLY signal that a run was retired — the janitor
    # needs it to GC a deleted run's blobs (N3). Defaults to "active" for injected test doubles.
    lifecycle_stage: str = "active"
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def is_finished(self) -> bool:
        return self.status == "FINISHED"

    @property
    def is_deleted(self) -> bool:
        return self.lifecycle_stage == "deleted"


def _import_mlflow():
    import mlflow
    from mlflow.tracking import MlflowClient

    return mlflow, MlflowClient


def validate_content(metrics: dict | None, tags: dict | None) -> None:
    """Pure, cheap validation of open content — call BEFORE any upload so a bad metric fails fast
    instead of after a GB transfer."""
    for k, v in (metrics or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"metric {k!r} must be a finite number, got {v!r}")
        try:
            fv = float(v)
        except (OverflowError, ValueError) as e:
            raise ValueError(f"metric {k!r} is not a usable float: {v!r} ({e})") from None
        if not math.isfinite(fv):
            raise ValueError(f"metric {k!r} must be finite, got {v!r}")
    for k in (tags or {}):
        if ids.is_reserved(k):
            raise ValueError(f"tag {k!r} collides with a reserved key {ids.RESERVED_TAGS}")


class MlflowIO:
    def __init__(self, tracking_uri: str, client: Any | None = None):
        self._tracking_uri = tracking_uri
        self._client = client

    def _get_client(self):
        if self._client is None:
            _mlflow, MlflowClient = _import_mlflow()
            _mlflow.set_tracking_uri(self._tracking_uri)
            self._client = MlflowClient(tracking_uri=self._tracking_uri)
        return self._client

    def _experiment_id(self, alias: str, create: bool) -> str | None:
        c = self._get_client()
        exp = c.get_experiment_by_name(alias)
        if exp is not None:
            return exp.experiment_id
        if not create:
            return None
        try:
            return c.create_experiment(alias)
        except Exception:  # noqa: BLE001 - lose a create race: another producer created it first
            exp = c.get_experiment_by_name(alias)
            if exp is None:
                raise
            return exp.experiment_id

    def create_run(self, *, alias: str, chip: str, region: str, workload_id: str) -> str:
        """Create the run (so its run_id is known for the S3 prefix) with the creation-time
        identity tags. artifacts_uri is set later once the upload target exists."""
        c = self._get_client()
        exp_id = self._experiment_id(alias, create=True)
        tags = ids.creation_tags(alias=alias, chip=chip, region=region, workload_id=workload_id)
        run = c.create_run(experiment_id=exp_id, tags=tags, run_name=ids.run_name(chip, workload_id))
        return run.info.run_id

    def finalize_run(self, run_id: str, *, artifacts_uri: str,
                     metrics: dict | None, params: dict | None, tags: dict | None) -> None:
        """Log content, then artifacts_uri, then mark FINISHED last (completeness contract)."""
        validate_content(metrics, tags)
        c = self._get_client()
        for k, v in (tags or {}).items():
            c.set_tag(run_id, k, str(v))
        for k, v in (params or {}).items():
            c.log_param(run_id, k, str(v))
        for k, v in (metrics or {}).items():
            c.log_metric(run_id, k, float(v))
        c.set_tag(run_id, ids.ARTIFACTS_URI_TAG, artifacts_uri)
        c.set_terminated(run_id, status="FINISHED")

    def fail_run(self, run_id: str) -> None:
        try:
            self._get_client().set_terminated(run_id, status="FAILED")
        except Exception:  # noqa: BLE001 - best-effort; the original error must propagate
            pass

    def _to_ref(self, run: Any) -> RunRef:
        tags = dict(run.data.tags)
        return RunRef(
            run_id=run.info.run_id,
            status=run.info.status,
            alias=tags.get(ids.ALIAS_TAG),
            chip=tags.get(ids.CHIP_TAG),
            region=tags.get(ids.REGION_TAG),
            workload_id=tags.get(ids.WORKLOAD_TAG),
            artifacts_uri=tags.get(ids.ARTIFACTS_URI_TAG),
            start_time=getattr(run.info, "start_time", None),
            lifecycle_stage=getattr(run.info, "lifecycle_stage", "active") or "active",
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
            tags=tags,
        )

    def resolve_alias(self, alias: str, page_size: int = 1000) -> list[RunRef]:
        c = self._get_client()
        exp_id = self._experiment_id(alias, create=False)
        if exp_id is None:
            raise LookupError(f"no experiment for alias {alias!r}")
        refs: list[RunRef] = []
        token = None
        while True:
            page = c.search_runs(experiment_ids=[exp_id], max_results=page_size, page_token=token)
            refs.extend(self._to_ref(r) for r in page)
            token = getattr(page, "token", None)
            if not token:
                break
        return refs

    def search(self, *, alias: str | None = None, filter_string: str = "",
               order_by: list[str] | None = None, page_size: int = 1000,
               max_results: int = 1000) -> list[RunRef]:
        """Search runs by MLflow filter (open tags/metrics/params) — the tag/experiment-number
        query surface. ``alias`` scopes to one experiment; None searches all. ``filter_string`` is
        MLflow's own DSL, e.g. "tags.run_no = '10'", "tags.framework = 'vllm' and metrics.tpot_ms <
        20". We do NOT constrain the tag vocabulary (fixed IDs / open content); the caller queries
        whatever they logged."""
        c = self._get_client()
        if alias:
            exp_id = self._experiment_id(alias, create=False)
            if exp_id is None:
                raise LookupError(f"no experiment for alias {alias!r}")
            exp_ids = [exp_id]
        else:
            exp_ids = [e.experiment_id for e in c.search_experiments()]
        if not exp_ids:
            return []
        refs: list[RunRef] = []
        token = None
        while len(refs) < max_results:
            page = c.search_runs(experiment_ids=exp_ids, filter_string=filter_string or "",
                                 order_by=order_by or [],
                                 max_results=min(page_size, max_results - len(refs)),
                                 page_token=token)
            refs.extend(self._to_ref(r) for r in page)
            token = getattr(page, "token", None)
            if not token:
                break
        return refs[:max_results]

    def resolve_run(self, run_id: str) -> RunRef:
        return self._to_ref(self._get_client().get_run(run_id))

    def delete_run(self, run_id: str) -> None:
        """Soft-delete the MLflow run (used by purge). Idempotent for an ALREADY-DELETED / absent
        run (that is a no-op, so a retry is safe), but a transient error (network/throttle) is
        RE-RAISED — swallowing it would let purge report success while the run stays live and its
        artifacts_uri now points at a deleted prefix (fail-closed, matching the janitor)."""
        try:
            self._get_client().delete_run(run_id)
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "error_code", None)
            if code in ("RESOURCE_DOES_NOT_EXIST", "RESOURCE_NOT_FOUND", "NOT_FOUND"):
                return  # already gone → done
            raise
