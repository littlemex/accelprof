"""experiment_store — the platform's fixed layer, as a plain library (NOT a service / MCP).

Producers call ``log`` to register a run (upload artifacts to S3 + record identity + open content
in MLflow). Consumers call ``resolve`` to find runs by alias/id and ``locate`` to get the
Pod-local path of a run's artifacts on the mounted bucket (AWS S3 Files) — no copy. ``download``
is a non-mounted fallback. ``purge`` deletes a run's blobs (+ optionally the MLflow run) and is
used by the janitor (which holds the scoped Delete grant; mcp-reader does not).

The contract this fixes is ONLY identity (ids.py) + the S3 layout (s3_layout.py). Metrics,
params, tags, and which artifact files exist are entirely the producer's choice.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from . import ids, s3_layout
from .mlflow_io import MlflowIO, RunRef, validate_content

_log = logging.getLogger(__name__)


def default_s3_client(region: str) -> Any:
    """Build the S3 client the store expects: pinned to SigV4, which is REQUIRED because the trace
    buckets are SSE-KMS (a presigned/GET against a KMS object with any other signer 400s). Turning
    the README's 's3v4 is required' note into code so producers/consumers don't each rediscover it."""
    import boto3
    from botocore.config import Config

    return boto3.client("s3", region_name=region, config=Config(signature_version="s3v4"))


@dataclass
class ExperimentStore:
    """Bind a tracking server + a trace bucket + an S3 client into the platform API.

    ``trace_bucket`` is this store's region's bucket (producers write here; the co-located
    consumer mounts the same bucket). ``mount_base`` is where that bucket is mounted on the
    consumer Pod (S3 Files); only ``locate`` needs it. Prefer :meth:`build` to get a correctly
    configured (SigV4) client; the raw constructor stays open for tests / injection.
    """
    trace_bucket: str
    s3_client: Any
    mlflow: MlflowIO
    mount_base: str | None = None
    # Auto-injected grouping tag value (e.g. "ddp"). When set, every run gets tag namespace=<this>
    # unless the caller overrides it. In a pod, wire it from the k8s namespace (downward API) so
    # runs are grouped by where they ran without the producer passing it each time.
    namespace: str | None = None

    @classmethod
    def build(cls, *, region: str, trace_bucket: str, tracking_uri: str,
              mount_base: str | None = None, namespace: str | None = None) -> "ExperimentStore":
        return cls(trace_bucket=trace_bucket, s3_client=default_s3_client(region),
                   mlflow=MlflowIO(tracking_uri=tracking_uri), mount_base=mount_base,
                   namespace=namespace)

    def __post_init__(self):
        sig = getattr(getattr(self.s3_client, "meta", None), "config", None)
        sigver = getattr(sig, "signature_version", None)
        # botocore spells SigV4 as both "s3v4" and "v4"; both are fine. None = an injected test
        # double whose signer we can't introspect. Anything else is an explicit wrong signer for
        # SSE-KMS trace buckets (GET/presign 400s) — fail loud at construction, matching the rest
        # of the platform's fail-at-boot contracts, rather than whispering a warning.
        if sigver not in (None, "s3v4", "v4"):
            raise ValueError(
                f"s3_client signature_version={sigver!r}; SSE-KMS trace buckets require SigV4 "
                f"('s3v4'/'v4'). Use ExperimentStore.build() or pass a SigV4 client.")

    # ---- producer side -----------------------------------------------------------------
    def log(self, alias: str, *, chip: str, region: str, workload_id: str,
            metrics: dict | None = None, params: dict | None = None,
            tags: dict | None = None, artifacts: list[str] | None = None) -> str:
        """Register one run. Order: validate identity + content (cheap, before any upload) ->
        create run (get run_id) -> upload artifacts under <alias>/<run_id>/ (content-verified) ->
        log content + artifacts_uri + FINISH. If anything after run creation fails, the partial S3
        prefix is best-effort deleted and the run is marked FAILED (so it is neither a storage leak
        nor mistaken for complete); the exception propagates. Returns the MLflow run_id."""
        ids.validate_identity(alias=alias, chip=chip, region=region, workload_id=workload_id)
        # Auto-inject the namespace grouping tag (caller's explicit namespace tag wins).
        if self.namespace and (tags is None or ids.NAMESPACE_TAG not in tags):
            tags = {ids.NAMESPACE_TAG: self.namespace, **(tags or {})}
        validate_content(metrics, tags)
        run_id = self.mlflow.create_run(alias=alias, chip=chip, region=region, workload_id=workload_id)
        uri = s3_layout.artifacts_uri(self.trace_bucket, alias, run_id)
        try:
            up = s3_layout.upload_artifacts(self.s3_client, self.trace_bucket, alias, run_id, artifacts)
            self.mlflow.finalize_run(run_id, artifacts_uri=up["uri"],
                                     metrics=metrics, params=params, tags=tags)
        except Exception:
            # Remove any objects already written so the janitor never has to find an orphan prefix
            # (the run has no artifacts_uri tag on the failure path, so nothing else could name it).
            try:
                s3_layout.delete_prefix(self.s3_client, uri)
            except Exception:  # noqa: BLE001 - cleanup is best-effort; original error wins
                _log.warning("failed to clean up partial artifacts at %s", uri, exc_info=True)
            self.mlflow.fail_run(run_id)
            raise
        return run_id

    # ---- consumer side -----------------------------------------------------------------
    def resolve(self, alias_or_id: str, *, by: str = "alias",
                include_unfinished: bool = False) -> list[RunRef]:
        """Find runs. ``by="alias"`` returns runs under the alias; ``by="id"`` returns the single
        run with that MLflow run_id. By default only usable runs are returned — FINISHED AND NOT
        soft-deleted. A soft-deleted run keeps status FINISHED (see RunRef), and ``by="id"`` uses
        get_run which, unlike search_runs, returns deleted runs too; without the is_deleted filter a
        consumer could resolve a run the janitor has already purged and then locate() an empty dir.
        Pass ``include_unfinished=True`` for diagnostics/GC (returns everything, unfiltered)."""
        if by == "id":
            runs = [self.mlflow.resolve_run(alias_or_id)]
        elif by == "alias":
            runs = self.mlflow.resolve_alias(alias_or_id)
        else:
            raise ValueError(f"by must be 'alias' or 'id', got {by!r}")
        return runs if include_unfinished else [r for r in runs if r.is_finished and not r.is_deleted]

    def search(self, *, alias: str | None = None, filter_string: str = "",
               order_by: list[str] | None = None, max_results: int = 1000,
               include_unfinished: bool = False) -> list[RunRef]:
        """Find runs by MLflow filter over the OPEN tags/metrics/params a producer logged — e.g.
        group an experiment's iterations with a ``run_no`` tag and query one, or collect every run
        of a sweep by ``tags.sweep``. ``alias`` scopes to one experiment (None = all). Only usable
        (FINISHED, not soft-deleted) runs by default; ``include_unfinished`` returns everything."""
        runs = self.mlflow.search(alias=alias, filter_string=filter_string, order_by=order_by,
                                  max_results=max_results)
        return runs if include_unfinished else [r for r in runs if r.is_finished and not r.is_deleted]

    def locate(self, run: RunRef) -> str:
        """The Pod-local directory of this run's artifacts on the mounted bucket (S3 Files) — no
        copy. Requires mount_base; raises if the run's artifacts live in a different bucket than
        this store mounts (cross-region), so a plausible-but-wrong path is never returned."""
        if not self.mount_base:
            raise ValueError("mount_base is not set; use download() for non-mounted access")
        if not run.artifacts_uri:
            raise LookupError(f"run {run.run_id} has no artifacts_uri")
        return s3_layout.local_dir_for_mount(run.artifacts_uri, self.mount_base,
                                             expected_bucket=self.trace_bucket)

    def download(self, run: RunRef, dest_dir: str) -> list[str]:
        """FALLBACK: copy a run's artifacts to a local dir (non-mounted / cross-region)."""
        if not run.artifacts_uri:
            raise LookupError(f"run {run.run_id} has no artifacts_uri")
        os.makedirs(dest_dir, exist_ok=True)
        return s3_layout.download_prefix(self.s3_client, run.artifacts_uri, dest_dir)

    def hold(self, run: RunRef) -> str:
        """Exempt a run's blobs from GC by writing the retention marker under its prefix. The hold
        survives the run's soft-deletion (it is an S3 sidecar object, not an MLflow tag). Requires
        s3:PutObject on the trace bucket. Returns the marker key."""
        if not run.artifacts_uri:
            raise LookupError(f"run {run.run_id} has no artifacts_uri")
        return s3_layout.write_retention_marker(self.s3_client, run.artifacts_uri)

    # ---- lifecycle / GC (needs s3:DeleteObject) ----------------------------------------
    def purge(self, run: RunRef, *, delete_mlflow_run: bool = True) -> int:
        """Delete a *known* run's S3 artifacts (and, by default, soft-delete the MLflow run).
        Returns the number of S3 objects deleted. This is the deliberate, single-run delete path
        (e.g. an ops CLI retiring a specific run). NOTE: the GC janitor does NOT call this — it
        works from bare S3 prefixes with no resolved RunRef, so it calls s3_layout.delete_prefix
        directly; keep the two delete callers in mind when changing delete semantics."""
        deleted = 0
        if run.artifacts_uri:
            deleted = s3_layout.delete_prefix(self.s3_client, run.artifacts_uri)
        if delete_mlflow_run:
            self.mlflow.delete_run(run.run_id)
        return deleted
