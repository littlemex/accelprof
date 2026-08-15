"""The one thing this platform fixes: the identity / linking contract.

Everything a producer logs is open (arbitrary metrics/params/tags/artifacts), EXCEPT a small
reserved set of keys that make a run *findable* and *joinable* across accelerators and regions.
This module defines and validates only that reserved set. If a run is missing or malforms a
reserved key it is not analyzable later (you cannot resolve it by alias, or you cannot tell GPU
from Neuron), so we fail loudly at log time rather than silently accept an unjoinable run.

Reserved keys (MLflow tags on every run):
  exp.alias      user-chosen handle; ALSO the MLflow experiment name, so resolve(alias) is a
                 single get_experiment_by_name — no cross-experiment scan.
  chip           "gpu" | "neuron"
  region         AWS region the run executed in
  workload_id    what was run (model/case); GPU and Neuron runs of the same workload join on this
  artifacts_uri  s3://<bucket>/<alias>/<run_id>/ — where this run's blobs live (set after upload)
  schema_version this contract's version

Content (metrics/params/tags beyond the reserved set) is passthrough — see mlflow_io.
"""
from __future__ import annotations

import re

SCHEMA_VERSION = "1"

ALIAS_TAG = "exp.alias"
CHIP_TAG = "chip"
REGION_TAG = "region"
WORKLOAD_TAG = "workload_id"
ARTIFACTS_URI_TAG = "artifacts_uri"
SCHEMA_VERSION_TAG = "schema_version"
# Conventional, AUTO-INJECTED grouping tag (e.g. "namespace=ddp") — the store stamps it from its
# configured namespace so every run is grouped/searchable without the producer passing it each
# time. Deliberately NOT reserved: a caller MAY override it per-run (their explicit tag wins).
NAMESPACE_TAG = "namespace"

# artifacts_uri is set by the store AFTER the run_id (and thus the S3 prefix) is known, so it is
# validated separately, not required from the caller of log().
RESERVED_TAGS = (ALIAS_TAG, CHIP_TAG, REGION_TAG, WORKLOAD_TAG, ARTIFACTS_URI_TAG, SCHEMA_VERSION_TAG)

CHIPS = ("gpu", "neuron")

# alias / workload_id land in an S3 key prefix and an MLflow experiment name, so restrict them to
# a filesystem/URI-safe charset (no spaces, slashes, quotes) — this prevents both a broken S3
# layout and MLflow filter-string trouble downstream.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_REGION = re.compile(r"^[a-z0-9-]{1,32}$")


def validate_identity(*, alias: str, chip: str, region: str, workload_id: str) -> None:
    """Validate the caller-supplied identity before a run is created / anything is uploaded.

    Raises ValueError with an actionable message on the first problem.
    """
    if not isinstance(alias, str) or not _SAFE_ID.match(alias):
        raise ValueError(
            f"alias must match {_SAFE_ID.pattern} (URI/FS-safe, no spaces/slashes), got {alias!r}")
    if chip not in CHIPS:
        raise ValueError(f"chip must be one of {CHIPS}, got {chip!r}")
    if not isinstance(region, str) or not _SAFE_REGION.match(region):
        raise ValueError(f"region must match {_SAFE_REGION.pattern}, got {region!r}")
    if not isinstance(workload_id, str) or not _SAFE_ID.match(workload_id):
        raise ValueError(
            f"workload_id must match {_SAFE_ID.pattern} (URI/FS-safe), got {workload_id!r}")


def creation_tags(*, alias: str, chip: str, region: str, workload_id: str) -> dict[str, str]:
    """Reserved tags known at run-creation time (everything except artifacts_uri, which is set
    after the run_id — and thus the S3 prefix — exists). This module is the single owner of both
    the reserved key NAMES and their assembly, so adding a reserved key here reaches the writer
    without a second edit in mlflow_io (see the platform design)."""
    return {
        ALIAS_TAG: alias,
        CHIP_TAG: chip,
        REGION_TAG: region,
        WORKLOAD_TAG: workload_id,
        SCHEMA_VERSION_TAG: SCHEMA_VERSION,
    }


def reserved_tags(*, alias: str, chip: str, region: str, workload_id: str,
                  artifacts_uri: str) -> dict[str, str]:
    """The full reserved tag set (creation tags + artifacts_uri)."""
    return {**creation_tags(alias=alias, chip=chip, region=region, workload_id=workload_id),
            ARTIFACTS_URI_TAG: artifacts_uri}


def run_name(chip: str, workload_id: str) -> str:
    """The MLflow run-name convention (cosmetic; kept here next to the other fixed conventions)."""
    return f"{chip}:{workload_id}"


def is_reserved(tag_key: str) -> bool:
    return tag_key in RESERVED_TAGS
