"""Runtime configuration for the analysis MCP (the pod-side profiling-analysis server).

Everything comes from the environment so the same image runs as a pod (Pod Identity, S3 Files
mounted) or locally (developer IAM, a local dir standing in for the mount). Nothing is hardcoded
to one account/region/bucket. The MCP_* names + their parsing are the shared deployment contract
in experiment_store.env (the janitor uses the same loader).

The analysis MCP resolves an experiment alias to its runs (MLflow, via experiment_store),
locates each run's artifacts ON THE MOUNTED trace bucket (S3 Files — no download), runs a
per-chip analyzer over those Pod-local files, and returns ADVICE. It never ships artifacts to
the caller.

The mount is OPTIONAL: a metadata-only deployment (list_runs/compare, MLflow-only) runs without
the S3 Files stack. When ``mount_base`` is unset, ``stage``/``analyze`` raise a clear error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from experiment_store import env

from .analyzers import Analyzer, build_analyzer

_ANALYZER_TIMEOUT_ENV = "MCP_ANALYZER_TIMEOUT_SECONDS"
# JSON name -> spec. spec is an argv list (command sugar) or an object discriminated by "type":
#   {"nsys-stats": ["nsys","stats","--sqlite","{tmp}/e.sqlite","{file:*.nsys-rep}"],
#    "neuron": {"type":"server","start":[...],"ready_port":3002,"query":["curl",...]}}
_ANALYZERS_ENV = "MCP_ANALYZERS"
_PORT_ENV = "MCP_PORT"
_DEFAULT_ANALYZER_TIMEOUT = 900
_DEFAULT_PORT = 8080


@dataclass(frozen=True)
class Config:
    tracking_uri: str
    region: str
    trace_bucket: str
    mount_base: str | None = None
    analyzer_timeout_seconds: int = _DEFAULT_ANALYZER_TIMEOUT
    port: int = _DEFAULT_PORT
    # name -> ready-to-run Analyzer; a deployment registers its tool analyzers (command or server
    # type) without a code change (fixed IDs / open content). The tool binaries come from a
    # tool-layered image; here we only carry the how-to-run specs (built into Analyzers).
    analyzers: dict[str, Analyzer] = field(default_factory=dict)


def _parse_analyzers(raw: str) -> dict[str, Analyzer]:
    """Parse MCP_ANALYZERS (JSON name -> spec) into ready Analyzers via analyzers.build_analyzer.
    Fails loud on a bad shape, unknown type, or a malformed placeholder token so a typo/embed never
    silently reaches the tool as a literal."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{_ANALYZERS_ENV} must be JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"{_ANALYZERS_ENV} must be a JSON object of name -> spec")
    return {name: build_analyzer(name, spec) for name, spec in obj.items()}


def load_config(env_map: dict[str, str] | None = None) -> Config:
    e = env_map if env_map is not None else env.os_environ()
    tracking = env.require(e, env.TRACKING_ENV, "the MLflow App / tracking-server ARN")
    region = env.region(e)
    bucket = env.require(e, env.TRACE_BUCKET_ENV, "the S3 Files-mounted trace bucket")
    mount_base = (e.get(env.MOUNT_BASE_ENV) or "").strip() or None
    timeout = env.optional_int(e, _ANALYZER_TIMEOUT_ENV, _DEFAULT_ANALYZER_TIMEOUT)
    if timeout < 1:
        raise ValueError(f"{_ANALYZER_TIMEOUT_ENV} must be >= 1, got {timeout}")
    port = env.optional_int(e, _PORT_ENV, _DEFAULT_PORT)
    analyzers = _parse_analyzers(e[_ANALYZERS_ENV]) if (e.get(_ANALYZERS_ENV) or "").strip() else {}
    return Config(tracking_uri=tracking, region=region, trace_bucket=bucket, mount_base=mount_base,
                  analyzer_timeout_seconds=timeout, port=port, analyzers=analyzers)
