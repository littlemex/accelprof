"""Single source of the MCP_* deployment environment contract.

The reserved env var names and their parsing (strip, region fallback chain, actionable errors)
were duplicated — and had drifted — between analysis_mcp/config.py and the janitor entrypoint.
They live here once. This is an *entrypoint* concern: the library core (store.py) never imports
it; only the CLI/service entrypoints and the janitor do.
"""
from __future__ import annotations

import os

TRACKING_ENV = "MCP_MLFLOW_TRACKING_URI"   # arn:aws:sagemaker:<region>:<acct>:mlflow-app/app-XXXX
REGION_ENV = "MCP_AWS_REGION"
TRACE_BUCKET_ENV = "MCP_TRACE_BUCKET"
MOUNT_BASE_ENV = "MCP_MOUNT_BASE"
NAMESPACE_ENV = "EXPERIMENT_NAMESPACE"     # producer-side grouping; auto-injected as tag namespace=


def _get(env: dict[str, str], key: str) -> str:
    return (env.get(key) or "").strip()


def require(env: dict[str, str], key: str, hint: str) -> str:
    v = _get(env, key)
    if not v:
        raise ValueError(f"{key} is required ({hint})")
    return v


def region(env: dict[str, str]) -> str:
    """MCP_AWS_REGION → AWS_REGION → AWS_DEFAULT_REGION; raise with an actionable message."""
    v = _get(env, REGION_ENV) or _get(env, "AWS_REGION") or _get(env, "AWS_DEFAULT_REGION")
    if not v:
        raise ValueError(f"{REGION_ENV} (or AWS_REGION / AWS_DEFAULT_REGION) is required")
    return v


def optional_int(env: dict[str, str], key: str, default: int) -> int:
    raw = _get(env, key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from None


def namespace(env: dict[str, str]) -> str | None:
    """Producer grouping namespace: EXPERIMENT_NAMESPACE, else the k8s downward-API POD_NAMESPACE,
    else None. Passed to ExperimentStore(namespace=...) to auto-inject the namespace tag."""
    return _get(env, NAMESPACE_ENV) or _get(env, "POD_NAMESPACE") or None


def os_environ() -> dict[str, str]:
    return dict(os.environ)
