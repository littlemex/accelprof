"""FastMCP (streamable-http) analysis MCP — the Layer-2 product the laptop connects to.

Runs ON A POD in an accelerator's cluster. The laptop reaches it via ``kubectl port-forward``
(no ALB/gateway) as a remote streamable-http MCP; it passes an experiment alias and gets back
ADVICE. Artifacts are read in place off the S3 Files mount and analyzed on the Pod — zero bytes
of profile data are sent to the laptop.

FastMCP serves streamable-http natively (verified in Phase 2), so the platform's own MCP needs
no stdio->http proxy. supergateway is only for wrapping EXTERNAL stdio-only MCPs (e.g. an
upstream nsight-ai / neuron-agentic server), which is a separate integration.

    MCP_MLFLOW_TRACKING_URI=arn:... MCP_AWS_REGION=<r> MCP_TRACE_BUCKET=<bucket> \\
    MCP_MOUNT_BASE=/traces  python -m analysis_mcp.server
"""
from __future__ import annotations

from typing import Any

from experiment_store import ExperimentStore

from .config import load_config
from .service import AnalysisService

SERVER_NAME = "analysis-mcp"
DEFAULT_HOST = "0.0.0.0"  # noqa: S104 - must be reachable via the k8s Service, not just loopback


def build_service() -> tuple[AnalysisService, int]:
    config = load_config()
    store = ExperimentStore.build(region=config.region, trace_bucket=config.trace_bucket,
                                  tracking_uri=config.tracking_uri, mount_base=config.mount_base)
    service = AnalysisService(store, analyzer_timeout_s=config.analyzer_timeout_seconds,
                              extra_analyzers=config.analyzers)
    return service, config.port


def build_server(service: AnalysisService, port: int = 8080) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as e:  # base install has no MCP runtime
        raise SystemExit("the analysis MCP requires the 'mcp' extra — install it with "
                         "`pip install \"accelprof[mcp]\"`") from e

    mcp = FastMCP(SERVER_NAME, host=DEFAULT_HOST, port=port)

    @mcp.tool()
    def stage_run(run_id: str) -> dict:
        """Ensure a RUN's artifacts are readable on the Pod (S3 Files mount, no download) and
        return their local dir + file inventory. Traverses to trigger S3 Files import. (Takes a
        run_id — in this platform an 'experiment' is the alias, so the tool is named for the run.)
        Find the run_id with the MLflow MCP; this MCP does not duplicate run search.)"""
        return service.stage(run_id)

    @mcp.tool()
    def resolve_artifacts(run_id: str = "", alias: str = "", chip: str = "",
                          pattern: str = "*") -> dict:
        """Map an MLflow identity to concrete profile file path(s) on the Pod's S3 Files mount.
        Give either run_id, or alias+chip (picks the latest FINISHED run of that chip). pattern is
        a glob (e.g. '*.nsys-rep', '*.neff'). Returns the dir + matching absolute paths (metadata
        only) — hand these to an analyzer or to an external tool that reads the same mount."""
        return service.resolve_artifacts(run_id or None, alias=alias or None,
                                         chip=chip or None, pattern=pattern)

    @mcp.tool()
    def analyze(run_id: str, analyzer: str = "inventory") -> dict:
        """Analyze a run's profiler artifacts on the Pod and return advice (never the bytes).
        analyzer selects the tool (builtin 'inventory'; a deployment registers nsys/neuron via
        MCP_ANALYZERS, whose argv templates use {file:GLOB}/{files:GLOB} to target the right file)."""
        return service.analyze(run_id, analyzer=analyzer)

    return mcp


def main() -> None:
    service, port = build_service()
    build_server(service, port=port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
