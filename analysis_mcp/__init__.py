"""analysis_mcp — the pod-side profiling-analysis MCP.

Resolves an experiment alias, reads its artifacts in place off the S3 Files mount, runs a
per-run analyzer on the Pod, and returns advice (never the artifact bytes).
"""
from .config import Config, load_config
from .service import AnalysisService

__all__ = ["Config", "load_config", "AnalysisService"]
