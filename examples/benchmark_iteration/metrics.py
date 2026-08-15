"""Framework-agnostic serving-benchmark metrics — the abstraction that lets you compare vLLM vs
sglang vs anything else WITHOUT fixing or constraining how each tool measures.

Design (deliberately thin, so it never becomes a straitjacket):
  * A small COMMON CORE of cross-framework metrics (throughput / TTFT / TPOT / ITL / E2E latency).
    These are the fields you actually want to compare across frameworks, so they get canonical
    names + units here. Every field is Optional — a tool that doesn't emit one just leaves it None.
  * PASSTHROUGH of everything else: a tool's own fields it measured that have no canonical slot are
    kept verbatim in ``extra`` (and the whole raw output is kept as an artifact). We do not drop or
    normalize them — the platform is "fixed IDs, open content", and so is this.
  * A per-framework ADAPTER maps that tool's idiosyncratic keys onto the common core. Adding a
    framework = adding an adapter (see adapters/); the platform code never changes.

``as_metrics()`` flattens the common core to the flat float dict MLflow wants (dropping None), so
runs are directly comparable/searchable regardless of which tool produced them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


def _finite(v) -> bool:
    """A value is loggable as a flat metric iff it is a real (non-bool) number that is finite.
    The flat-metrics contract here is 'finite float' — NaN/inf never belong in a comparable metric
    (a failed run can produce NaN latency, and JSON parsing admits NaN/Infinity), so they are kept
    out of the metric dict at the source rather than blowing up the whole run at log time."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class Stat:
    """A latency distribution in milliseconds (any subset may be None)."""
    mean: float | None = None
    p50: float | None = None
    p99: float | None = None

    def flatten(self, prefix: str) -> dict[str, float]:
        return {f"{prefix}_{k}": float(v) for k, v in
                (("mean", self.mean), ("p50", self.p50), ("p99", self.p99)) if _finite(v)}


@dataclass(frozen=True)
class BenchmarkMetrics:
    framework: str
    # throughput
    request_throughput_req_s: float | None = None
    output_throughput_tok_s: float | None = None
    total_token_throughput_tok_s: float | None = None
    # latency distributions (ms)
    ttft_ms: Stat = field(default_factory=Stat)   # time to first token
    tpot_ms: Stat = field(default_factory=Stat)   # time per output token
    itl_ms: Stat = field(default_factory=Stat)    # inter-token latency
    e2e_ms: Stat = field(default_factory=Stat)    # end-to-end request latency
    # run shape
    completed: float | None = None
    duration_s: float | None = None
    # accuracy is a SEPARATE concern (eval harness, not the serving benchmark); optional single value
    accuracy: float | None = None
    # anything the tool measured that has no canonical slot — kept verbatim, never dropped
    extra: dict = field(default_factory=dict)

    def as_metrics(self) -> dict[str, float]:
        """Flat float dict of finite floats (the metric contract here). Canonical names, so
        cross-framework compare just works. Non-finite values are dropped at the source — including
        in the canonical core (a failed run can yield NaN latency): a metric must be a finite float,
        and a bad number must never make the whole run unloggable. Non-finite/non-numeric extras that
        are still worth keeping (e.g. request_rate=inf) are surfaced as tags by run_iteration."""
        m: dict[str, float] = {}
        for k, v in (("request_throughput_req_s", self.request_throughput_req_s),
                     ("output_throughput_tok_s", self.output_throughput_tok_s),
                     ("total_token_throughput_tok_s", self.total_token_throughput_tok_s),
                     ("completed", self.completed), ("duration_s", self.duration_s),
                     ("accuracy", self.accuracy)):
            if _finite(v):
                m[k] = float(v)
        m.update(self.ttft_ms.flatten("ttft_ms"))
        m.update(self.tpot_ms.flatten("tpot_ms"))
        m.update(self.itl_ms.flatten("itl_ms"))
        m.update(self.e2e_ms.flatten("e2e_ms"))
        for k, v in self.extra.items():               # numeric extras are comparable too
            if _finite(v):
                m[f"extra_{k}"] = float(v)
        return m


class MetricAdapter(Protocol):
    """Map one framework's raw benchmark output (parsed JSON) to BenchmarkMetrics."""
    framework: str
    def normalize(self, raw: dict) -> BenchmarkMetrics: ...


# framework name -> adapter. Register your own with @register (see adapters/).
_REGISTRY: dict[str, MetricAdapter] = {}


def register(adapter: MetricAdapter) -> MetricAdapter:
    _REGISTRY[adapter.framework] = adapter
    return adapter


def get_adapter(framework: str) -> MetricAdapter:
    if framework not in _REGISTRY:
        raise ValueError(f"no metric adapter for {framework!r}; registered: {sorted(_REGISTRY)}. "
                         f"Add one under adapters/ (see README).")
    return _REGISTRY[framework]


def registered() -> list[str]:
    return sorted(_REGISTRY)
