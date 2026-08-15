"""Adapter for SGLang's ``python -m sglang.bench_serving --output-file`` record (JSONL).

Shows the abstraction absorbing naming differences from vLLM: SGLang names E2E latency
``*_e2e_latency_ms`` (vLLM: ``*_e2el_ms``) and the combined throughput ``total_throughput``
(vLLM: ``total_token_throughput``) — the adapter maps those onto the canonical core and passes
the rest (``input_throughput``, ``concurrency``, ``accept_length``, …) through verbatim as ``extra``.
Note: ``--output-file`` APPENDS one JSON object per run, so the file is JSONL; the loader in
run_iteration reads the LAST record."""
from __future__ import annotations

from ..metrics import BenchmarkMetrics, Stat, register


class SglangAdapter:
    framework = "sglang"

    def normalize(self, raw: dict) -> BenchmarkMetrics:
        g = raw.get
        used = {"request_throughput", "output_throughput", "total_throughput",
                "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
                "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
                "mean_e2e_latency_ms", "median_e2e_latency_ms", "p99_e2e_latency_ms",
                "completed", "duration"}
        return BenchmarkMetrics(
            framework=self.framework,
            request_throughput_req_s=g("request_throughput"),
            output_throughput_tok_s=g("output_throughput"),
            total_token_throughput_tok_s=g("total_throughput"),  # SGLang: combined in/out tok/s
            ttft_ms=Stat(g("mean_ttft_ms"), g("median_ttft_ms"), g("p99_ttft_ms")),
            tpot_ms=Stat(g("mean_tpot_ms"), g("median_tpot_ms"), g("p99_tpot_ms")),
            itl_ms=Stat(g("mean_itl_ms"), g("median_itl_ms"), g("p99_itl_ms")),
            e2e_ms=Stat(g("mean_e2e_latency_ms"), g("median_e2e_latency_ms"), g("p99_e2e_latency_ms")),
            completed=g("completed"), duration_s=g("duration"),
            # e.g. input_throughput, concurrency, accept_length, max_output_tokens_per_s, backend
            extra={k: v for k, v in raw.items() if k not in used},
        )


register(SglangAdapter())
