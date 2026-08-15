"""Adapter for vLLM's benchmarks/benchmark_serving.py --save-result JSON."""
from __future__ import annotations

from ..metrics import BenchmarkMetrics, Stat, register


class VllmAdapter:
    framework = "vllm"

    def normalize(self, raw: dict) -> BenchmarkMetrics:
        g = raw.get
        used = {"request_throughput", "output_throughput", "total_token_throughput",
                "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
                "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
                "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
                "completed", "duration"}
        return BenchmarkMetrics(
            framework=self.framework,
            request_throughput_req_s=g("request_throughput"),
            output_throughput_tok_s=g("output_throughput"),
            total_token_throughput_tok_s=g("total_token_throughput"),
            ttft_ms=Stat(g("mean_ttft_ms"), g("median_ttft_ms"), g("p99_ttft_ms")),
            tpot_ms=Stat(g("mean_tpot_ms"), g("median_tpot_ms"), g("p99_tpot_ms")),
            itl_ms=Stat(g("mean_itl_ms"), g("median_itl_ms"), g("p99_itl_ms")),
            e2e_ms=Stat(g("mean_e2el_ms"), g("median_e2el_ms"), g("p99_e2el_ms")),  # vLLM: e2el
            completed=g("completed"), duration_s=g("duration"),
            extra={k: v for k, v in raw.items() if k not in used},  # e.g. total_*_tokens, goodput
        )


register(VllmAdapter())
