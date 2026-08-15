# benchmark_iteration — a worked example + template for the iterate loop

This is an **example, not the contract**. The platform (`experiment_store`) fixes only identity +
layout; *what* you measure and *how* you analyze it are yours. This directory is a copy-and-adapt
starting point for the real loop:

```
profile (nsys / neuron-explorer via analysis-mcp)  ->  serve the model  ->  benchmark it
   ->  LOG the run (this example)  ->  search / compare / analyze  ->  change one thing  ->  repeat
```

Each turn of the loop is one **run**, grouped so you can find and compare them:
`alias` = the experiment, `run_no` = the iteration number, `namespace` = auto-stamped grouping, plus
any **free tags** you attach (`sweep=qps`, `qps=8`, `tp=2`, …). All are searchable.

## The framework-agnostic metrics abstraction (`metrics.py`)

Different serving frameworks report the same concepts under different names (vLLM's `p99_e2el_ms`
vs SGLang's `p99_e2e_latency_ms`; vLLM's combined `total_token_throughput` vs SGLang's split
`input_throughput`/`output_throughput`). We do **not** force a framework to change how it measures.
Instead:

- a small **common core** (`BenchmarkMetrics`: throughput / TTFT / TPOT / ITL / E2E latency,
  every field optional) gives canonical names to the cross-framework-comparable metrics;
- a per-framework **adapter** (`adapters/<framework>.py`) maps that tool's keys onto the core;
- everything the tool measured that has no canonical slot is passed through verbatim in `extra`
  (numeric extras become `extra_*` metrics; the rest become `extra_*` tags), and the raw output is
  kept as an artifact.

So `compare` / `search_runs` work across frameworks (canonical keys), while nothing the framework
uniquely measures is lost or constrained.

## Add your framework (the plug-in point)

Drop `adapters/<yourframework>.py`:

```python
from ..metrics import BenchmarkMetrics, Stat, register

class MyAdapter:
    framework = "myframework"
    def normalize(self, raw: dict) -> BenchmarkMetrics:
        g = raw.get
        return BenchmarkMetrics(
            framework=self.framework,
            output_throughput_tok_s=g("tokens_per_sec"),
            ttft_ms=Stat(g("ttft_avg"), g("ttft_p50"), g("ttft_p99")),
            # ... map what maps; the rest is passthrough:
            extra={k: v for k, v in raw.items() if k not in {...mapped keys...}},
        )

register(MyAdapter())
```

then import it in `adapters/__init__.py`. No platform change.

## Produce the numbers (SGLang is the default measurement tool)

The platform does not measure for you — you run a real load against your running server and hand the
result to `run_iteration.py`. This example uses **SGLang's tooling**, because it ships both a
serving-performance benchmark and an accuracy eval as first-class, framework-agnostic clients (they
speak the OpenAI/HTTP API, so they measure *any* OpenAI-compatible server — SGLang or vLLM):

```bash
# Performance — SGLang bench_serving (TTFT/TPOT/ITL/E2E percentiles, throughputs; PD-disaggregation
# aware via --pd-separated; can trigger the server torch profiler via --profile). Appends JSONL:
python -m sglang.bench_serving --backend sglang --host <host> --port 30000 \
  --dataset-name random --num-prompts 100 --random-input-len 256 --random-output-len 128 \
  --output-file /out/sglang_result.jsonl

# Accuracy — SGLang run_eval, a lightweight client against the served endpoint. ~10 built-in
# benchmarks: mmlu, math, gsm8k, mgsm, gpqa, humaneval, longbench_v2, mmmu, aime25, ...
# Pass --host/--port, not --base-url (run_eval appends the OpenAI path itself). It uses the OpenAI
# client, so set a dummy key.
OPENAI_API_KEY=dummy python -m sglang.test.run_eval --model <served-model> \
  --eval-name mmlu --num-examples 200 --host <host> --port 30000   # writes {"score": ...}
```

Write the perf `--output-file` to node-local or POSIX storage (an emptyDir or the pod's own disk),
since `bench_serving` appends to it, and use a fresh file per run so "last record" unambiguously
means this run. `run_iteration` accepts
`--framework` (serving framework → the searchable tag) separately from `--tool` (result-schema
adapter, defaults to `--framework`) — set `--tool sglang --framework vllm` when SGLang's client
measured a vLLM server.

For a much larger accuracy task set, `lm-eval` (lm-evaluation-harness) in server mode works against
the same endpoint and is framework-agnostic:
`lm-eval --model local-completions --tasks gsm8k --model_args model=<m>,base_url=http://<host>:30000/v1/completions`.

**Where to run the client.** The perf and eval clients import the framework package, which pulls
in torch, so there is no torch-free client to `pip install`; run the client from the serving image
itself. That image is large (its CUDA layers extract to well over 10 GB), so schedule the client
onto a nodepool whose nodes have room for it — `nodeSelector` the GPU nodepool and tolerate its
`nvidia.com/gpu` taint, but omit the `nvidia.com/gpu` resource so the client takes no accelerator.
Landing on the same node as the server reuses the cached image with no re-pull. Two caveats:
co-locating client and server shares CPU and network, which makes latencies optimistic (fine for
relative sweeps, not absolute SLOs); and container image layers live on the node's local disk, never
on a shared file system (see `docs/PLATFORM.md` for the storage-tier rules).

`adapters/vllm.py` is a second adapter kept as an illustration: its schema differs (`*_e2el_ms`,
`total_token_throughput`) and the adapter maps it onto the same canonical core. If you serve vLLM,
SGLang's `bench_serving --backend vllm` can measure it too.

## Run one iteration

Inside the platform image (`PYTHONPATH=infra/bench`), with the deployment env set
(`MCP_MLFLOW_TRACKING_URI`, `MCP_AWS_REGION`, `MCP_TRACE_BUCKET`, and — for the auto namespace tag
— `EXPERIMENT_NAMESPACE` or the k8s `POD_NAMESPACE` via the downward API):

```bash
python -m examples.benchmark_iteration.run_iteration \
  --alias qwen-serving --chip gpu --region <region> --workload-id sharegpt-qps8 \
  --framework sglang --result /out/sglang_result.jsonl --run-no 10 \
  --eval-result /out/mmlu.json --tag sweep=qps --tag qps=8
```

That normalizes the perf result (SGLang's JSONL — the last record is used), pulls the accuracy score
out of the eval result, logs a run (metrics + `run_no`/`framework`/`namespace`/your tags + accuracy),
and keeps both raw files as artifacts under the run's S3 prefix. Accuracy is a separate concern (an
eval harness, not the serving benchmark): pass `--eval-result` (sglang run_eval / lm-eval) to extract
it, or `--accuracy <float>` if you already have it.

## Query + analyze (build your side here)

- **Find iterations**: `search_runs(filter="tags.run_no = '10'")`,
  `search_runs(filter="tags.sweep = 'qps'", order_by="metrics.output_throughput_tok_s DESC")`.
- **Compare accelerators**: `compare(alias)` puts the latest run per chip side by side.
- **Profile a run's artifacts**: `analyze(run_id, "nsys-stats")` for a GPU trace, `neuron-summary`
  for a Neuron `.neff`/`.ntff` pair.
- **Reuse the official MLflow MCP** for raw experiment/run browsing (`mlflow mcp run`); it
  complements `search_runs`. See [`../../docs/mlflow-mcp.md`](../../docs/mlflow-mcp.md).

Your ETL, dashboards, and regression gates build on top of these; that layer is yours. This
directory gives you a working starting point.
