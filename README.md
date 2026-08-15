# xprof — cross-accelerator experiment store + profiling-analysis MCP

[![ci](https://github.com/littlemex/xprof/actions/workflows/ci.yml/badge.svg)](https://github.com/littlemex/xprof/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

Record every GPU and Neuron run under one identity and layout, then turn a run's profiler artifacts
into implementation advice over MCP — the multi-gigabyte artifacts are read in place and never
copied to the client.

xprof is self-contained software: a Python library plus an MCP server. It depends on nothing but
Python and the services *you already use* — an object store, an MLflow tracking server, and a
directory where a run's artifacts are readable. It assumes no particular orchestrator; running it on
Kubernetes is one deployment option, not a requirement.

xprof fixes only what must be stable for producers and consumers to interoperate — an experiment's
identity and its S3 + MLflow layout — and leaves the rest open: which metrics you record, which
profiling tools you run, and which reports you build are yours to decide.

## The one principle: fixed IDs / open content

A run is addressed by a small set of reserved tags — alias, chip, region, workload id, artifacts
URI, schema version — and stored under a deterministic prefix, `s3://<bucket>/<alias>/<run_id>/`,
with MLflow as the searchable index. Everything else (metrics, parameters, free-form tags, and the
set of artifact files) is recorded verbatim; xprof never enumerates or normalizes it. This is what
lets a GPU run and a Neuron run — measured by different tools that report different keys — be
searched and compared side by side without deciding in advance what "comparable" means.

## Two pieces

The PyPI distribution is **`xprof-store`** (the name `xprof` is taken); the import packages and the
console script keep the `xprof` name.

| Piece | Import / command | Role |
|---|---|---|
| `experiment_store` | `pip install xprof-store` | The fixed-IDs / open-content library a producer imports to log a run and a consumer imports to resolve one. Depends only on boto3 + MLflow (add the `[sagemaker]` extra for a SageMaker-managed MLflow). |
| analysis MCP | `pip install "xprof-store[mcp]"` → `xprof-analysis-mcp` | An MCP server that maps a run to its artifact files in a local directory and runs an analyzer over them, returning advice — a text finding, never the bytes. |

`examples/` holds copy-and-adapt producer templates (an nsys producer, a CPU-only Neuron compile
recipe, a benchmark-iteration template); they are not shipped in the wheel.

## Record a run (producer)

```bash
pip install xprof-store     # library only — no MCP server pulled in
```
```python
from experiment_store import ExperimentStore
store = ExperimentStore.build(region=REGION, trace_bucket=TRACE_BUCKET, tracking_uri=MLFLOW_URI)
store.log("llama3-8b-parity", chip="gpu", region=REGION, workload_id="prefill-bs1",
          metrics={"cosine": 0.9997}, tags={"run_no": "1"}, artifacts=["/tmp/run.nsys-rep"])
```

## Analyze a run (the MCP)

```bash
pip install "xprof-store[mcp]"
MCP_MLFLOW_TRACKING_URI=<uri> MCP_AWS_REGION=<region> MCP_TRACE_BUCKET=<bucket> \
  MCP_MOUNT_BASE=/path/to/artifacts xprof-analysis-mcp     # serves streamable-http on MCP_PORT (8080)
```

| Env var | Meaning |
|---|---|
| `MCP_MLFLOW_TRACKING_URI` | MLflow tracking server (URI or SageMaker MLflow ARN) — required |
| `MCP_AWS_REGION` | AWS region for the object store — required |
| `MCP_TRACE_BUCKET` | the bucket holding the run artifacts — required |
| `MCP_MOUNT_BASE` | directory where `<alias>/<run_id>/` files are readable — optional (omit for metadata-only) |
| `MCP_PORT` | listen port (default 8080) |
| `MCP_ANALYZERS` | JSON map registering extra `command`/`server` analyzers (nsys/neuron/…) |

Register it in any MCP client — for example Claude Code:

```bash
claude mcp add --transport http xprof http://127.0.0.1:8080/mcp
```

Find the run's id with the MLflow MCP (xprof does not duplicate run search), then `stage_run(<id>)`
→ `analyze(<id>, "nsys-stats")`. The
MCP reads artifacts from `MCP_MOUNT_BASE` — a plain directory where `<alias>/<run_id>/` files are
readable. **How that directory is populated is your choice**: an AWS S3 Files read-only mount, an
NFS export, or a local sync. `MCP_MOUNT_BASE` is optional — omit it for a metadata-only server.

## Testing

```bash
pip install -e ".[mcp,test]"
MLFLOW_ALLOW_FILE_STORE=true python -m pytest experiment_store/ analysis_mcp/ examples/ -q
```

Tests run offline: a mocked S3 (moto) and a file-store MLflow stand in for the cloud services, and a
temporary directory stands in for the artifact directory. `MLFLOW_ALLOW_FILE_STORE=true` also lets
you drive the store against a local file-store MLflow and a local directory, with no cloud, to try
the flow before wiring real services.

## Related projects

- **[xprof-knowledge](https://github.com/littlemex/xprof-knowledge)** — the tuning-knowledge MCP;
  pair it with this one to turn an analysis into a next step. Run each on its own `MCP_PORT` (they
  both default to 8080) when hosting both.
- The official **MLflow MCP** (`pip install "mlflow[mcp]"`, `mlflow mcp run`) — run discovery and
  search, which this project deliberately does not duplicate.

## Running in a container

`Dockerfile.analysis-mcp` is a reference image that `pip install`s the package and runs the console
script; `Dockerfile.analysis-mcp-nsys` layers the Nsight Systems CLI so an `nsys stats` analyzer can
be registered. A Kubernetes deployment that hosts this MCP alongside others, and provisions the
object store / MLflow / artifact mount, is a separate concern (the `distributed-ai` deployment repo),
not a dependency of this project. See [`docs/PLATFORM.md`](docs/PLATFORM.md) for the architecture.

## Contributing & license

Add a profiling tool by registering a `command`/`server` analyzer via `MCP_ANALYZERS` (no code
change); add a serving framework to the benchmark example by dropping an adapter under
`examples/benchmark_iteration/adapters/`. Keep `MLFLOW_ALLOW_FILE_STORE=true python -m pytest
experiment_store/ analysis_mcp/ examples/ -q` green.

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
