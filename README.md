# accelprof

[![ci](https://github.com/littlemex/accelprof/actions/workflows/ci.yml/badge.svg)](https://github.com/littlemex/accelprof/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

**Give every GPU and Neuron run one identity, then turn its profiler artifacts into advice over MCP.**

accelprof is two things that share one convention:

- **`experiment_store`** — a small library. A producer calls `log(...)` to record a run; a consumer
  calls `resolve(...)` / `locate(...)` to find it again. Runs live under a deterministic
  `s3://<bucket>/<alias>/<run_id>/` prefix with MLflow as the searchable index.
- **`accelprof-analysis-mcp`** — an MCP server. It maps a run to its artifact files on a directory it
  can read, runs an analyzer over them, and returns the finding as text. The trace itself — often
  multiple gigabytes — is read in place and never sent to the client.

It assumes no orchestrator; Kubernetes is one way to host the MCP, not a requirement.

## Fixed IDs, open content

A run is addressed by a fixed set of reserved tags — `alias`, `chip`, `region`, `workload_id`,
`artifacts_uri`, `schema_version` — and nothing else is constrained. Metrics, parameters, free-form
tags, and the set of artifact files are recorded verbatim; the store never enumerates or normalizes
them. So GPU and Neuron runs — profiled by different tools with different keys — share one index
without a fixed schema of what is "comparable".

## Install

```bash
pip install accelprof            # the experiment_store library
pip install "accelprof[mcp]"     # + the accelprof-analysis-mcp server
```

Requirements: Python 3.10+, an S3-compatible object store for artifacts, an MLflow tracking server
(self-hosted, or SageMaker-managed via the `[sagemaker]` extra), and AWS credentials on the standard
chain (environment, shared config, or an instance/Pod role). The base install imports as
`experiment_store`; the `[mcp]` extra adds the `analysis_mcp` module and the console script.

## Record a run (producer)

```python
from experiment_store import ExperimentStore

store = ExperimentStore.build(region=REGION, trace_bucket=TRACE_BUCKET, tracking_uri=MLFLOW_URI)
store.log(
    "llama3-8b-parity",                 # alias — the experiment this run belongs to
    chip="gpu", region=REGION, workload_id="prefill-bs1",
    metrics={"cosine": 0.9997},         # open content: any keys you measure
    tags={"run_no": "1"},
    artifacts=["/tmp/run.nsys-rep"],    # uploaded under s3://<bucket>/<alias>/<run_id>/
)
```

The store sets `artifacts_uri` and `schema_version`, and a successful `log()` marks the run FINISHED.
A consumer finds the run again and reads its artifacts in place — no download. To use `locate`, pass
`mount_base=` to `build`:

```python
runs = store.resolve("llama3-8b-parity")     # FINISHED runs under the alias -> list[RunRef]
local_dir = store.locate(runs[0])            # the run's directory on the mounted bucket
```

`examples/` holds copy-and-adapt producer templates: an `nsys` GPU producer
([`examples/gpu_nsys/produce.py`](examples/gpu_nsys/produce.py)), a device-free Neuron compile recipe
([`examples/neuron_cpu_compile.py`](examples/neuron_cpu_compile.py)), and a benchmark-iteration
template ([`examples/benchmark_iteration/`](examples/benchmark_iteration/)). They are not shipped in
the wheel.

## Analyze a run (the MCP)

```bash
MCP_MLFLOW_TRACKING_URI=<uri> MCP_AWS_REGION=<region> MCP_TRACE_BUCKET=<bucket> \
  MCP_MOUNT_BASE=/path/to/artifacts accelprof-analysis-mcp     # streamable-http on MCP_PORT (8080)
```

| Env var | Meaning |
|---|---|
| `MCP_MLFLOW_TRACKING_URI` | MLflow tracking server — URI or SageMaker MLflow ARN (required) |
| `MCP_AWS_REGION` | AWS region of the object store (required) |
| `MCP_TRACE_BUCKET` | bucket holding the run artifacts (required) |
| `MCP_MOUNT_BASE` | directory where `<alias>/<run_id>/` files are readable — required by the file-reading tools (`stage_run` / `resolve_artifacts` / `analyze`) |
| `MCP_PORT` | listen port (default `8080`) |
| `MCP_ANALYZERS` | JSON map registering extra `command`/`server` analyzers (e.g. nsys, neuron) |

Register it with any MCP client — for example Claude Code:

```bash
claude mcp add --transport http accelprof http://127.0.0.1:8080/mcp
```

The server exposes three tools. Run discovery is intentionally not among them — find the `run_id`
with the MLflow MCP (see [`docs/mlflow-mcp.md`](docs/mlflow-mcp.md)), then:

| Tool | What it does |
|---|---|
| `stage_run(run_id)` | Ensure the run's artifacts are readable under `MCP_MOUNT_BASE` (triggers a lazy mount import, no copy) and return the local dir + file inventory. |
| `resolve_artifacts(run_id \| alias+chip, pattern="*")` | Return the absolute path(s) of artifact files matching a glob (e.g. `*.nsys-rep`, `*.neff`); `alias+chip` picks the latest FINISHED run of that chip. Hand the paths to an analyzer or any tool reading the same mount. |
| `analyze(run_id, analyzer="inventory")` | Run an analyzer over the staged dir and return the finding as text. |

The built-in `inventory` analyzer needs no external tool and confirms the stage → analyze path end
to end:

```jsonc
{
  "run_id": "a1b2c3…", "chip": "gpu", "analyzer": "inventory",
  "dir": "/path/to/artifacts/llama3-8b-parity/a1b2c3…",
  "advice": "inventory of …:\n  run.nsys-rep\t734003200\ntotal_files=1 total_bytes=734003200"
}
```

A deployment registers real tools through `MCP_ANALYZERS` — a JSON map of name → spec, where a spec
is an argv list (a `command` analyzer) or an object with `type: command | server`. Argv templates use
`{dir}` / `{file:GLOB}` / `{files:GLOB}` placeholders, the tool runs without a shell, and its stdout
becomes the advice:

```bash
MCP_ANALYZERS='{"nsys-stats": ["nsys","stats","{file:*.nsys-rep}"]}'   # type defaults to "command"
```

A `server`-type spec (`{"type":"server","start":[…],"ready_port":3002,"query":[…]}`) drives a tool
that serves results rather than printing them (e.g. `neuron-explorer` for Neuron).

How `MCP_MOUNT_BASE` gets populated is your choice: an S3 Files read-only mount, an NFS export, or a
local sync.

## Testing

```bash
pip install -e ".[mcp,test]"
MLFLOW_ALLOW_FILE_STORE=true python -m pytest experiment_store/ analysis_mcp/ examples/ -q
```

The suite runs with no cloud: a mocked S3 (moto) and a file-store MLflow stand in for the services,
and a temporary directory stands in for the artifact mount — so you can exercise the full
log → resolve → stage → analyze path before wiring anything real. (`MLFLOW_ALLOW_FILE_STORE` is this
repo's own opt-in to run the tests against MLflow's file store.)

## Hosting

`Dockerfile.analysis-mcp` is a reference image that installs the package and runs the console script;
`Dockerfile.analysis-mcp-nsys` layers the Nsight Systems CLI so an `nsys stats` analyzer can be
registered, and `Dockerfile.neuron-cc` layers the Neuron compiler for the device-free compile recipe.
Provisioning the object store, MLflow, and the artifact mount, and hosting this MCP alongside others,
is handled by a separate deployment repo (`distributed-ai`). See [`docs/PLATFORM.md`](docs/PLATFORM.md)
for the architecture.

The server authenticates to S3 and MLflow via the standard AWS credential chain, and serves plain
streamable-http with no built-in authentication — bind it to loopback and reach it with
`kubectl port-forward`, or front it with your own auth layer. `stage_run` only verifies the files are
readable (triggering a lazy S3 mount import if the backend needs one); it never copies data.

## Related projects

- **[accelprof-knowledge](https://github.com/littlemex/accelprof-knowledge)** — a knowledge MCP of
  GPU/Neuron tuning playbooks. Pair it with this one to turn a finding into a next step. When hosting
  both, give each its own `MCP_PORT`.
- The official **MLflow MCP** (`pip install "mlflow[mcp]"`, `mlflow mcp run`) — run discovery and
  search, which accelprof deliberately does not duplicate.

## Contributing & license

Add a profiling tool by registering a `command`/`server` analyzer via `MCP_ANALYZERS` — no code
change. Add a serving framework to the benchmark example by dropping an adapter under
[`examples/benchmark_iteration/adapters/`](examples/benchmark_iteration/adapters/). Keep the test
command above green.

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
