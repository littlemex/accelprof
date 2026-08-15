# xprof — cross-accelerator profiling data & analysis

Run the same model on NVIDIA GPU and AWS Neuron, record every run under one identity and layout,
and turn a run's profiler artifacts into implementation advice from a laptop — the multi-gigabyte
artifacts are read in place and stay in the cluster.

xprof fixes only what must be stable for producers and consumers to interoperate — an experiment's
identity and its S3 + MLflow layout — and leaves the rest open: which metrics you record, which
profiling tools you run, and which reports you build are yours to decide.

It is one of three MCPs an agent connects to, each with a single responsibility: **MLflow MCP**
searches experiments and runs, this **analysis MCP** resolves a run to its profile files on the
mount and analyzes them, and **[xprof-knowledge](https://github.com/littlemex/xprof-knowledge)**
serves the GPU/Neuron tuning know-how. Deployment (Helm charts, Terraform, and MCP hosting) lives in
the **distributed-ai** repository, which consumes this repo's container image.

## The one principle: fixed IDs / open content

A run is addressed by a small set of reserved tags — alias, chip, region, workload id, artifacts
URI, schema version — and stored under a deterministic prefix, `s3://<bucket>/<alias>/<run_id>/`,
with MLflow as the searchable index. Everything else (metrics, parameters, free-form tags, and the
set of artifact files) is recorded verbatim; the platform never enumerates or normalizes it. This
is what lets a GPU run and a Neuron run — measured by different tools that report different keys —
be searched and compared side by side without the platform deciding in advance what "comparable"
means.

## Components

```
producer (any tool)                          laptop
  experiment_store.log(...)                    analysis MCP over kubectl port-forward
        |                                          |  pass an alias, get advice back
        v                                          v
        trace bucket  +  MLflow index  +  S3 Files mount (read-only, in-place reads)
```

| Component | Path | Role |
|---|---|---|
| `experiment_store` | [`experiment_store/`](experiment_store/) | The fixed IDs / open content library every producer and consumer imports. |
| `analysis-mcp` | [`analysis_mcp/`](analysis_mcp/) | A CPU Pod (no accelerator) that reads a run's artifacts on the mount and returns advice — a text finding, never the artifact bytes. |
| examples | [`examples/`](examples/) | Reference producers, not the contract: the collection you attach to a real workload (an nsys producer, a CPU-only Neuron compile recipe, a benchmark-iteration template). |

Deployment — the Helm charts, the S3 Files mount, and the Terraform for the trace buckets, MLflow,
and IAM — lives in the **distributed-ai** repository, not here. For the architecture and the S3
Files mount design, see [`docs/PLATFORM.md`](docs/PLATFORM.md).

## Quickstart

The library is imported, not installed; run from `infra/bench` (or add it to `PYTHONPATH`) with
Python 3.10+.

Record a run (producer, anywhere the trace bucket and MLflow are reachable):

```python
from experiment_store import ExperimentStore
store = ExperimentStore.build(region=REGION, trace_bucket=TRACE_BUCKET, tracking_uri=MLFLOW_URI)
store.log("llama3-8b-parity", chip="gpu", region=REGION, workload_id="prefill-bs1",
          metrics={"cosine": 0.9997}, artifacts=["/tmp/run.nsys-rep"])
```

Analyze a run (consumer, from a laptop):

```bash
kubectl port-forward svc/analysis-mcp -n mcp 8080:8080
# register http://127.0.0.1:8080/mcp as a streamable-http MCP, then:
#   list_runs("llama3-8b-parity") -> stage_run(<run_id>) -> analyze(<run_id>, "nsys-stats")
```

## Testing

```bash
pip install -r experiment_store/requirements.txt pytest moto
MLFLOW_ALLOW_FILE_STORE=true python -m pytest experiment_store/ analysis_mcp/ examples/ -q
```

Tests run offline: a mocked S3 (moto) and a file-store MLflow stand in for the cloud services, and
a temporary directory stands in for the mount.

## Cost

Every paid resource — the MLflow application and the S3 Files file system — is opt-in; see the
gating switches in [`docs/PLATFORM.md`](docs/PLATFORM.md).
