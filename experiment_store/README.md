# experiment_store — the fixed platform library

A plain Python library — imported, never a service — that is the only fixed part of the platform.
It standardizes experiment identity and the S3 + MLflow layout (the [fixed IDs / open content
principle](../README.md)) and passes everything else — metrics, parameters, tags, and the set of
artifact files — through untouched.

## Contract

Every run carries these reserved tags, validated by `ids.py`: `exp.alias`, `chip` (gpu|neuron),
`region`, `workload_id`, `artifacts_uri`, `schema_version`. `metrics`, `params`, and `tags` are
logged verbatim; the platform never enumerates a metric set. The alias is also the MLflow
experiment name, so `resolve(alias)` is a single lookup.

A run's artifacts live under `s3://<trace-bucket>/<alias>/<run_id>/`, referenced by the
`artifacts_uri` tag — the large profiler files (`.nsys-rep`, `.ncu-rep`, `.neff`/`.ntff`, HLO
protobufs) go here, not into MLflow's own artifact store.

## API

```python
from experiment_store import ExperimentStore

store = ExperimentStore.build(region=REGION, trace_bucket=TRACE_BUCKET,
                              tracking_uri=MLFLOW_URI, mount_base="/traces")

# producer
run_id = store.log("llama3-8b-parity", chip="gpu", region=REGION,
                   workload_id="prefill-bs1", metrics={"cosine": 0.9997},
                   tags={"run_no": "1"}, artifacts=["/tmp/run.nsys-rep"])

# consumer
runs = store.resolve("llama3-8b-parity")                          # or resolve(run_id, by="id")
hits = store.search(filter_string="tags.run_no = '1'")            # MLflow filter over open tags/metrics
path = store.locate(runs[0])   # <mount_base>/<alias>/<run_id>/ — an in-place read, no copy
# store.download(runs[0], "/scratch")   # fallback for a non-mounted or cross-region consumer

# retention
store.hold(runs[0])    # exempt a run's artifacts from garbage collection
store.purge(runs[0])   # delete a run's artifacts (a delete-capable role, never the reader role)
```

- **Consumers read artifacts in place** via `locate`, which returns the path under `mount_base`
  where the run's files are readable (the bucket exposed as a filesystem — an S3 Files or NFS mount,
  or any synced directory), so artifacts are not re-downloaded. `download` is a fallback only.
- **`search`** takes any MLflow filter over the open tags and metrics a producer logged
  (`"tags.run_no = '10'"`, `"metrics.tpot_ms < 20"`), with optional `alias` scoping and `order_by`.
- **The S3 client must sign with SigV4:** the trace buckets are SSE-KMS, and reads against
  KMS-encrypted objects require it. `build` configures this; a non-SigV4 client is rejected at
  construction.
- Uploads are content-verified server-side and fail closed, including multipart artifacts.

## Namespace grouping

`build(namespace=...)` — or the `EXPERIMENT_NAMESPACE` environment variable, falling back to the
pod's `POD_NAMESPACE` — stamps a `namespace` tag on every run the store logs, so a team's runs are
groupable and searchable (`tags.namespace = '...'`) without the platform defining what a namespace
means. A caller's explicit `namespace` tag wins.

## Modules

`ids.py` (reserved-key contract) · `s3_layout.py` (layout, verified transfer, list, download,
delete, retention marker) · `mlflow_io.py` (schemaless MLflow read/write) · `store.py` (the public
`ExperimentStore`) · `env.py` (shared `MCP_*` deployment-env loader) · `janitor.py` (orphan-GC
library + `python -m experiment_store.janitor` entrypoint; its operation and safety invariants are
in [`docs/PLATFORM.md`](../docs/PLATFORM.md)).

## Tests

```bash
# from infra/bench
pip install -r experiment_store/requirements.txt pytest moto
MLFLOW_ALLOW_FILE_STORE=true python -m pytest experiment_store/ -q
```

Tests run offline against a mocked S3 (moto) and a file-store MLflow, covering the full lifecycle
(log, resolve, locate, download, purge), the identity and SigV4 guards, and the janitor's safety
invariants.
