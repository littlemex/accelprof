# Using the official MLflow MCP alongside this platform

The MLflow project ships an MCP server (`pip install 'mlflow[mcp]>=3.5.1'`, `mlflow mcp run`,
**stdio**). It wraps MLflow's own CLI as MCP tools, so it gives a laptop a generic "browse and query
my experiments/runs" surface for free. We **reuse it** rather than reinvent that surface; our
`analysis-mcp` adds only what MLflow MCP structurally cannot do (the S3 Files profile-file view).

## How to run it against our tracking server

- `MLFLOW_TRACKING_URI` = the SageMaker-managed MLflow **App ARN** (same value producers use).
  Because MLflow MCP uses the real MLflow Python client, install `sagemaker-mlflow` in the same env
  so requests are SigV4-signed — no code change, it dispatches on the ARN scheme.
- `MLFLOW_MCP_TOOLS=ml` (or `all`) so the **experiments + runs** tools are exposed. The default
  `genai` exposes traces/scorers, which we don't use — set this or you won't get run tools.
- It speaks **stdio**, so expose it to a laptop with **our `remote-mcp` bridge** (supergateway):
  `stdioCommand: ["mlflow","mcp","run"]` on `Dockerfile.remote-mcp-bridge` (which already installs
  `mlflow[mcp]` + `sagemaker-mlflow`). This is the bridge's first real customer.

## What it gives us (and the one thing it doesn't)

- `search_experiments` / `get_experiment`, `list_runs` / `describe_run` (a.k.a. get_run) → run
  metrics, params, tags, status, and the run's `artifact_uri`; run search by MLflow filter.
- So for **experiment/run management** — browse campaigns, read an iteration's metrics, filter by
  tags (our `run_no`/`framework`/`namespace`/sweep tags) — MLflow MCP works directly.
- The one gap: `describe_run.artifact_uri` is the MLflow **artifact store** URI (the small
  mlflow-artifacts bucket, 200 MB cap), **not** our per-run `artifacts_uri` tag that points at the
  GB trace prefix, and it has no notion of the **S3 Files mount path**. Mapping id/alias → the
  mounted profile file, and running the profiler over it, is exactly `analysis-mcp`
  (`resolve_artifacts` / `analyze`). That mapping is this platform's reason to exist; no MLflow MCP
  can do it.

## Recommended split (register both on the laptop)

| Need | Use |
|---|---|
| Browse experiments, read a run's metrics/params/tags, generic run search | **MLflow MCP** (`mlflow mcp run`, via remote-mcp bridge) |
| Chip-aware run list / GPU-vs-Neuron `compare` / tag search returning our fields | `analysis-mcp.list_runs` / `compare` / `search_runs` |
| Resolve an id/alias to the profile file on the S3 Files mount, run nsys/neuron analysis | `analysis-mcp.resolve_artifacts` / `analyze` |

Add MLflow MCP with `claude mcp add` pointing at the bridged endpoint, and `analysis-mcp` over
`kubectl port-forward`. They compose: MLflow MCP to find/inspect the run, `analysis-mcp` to profile
its artifacts.
