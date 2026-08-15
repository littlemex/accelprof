# Using the official MLflow MCP alongside accelprof

The MLflow project ships an MCP server (`pip install 'mlflow[mcp]>=3.5.1'`, `mlflow mcp run`,
**stdio**) that wraps MLflow's CLI as MCP tools — a generic "browse and query my experiments/runs"
surface. accelprof **reuses it** for run discovery rather than reinventing that surface; the analysis MCP
adds only what MLflow MCP structurally cannot do: mapping a run to its profile files and analyzing
them.

## Running it against your tracking server

- `MLFLOW_TRACKING_URI` = your tracking server (the same value producers use). For a plain MLflow
  server just point at its URI; for a SageMaker-managed MLflow, install `sagemaker-mlflow` in the
  same environment so requests are SigV4-signed (it dispatches on the ARN scheme, no code change).
- `MLFLOW_MCP_TOOLS=ml` (or `all`) so the **experiments + runs** tools are exposed. The default
  `genai` exposes traces/scorers, which accelprof does not use — set this or you get no run tools.
- MLflow MCP speaks **stdio**. To reach it from an HTTP-only client, put any stdio→streamable-http
  bridge (e.g. supergateway) in front of it; that bridging is a deployment detail, not part of accelprof.

## What it gives you (and the one thing it doesn't)

- `search_experiments` / `get_experiment`, `list_runs` / `describe_run` → a run's metrics, params,
  tags, status, and `artifact_uri`, plus run search by MLflow filter. So run discovery — browse
  experiments, read an iteration's metrics, filter by tags (`run_no`, `framework`, `namespace`, …) —
  is MLflow MCP's job.
- The gap: `describe_run.artifact_uri` is the MLflow artifact-store URI, not accelprof's `artifacts_uri`
  tag that points at the large profiler prefix, and it has no notion of the local artifact
  directory. Mapping id/alias → the on-disk profile file and running the profiler over it is exactly
  the accelprof analysis MCP (`resolve_artifacts` / `analyze`).

## Recommended split

| Need | Use |
|---|---|
| Browse experiments, read a run's metrics/params/tags, run search | **MLflow MCP** |
| Resolve an id/alias to the profile file on disk, run nsys/neuron analysis | accelprof `resolve_artifacts` / `analyze` |

Register both MCPs in your client: MLflow MCP to find and inspect the run, accelprof to profile its
artifacts.
