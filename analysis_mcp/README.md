# analysis_mcp — profiling-analysis MCP

Analysis is decoupled from production. An accelerator (GPU/Neuron) workload only *produces*
profiler artifacts and writes them to the trace bucket; reading and analyzing them needs no
accelerator, so this MCP runs on a **CPU Pod** that mounts the bucket read-only via AWS S3 Files.
Given an experiment alias it resolves the runs through `experiment_store`, reads each run's
artifacts in place off the mount, runs a per-run analyzer, and returns advice — a text finding,
never the artifact bytes. FastMCP serves streamable-http natively, so a laptop reaches it directly
with `kubectl port-forward` (register `http://127.0.0.1:<port>/mcp`); no gateway is involved.

The chart pins the Pod to a CPU nodepool (`node-role: cpu`, no GPU request). Reads are in place, so
no local disk is needed for the artifacts themselves; a tool that exports scratch (for example
`nsys stats --sqlite`) writes to a `/tmp` emptyDir, so size the CPU node's ephemeral storage and the
pod's resources for the largest trace you analyze.

## Tools

The surface is intentionally narrow — three tools that do only what nothing else can. Run discovery
and search are the MLflow MCP's job (see below); this MCP is not a second run browser.

- `resolve_artifacts(run_id | alias+chip, pattern)` — the **id/alias → file-path** contract: map an
  MLflow identity to the concrete Pod-local path(s) of matching profile files on the S3 Files mount
  (glob, e.g. `*.nsys-rep`, `*.neff`). Returns metadata only (dir + absolute paths). Hand these to
  an analyzer, or to an external tool/MCP that reads the **same** mount at the same `mountBase`.
- `stage_run(run_id)` — ensure the run's artifacts are readable on the Pod (S3 Files mount, no
  copy) and return their local dir + file inventory. Traverses the dir to trigger S3 Files
  metadata import (first-access import can otherwise miss a freshly-synced object). Raises if the
  dir is not present on the Pod (a down/misconfigured mount) rather than returning an empty
  inventory.
- `analyze(run_id, analyzer="inventory")` — run an analyzer over the staged dir and return advice.

Find a run first with the official **MLflow MCP** (`pip install "mlflow[mcp]"`, `mlflow mcp run`),
which browses and searches experiments/runs; hand its `run_id` to the tools above. That MCP cannot
map a run to a mount path — which is exactly what `resolve_artifacts` adds. Run it behind
`charts/remote-mcp`, alongside this one.

## Analyzers (pluggable, tool-agnostic — one contract, two execution strategies)

Fixed IDs / open content applies here too: analyzers are pluggable, not a baked-in metric. Every
analyzer is the same contract — `(run_dir, timeout) -> advice` — so `analyze(run_id, name)` is
identical for all of them; what differs is only HOW the tool is driven. They share the token DSL
(`{dir}`, `{tmp}` = writable scratch since the mount is read-only, `{file:GLOB}`, `{files:GLOB}`)
and result shaping (noise-drop, empty-annotation, length bound). Two types, so each tool's shape is
honored without forcing one into the other:

- **command** (nsys/ncu, neuron-explorer, any CLI): expand argv, run once, capture stdout. A bare
  argv list in `MCP_ANALYZERS` is sugar for this. Builtins: `inventory` (dependency-free), `nsys-stats`/
  `nsys-analyze` (pre-wired `{tmp}` + noise-drop; need `nsys`), and **`neuron-summary`**
  (`neuron-explorer view -n {file:*.neff} -s {file:*.ntff} --output-format summary-text`; needs
  `neuron-explorer`). Most profiler CLIs — including neuron-explorer's `summary-text`/`summary-json`
  report — are run-to-completion, so this covers them.
- **server** (only for a tool that SERVES results instead of printing them — e.g. neuron-explorer's
  `--output-format db` InfluxDB/UI mode): start it as a background server, wait for its `ready_port`,
  run a query whose stdout is the advice, then tear it down.
  `{"type":"server","start":[...],"ready_port":3002,"query":[...]}`.

Tool binaries stay out of the base image (build FROM it and add the tool, e.g.
`Dockerfile.analysis-mcp-nsys`), so the platform image stays small and tool-agnostic.

Analyzers invoke each tool's own CLI directly: there is no upstream MCP that analyzes a profiler
file, so the `remote-mcp` bridge is only for stdio MCPs such as MLflow's, not for profiler analysis.

## Config (env)

`MCP_MLFLOW_TRACKING_URI` (MLflow App ARN), `MCP_AWS_REGION`, `MCP_TRACE_BUCKET` (the trace bucket
mounted for this pod's region), `MCP_ANALYZER_TIMEOUT_SECONDS`, and `MCP_ANALYZERS` (a JSON map of
analyzer name to argv/spec, registered at boot on top of the builtins). `MCP_MOUNT_BASE` (mount
path, e.g. `/traces`) is **optional**: unset it for a metadata-only deployment (`list_runs`/
`compare`) that needs no S3 Files stack; `stage_run`/`analyze` then raise a clear error. The env
names and parsing are shared with the janitor via `experiment_store/env.py`.

## Tests

`MLFLOW_ALLOW_FILE_STORE=true python -m pytest analysis_mcp/ -q` — a temporary directory stands in
for the mount, so the service, both analyzer types, and the token DSL are covered offline.
