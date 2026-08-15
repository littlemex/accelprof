# analysis_mcp — profiling-analysis MCP

Analysis is decoupled from production. An accelerator (GPU/Neuron) workload only *produces* profiler
artifacts and stores them; reading and analyzing them needs no accelerator, so this MCP runs as an
ordinary CPU process. Given a run id it resolves the run through `experiment_store`, reads the run's
artifacts in place from `MCP_MOUNT_BASE` (a directory where `<alias>/<run_id>/` files are readable),
runs a per-run analyzer, and returns advice — a text finding, never the artifact bytes. FastMCP
serves streamable-http natively, so any MCP client connects directly at `http://<host>:<port>/mcp`;
no gateway is involved.

Reads are in place, so no local disk is needed for the artifacts themselves; a tool that exports
scratch (for example `nsys stats --sqlite`) writes under `TMPDIR`, so give the process room for the
largest trace you analyze. How `MCP_MOUNT_BASE` is populated (an S3 Files or NFS mount, a local
sync) is a deployment choice, not something this server assumes.

## Tools

The surface is intentionally narrow — three tools that do only what nothing else can. Run discovery
and search are the MLflow MCP's job (see below); this MCP is not a second run browser.

- `resolve_artifacts(run_id | alias+chip, pattern)` — the **id/alias → file-path** contract: map an
  MLflow identity to the concrete local path(s) of matching profile files under `MCP_MOUNT_BASE`
  (glob, e.g. `*.nsys-rep`, `*.neff`). Returns metadata only (dir + absolute paths). Hand these to
  an analyzer, or to an external tool/MCP that reads the **same** directory.
- `stage_run(run_id)` — ensure the run's artifacts are readable (no copy) and return their local dir
  + file inventory. Traverses the dir to trigger any lazy backing-store import (a first-access import
  can otherwise miss a freshly-synced object). Raises if the dir is absent (a down/misconfigured
  backing store) rather than returning an empty inventory.
- `analyze(run_id, analyzer="inventory")` — run an analyzer over the staged dir and return advice.

Find a run first with the official **MLflow MCP** (`pip install "mlflow[mcp]"`, `mlflow mcp run`),
which browses and searches experiments/runs; hand its `run_id` to the tools above. That MCP cannot
map a run to a file path — which is exactly what `resolve_artifacts` adds. Run both MCPs side by side
in your client.

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
