# Architecture

The overview lives in the top-level [README](../README.md); this document is the deeper software
reference. It is deliberately infrastructure-agnostic: xprof depends only on an object store, an
MLflow tracking server, and a directory where a run's artifacts are readable. Where those come from
— and how the software is deployed and operated (containers, an orchestrator, mount provisioning, a
garbage-collection schedule) — is a deployment concern, one option for which is the `distributed-ai`
repository.

## Components

```
producer (any tool)                          consumer (any MCP client)
  experiment_store.log(alias, chip, ...)        analysis MCP over streamable-http
        |                                              |  pass a run_id, get advice
        v                                              v
        object store  +  MLflow index  +  artifact directory (read in place, no download)
        ^                                              |
        |  read in place                               v
   analysis MCP (CPU process)   resolve run -> locate under MCP_MOUNT_BASE -> analyze -> advice
```

| Component | Path | What it provides |
|---|---|---|
| **experiment_store** | [`experiment_store/`](../experiment_store/) | The fixed library (not a service): the identity contract, the object-store layout, the MLflow boundary, and the orphan garbage collector. |
| **analysis MCP** | [`analysis_mcp/`](../analysis_mcp/) | A FastMCP server (an ordinary CPU process) that resolves a run to its artifact paths under `MCP_MOUNT_BASE` and runs pluggable analyzers in place, returning advice. Analyzers come in two types — `command` for nsys/ncu and CLI tools, `server` for tools that serve results. |
| **examples** | [`examples/`](../examples/) | Reference producers, not the contract: `gpu_nsys/`, a CPU-only Neuron compile recipe, and `benchmark_iteration/`. |

Run discovery and search are the MLflow MCP's job, and the tuning know-how is the xprof-knowledge
MCP's; xprof neither duplicates the former nor embeds the latter.

## Reading artifacts in place

The analysis MCP reads a run's files from `MCP_MOUNT_BASE`, a plain directory in which
`<alias>/<run_id>/` files are readable. It does not care how that directory is provided — a
read-only object-store mount (e.g. AWS S3 Files), an NFS export, or a local sync all work — so long
as reads are cheap and the files are not copied to the client. Two properties matter wherever it
runs: the artifacts should be **read-only** to the MCP (only the janitor deletes), and analysis
artifacts should be kept separate from operational inference weights (never serve inference off the
same path). The specific mount technology and its IAM are a deployment detail (see distributed-ai
for the S3 Files setup).

## Garbage collection

The `janitor` (`experiment_store.janitor`) removes orphaned `<alias>/<run_id>/` prefixes — those
with no live run behind them (a crashed, failed, or soft-deleted run) that object-store lifecycle
rules cannot. It is fail-closed — any non-authoritative MLflow response aborts the sweep without
deleting — and grace-period guarded, so an in-flight upload is never purged. It uses a dedicated
delete-capable credential, never the read-only reader credential, and runs as a dry-run by default
(`JANITOR_APPLY=1` to delete); a `store.hold()` retention marker exempts a run's artifacts. Where it
is scheduled (a cron job, a function, a manual run) is a deployment choice.
