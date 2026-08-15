# Architecture

The overview lives in the top-level [README](../README.md); this document is the deeper software
reference. Deployment and operations — the Helm charts, the Terraform for the trace buckets, MLflow,
IAM, and S3 Files, the MCP hosting, and the garbage-collection schedule — live in the
**distributed-ai** repository, which runs this repo's image.

## Components

```
producer (any tool)                             laptop
  experiment_store.log(alias, chip, metrics, artifacts)   analysis MCP over kubectl port-forward
        |                                                     |  pass a run_id, get advice
        v                                                     v
        trace bucket  +  MLflow index  +  S3 Files mount (read-only, in-place reads)
        ^                                                     |
        |  read in place (no download)                        v
   analysis-mcp (CPU Pod)   resolve run -> locate on mount -> analyze -> advice
```

| Component | Path | What it provides |
|---|---|---|
| **experiment_store** | [`experiment_store/`](../experiment_store/) | The fixed platform library (not a service): the identity contract, the S3 layout, the MLflow boundary, and the orphan garbage collector. |
| **analysis-mcp** | [`analysis_mcp/`](../analysis_mcp/) | A FastMCP service on a CPU Pod (no accelerator) that resolves a run to its artifact paths on the mount and runs pluggable analyzers in place, returning advice. Analyzers come in two types — `command` for nsys/ncu and CLI tools, `server` for tools that serve results. |
| **examples** | [`examples/`](../examples/) | Reference producers, not the contract: `gpu_nsys/`, a CPU-only Neuron compile recipe, and `benchmark_iteration/`. |

Run discovery and search are the MLflow MCP's job, and the tuning know-how is the xprof-knowledge
MCP's; this repo neither duplicates the former nor embeds the latter. The trace buckets, MLflow, and
the S3 Files mount are provisioned by distributed-ai.

## S3 Files: reading artifacts in place

Consumers mount the trace bucket with AWS S3 Files — an EFS-backed, POSIX file system — via the EFS
CSI driver and read `.nsys-rep`, `.ncu-rep`, `.neff`, and `.ntff` files directly, with no download.
The mount has three hard requirements (configured in distributed-ai):

- the PV `volumeHandle` must include an access point, in the form
  `s3files:<FileSystemId>::<AccessPointId>`; a bare file-system id is handled as plain EFS and does
  not mount;
- the volume is `ReadWriteMany`, with read-only enforced at the pod and by the mount IAM role
  (mount-client access only, no write);
- the CSI node ServiceAccount holds `s3files:ClientMount` through Pod Identity, not the node role.

Keep analysis artifacts (mounted, read-only) separate from operational inference weights (fast local
storage); never serve inference off the mount.

## Garbage collection

The `janitor` (`experiment_store.janitor`) removes orphaned `<alias>/<run_id>/` prefixes — those
with no live run behind them (a crashed, failed, or soft-deleted run) that S3 lifecycle rules
cannot. It is fail-closed — any non-authoritative MLflow response aborts the sweep without deleting
— and grace-period guarded, so an in-flight upload is never purged. It uses a dedicated
delete-capable role, never the read-only reader role, and runs as a dry-run by default
(`JANITOR_APPLY=1` to delete); a `store.hold()` retention marker exempts a run's artifacts. Its
compute placement (CronJob or Lambda) and delete-capable IAM are configured in distributed-ai.
