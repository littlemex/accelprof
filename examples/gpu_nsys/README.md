# examples/gpu_nsys — register a Nsight Systems trace (one example producer)

Illustrative only (see `../README.md`). Demonstrates the producer half of the platform: capture a
GPU run with Nsight Systems, then log it under an alias with `experiment_store`.

```bash
# 1. capture (workload-specific; produces a .nsys-rep)
nsys profile -o /tmp/run.nsys-rep python my_workload.py

# 2. register the run (identity fixed; metrics illustrative)
export TRACE_BUCKET=$(terraform -chdir=../../../data-layer output -json trace_buckets | jq -r '."ap-northeast-1"')
export MLFLOW_APP_ARN=$(terraform -chdir=../../../data-layer output -raw mlflow_app_arn)
python -m examples.gpu_nsys.produce \
  --alias llama3-8b-parity --workload prefill-bs1 --region ap-northeast-1 \
  --trace /tmp/run.nsys-rep --trace-bucket "$TRACE_BUCKET" --tracking-uri "$MLFLOW_APP_ARN" \
  --cosine 0.9997 --latency-p50-ms 110
```

The trace lands at `s3://$TRACE_BUCKET/llama3-8b-parity/<run_id>/run.nsys-rep`; a consumer
(the GPU analysis MCP) later resolves the alias, reads that `.nsys-rep` in place off the mounted
bucket, and runs `nsys stats`/`nsys analyze` on the Pod. The `--cosine`/`--latency-p50-ms` flags
are examples — log whatever metrics your GPU-vs-Neuron comparison actually uses.
