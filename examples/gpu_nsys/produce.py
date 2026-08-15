"""Example GPU producer: register an already-captured Nsight Systems trace with experiment_store.

THIS IS ONE EXAMPLE, NOT THE CONTRACT (see ../README.md). It shows the producer pattern:
capture is workload-specific (you run your workload under `nsys profile ... -o run.nsys-rep`);
this script takes the resulting artifacts and logs a run under an alias, with an *illustrative*
metric set. Swap the metrics, the tool, or the whole script freely — experiment_store does not
care what you put in, only that the identity keys are set.

Usage (capture first, then register):
    nsys profile -o /tmp/run.nsys-rep python my_workload.py
    python -m examples.gpu_nsys.produce \
        --alias llama3-8b-parity --workload prefill-bs1 --region ap-northeast-1 \
        --trace /tmp/run.nsys-rep --trace-bucket "$TRACE_BUCKET" \
        --tracking-uri "$MLFLOW_APP_ARN" --cosine 0.9997 --latency-p50-ms 110
"""
from __future__ import annotations

import argparse

from experiment_store import ExperimentStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a captured nsys trace (example producer).")
    ap.add_argument("--alias", required=True)
    ap.add_argument("--workload", required=True, dest="workload_id")
    ap.add_argument("--region", required=True)
    ap.add_argument("--trace", required=True, help="path to the captured .nsys-rep (file or dir)")
    ap.add_argument("--trace-bucket", required=True)
    ap.add_argument("--tracking-uri", required=True, help="MLflow App ARN or tracking URI")
    # Illustrative metrics only — replace with whatever your comparison needs.
    ap.add_argument("--cosine", type=float, default=None)
    ap.add_argument("--latency-p50-ms", type=float, default=None, dest="latency_p50_ms")
    args = ap.parse_args()

    store = ExperimentStore.build(region=args.region, trace_bucket=args.trace_bucket,
                                  tracking_uri=args.tracking_uri)

    metrics = {k: v for k, v in {"cosine": args.cosine,
                                 "latency_p50_ms": args.latency_p50_ms}.items() if v is not None}
    run_id = store.log(args.alias, chip="gpu", region=args.region, workload_id=args.workload_id,
                       metrics=metrics, params={"tool": "nsight-systems"}, artifacts=[args.trace])
    print(run_id)


if __name__ == "__main__":
    main()
