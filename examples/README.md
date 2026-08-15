# examples/ — reference producers (not the platform contract)

A producer is not a standalone job you run instead of your real work — it is the collection you
attach to your real workload. You profile or benchmark the serving or training process you already
run, and the same run reports its results to the platform through one call, `experiment_store.log`.
Everything here is one illustrative way to do that, not part of the fixed contract: the platform
fixes only identity and the S3/MLflow layout, while what you measure, which tool you profile with,
and which metrics you log are open and differ per workload and per team. These live under
`examples/` (not in `experiment_store/`) to make it obvious they can be copied, replaced, or
ignored.

- `gpu_nsys/` — wrap a GPU process with Nsight Systems and register the capture (its metric set is
  illustrative, not required).
- `neuron_cpu_compile.py` — compile a model to a Neuron `.neff` on CPU, with no Trainium device.
- `benchmark_iteration/` — a template for the iterate loop (serve, benchmark, log, search,
  compare) with a framework-agnostic metrics abstraction and clear plug-in points.

To add a different tool, chip, or metric set, add a sibling directory here; do not modify
`experiment_store/`.
