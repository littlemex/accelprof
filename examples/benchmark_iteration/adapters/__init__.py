"""Importing this package registers the bundled adapters. Add your own framework by dropping a
module here that builds a BenchmarkMetrics and calls metrics.register(YourAdapter()), then import
it (add it to the line below or import your module before calling get_adapter)."""
from . import sglang, vllm  # noqa: F401  (import side effect: register the adapters)
