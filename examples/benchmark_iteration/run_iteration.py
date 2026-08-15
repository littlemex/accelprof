"""Log one benchmark iteration to the platform — the reusable core of the iterate loop
(profile -> serve -> benchmark -> LOG -> search/compare/analyze). NOT the contract; a worked
example you copy and adapt. It takes a framework's raw benchmark result, normalizes it through that
framework's adapter (metrics.py), and records a run keyed by experiment alias + run_no + namespace
+ your free tags, with the raw result kept as an artifact.

CLI (inside the platform image, PYTHONPATH=infra/bench):
    python -m examples.benchmark_iteration.run_iteration \
      --alias qwen-serving --chip gpu --region us-east-1 --workload-id sharegpt-qps8 \
      --framework sglang --result /out/sglang_result.jsonl --run-no 10 \
      --eval-result /out/mmlu.json --tag sweep=qps --tag qps=8

``--result`` accepts SGLang's ``bench_serving --output-file`` (JSONL; the last record is used).
``--eval-result`` extracts the accuracy score from ``sglang.test.run_eval`` / lm-eval output (and
tags which task+metric it came from); or pass ``--accuracy`` directly. ``--tool`` selects the
result-schema adapter and defaults to ``--framework`` — split them only to measure one serving
framework with another's client (e.g. ``--framework vllm --tool sglang`` when SGLang's bench_serving
drove a vLLM server). SGLang is the default measurement tool (perf + eval); the vllm adapter remains
as an illustration of the pluggable abstraction.
"""
from __future__ import annotations

import argparse
import json

from experiment_store import ExperimentStore, env

from . import adapters  # noqa: F401  (registers vllm/sglang adapters)
from .metrics import get_adapter

_TAG_MAX = 450  # keep tag values well under MLflow's ~500-char limit


def load_benchmark_result(path: str) -> dict:
    """Read a framework benchmark result. Some tools (SGLang's ``bench_serving --output-file``)
    APPEND one JSON object per run, so the file is JSONL — take the LAST record (the most recent
    run). A plain single-object JSON file is loaded as-is. NOTE: appending means a run that died
    before writing leaves the PREVIOUS run's record as the last line — use a fresh ``--output-file``
    per run (the example does) so 'last record' unambiguously means 'this run'."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)                       # single JSON object (possibly multi-line)
    except json.JSONDecodeError:
        pass
    for ln in reversed(text.splitlines()):            # JSONL -> last parseable record
        if ln.strip():
            try:
                return json.loads(ln)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: last non-empty line is not valid JSON ({e})") from None
    raise ValueError(f"{path}: no JSON records (empty file)")


def accuracy_from_eval(path: str) -> tuple[float, str | None, str]:
    """Extract the accuracy score from an eval harness result, returning (score, task, metric) so
    the caller can tag WHICH task/metric it is (else a run's ``accuracy`` has no defined meaning and
    cross-run compare is meaningless). SGLang's ``run_eval`` writes ``{"score": ...}`` (single) /
    ``{"mean_score": ...}`` (repeated); lm-eval nests under ``results.<task>.<metric>``. A bool is
    NOT a number here. Multiple lm-eval tasks are rejected — run one task, or split the runs."""
    obj = load_benchmark_result(path)
    for k in ("score", "mean_score", "accuracy", "acc"):
        v = obj.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v), None, k
    results = obj.get("results")                      # lm-eval shape: {"results": {task: {metric: v}}}
    if isinstance(results, dict):
        tasks = [t for t, mv in results.items() if isinstance(mv, dict)]
        if len(tasks) > 1:
            raise ValueError(f"{path}: eval has multiple tasks {tasks}; run one task per iteration "
                             f"so the logged accuracy has a single defined meaning")
        if tasks:
            task = tasks[0]
            for mk in ("acc,none", "acc", "exact_match,none", "exact_match",
                       "acc_norm,none", "acc_norm"):
                v = results[task].get(mk)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v), task, mk
    raise ValueError(f"no accuracy score found in {path}; keys={sorted(obj)}")


def _tag_value(v) -> str | None:
    """A framework extra is worth tagging only if it is a SCALAR (a big list/dict — SGLang's
    per-request arrays in verbose mode — would blow the tag-length limit and must not become one).
    Returns the truncated string form, or None to skip."""
    if isinstance(v, (str, bool, int, float)):
        return str(v)[:_TAG_MAX]
    return None


def log_benchmark_iteration(store: ExperimentStore, *, alias: str, chip: str, region: str,
                            workload_id: str, framework: str, result: dict, run_no: int | str,
                            tool: str | None = None, tags: dict | None = None,
                            params: dict | None = None, accuracy: float | None = None,
                            artifacts: list[str] | None = None) -> str:
    """Normalize `result` via the `tool` adapter (defaults to `framework`) and log a run. Returns the
    MLflow run_id. Grouping keys: run_no (this iteration) + framework are tagged automatically;
    namespace is auto-injected by the store; your `tags` are merged on top. Cross-framework metrics
    land under canonical names, so compare/search work regardless of framework."""
    metrics = get_adapter(tool or framework).normalize(result)
    m = metrics.as_metrics()
    if accuracy is not None:
        m["accuracy"] = float(accuracy)
    run_tags = {"run_no": str(run_no), "framework": framework, **(tags or {})}
    # surface framework extras that as_metrics() did NOT already log as a metric (non-numeric, or
    # non-finite numeric like SGLang's request_rate=inf) as searchable tags — scalars only.
    for k, v in metrics.extra.items():
        if f"extra_{k}" in m:
            continue                                  # already a (finite-numeric) metric
        tv = _tag_value(v)
        if tv is not None:
            run_tags.setdefault(f"extra_{k}", tv)
    return store.log(alias, chip=chip, region=region, workload_id=workload_id,
                     metrics=m, params=params, tags=run_tags, artifacts=artifacts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--workload-id", required=True)
    ap.add_argument("--framework", required=True, help="serving framework being measured (tag)")
    ap.add_argument("--tool", default=None,
                    help="result-schema adapter; defaults to --framework. Set only when the "
                         "measurement client differs from the serving framework")
    ap.add_argument("--result", required=True, help="framework benchmark JSON/JSONL to normalize + keep")
    ap.add_argument("--run-no", required=True)
    ap.add_argument("--accuracy", type=float, default=None, help="accuracy value (if you have it directly)")
    ap.add_argument("--eval-result", default=None,
                    help="eval harness JSON (sglang run_eval / lm-eval) to extract accuracy from + keep")
    ap.add_argument("--tag", action="append", default=[], metavar="K=V", help="free tag; repeatable")
    args = ap.parse_args()

    if args.accuracy is not None and args.eval_result:
        ap.error("pass either --accuracy or --eval-result, not both (they'd conflict)")
    for t in args.tag:
        if "=" not in t:
            ap.error(f"--tag must be K=V, got {t!r}")

    e = env.os_environ()
    store = ExperimentStore.build(
        region=env.region(e),
        trace_bucket=env.require(e, env.TRACE_BUCKET_ENV, "the trace bucket"),
        tracking_uri=env.require(e, env.TRACKING_ENV, "the MLflow tracking URI / app ARN"),
        namespace=env.namespace(e))
    result = load_benchmark_result(args.result)
    tags = dict(t.split("=", 1) for t in args.tag)
    accuracy = args.accuracy
    artifacts = [args.result]
    if args.eval_result:
        accuracy, eval_task, eval_metric = accuracy_from_eval(args.eval_result)
        if eval_task:
            tags.setdefault("eval_task", eval_task)
        tags.setdefault("eval_metric", eval_metric)   # so a logged accuracy's meaning is explicit
        artifacts.append(args.eval_result)
    run_id = log_benchmark_iteration(
        store, alias=args.alias, chip=args.chip, region=args.region, workload_id=args.workload_id,
        framework=args.framework, tool=args.tool, result=result, run_no=args.run_no, tags=tags,
        accuracy=accuracy, artifacts=artifacts)
    print(run_id)


if __name__ == "__main__":
    main()
