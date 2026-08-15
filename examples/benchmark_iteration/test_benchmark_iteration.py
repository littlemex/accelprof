"""Tests for the benchmark-iteration example: adapters normalize each framework's real-schema
output to the SAME canonical metrics, and log_benchmark_iteration records a searchable run."""
from __future__ import annotations

import json
import os

import boto3
import pytest

from examples.benchmark_iteration import adapters  # noqa: F401 (register)
from examples.benchmark_iteration.metrics import get_adapter, registered

mlflow = pytest.importorskip("mlflow")
moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

from examples.benchmark_iteration.run_iteration import (  # noqa: E402
    accuracy_from_eval, load_benchmark_result, log_benchmark_iteration)
from experiment_store import ExperimentStore  # noqa: E402
from experiment_store.mlflow_io import MlflowIO  # noqa: E402

_HERE = os.path.dirname(__file__)


def _sample(name):
    with open(os.path.join(_HERE, "samples", name), encoding="utf-8") as f:
        return json.load(f)


def test_adapters_registered():
    assert {"vllm", "sglang"} <= set(registered())


def test_vllm_and_sglang_normalize_to_same_canonical_keys():
    v = get_adapter("vllm").normalize(_sample("vllm_result.json"))
    s = get_adapter("sglang").normalize(_sample("sglang_result.json"))
    # canonical latency mapping absorbs the naming difference (vLLM e2el vs SGLang e2e_latency)
    assert v.e2e_ms.p99 == 3400.9 and s.e2e_ms.p99 == 3200.0
    assert v.ttft_ms.p50 == 72.5 and s.ttft_ms.p50 == 68.0
    vm, sm = v.as_metrics(), s.as_metrics()
    # both frameworks expose the SAME comparable keys
    for k in ("ttft_ms_p99", "tpot_ms_mean", "output_throughput_tok_s", "e2e_ms_p50"):
        assert k in vm and k in sm
    # framework-specific fields are preserved, not dropped
    assert v.extra.get("total_input_tokens") == 40000       # vLLM extra
    assert s.extra.get("input_throughput") == 975.6         # SGLang-only extra (no canonical slot)
    # combined throughput maps to the SAME canonical field from each tool's own key
    assert v.total_token_throughput_tok_s == 1558.2         # vLLM: total_token_throughput
    assert s.total_token_throughput_tok_s == 1600.0         # SGLang: total_throughput


def test_non_finite_extra_is_dropped_not_fatal():
    """SGLang emits request_rate=inf when unthrottled; a passthrough extra must never make a run
    unloggable — non-finite numeric extras are dropped from the flat metrics."""
    m = get_adapter("sglang").normalize({**_sample("sglang_result.json"), "request_rate": float("inf")})
    flat = m.as_metrics()
    assert "extra_request_rate" not in flat            # inf dropped
    assert all(v == v and abs(v) != float("inf") for v in flat.values())  # all finite
    # a finite request_rate is kept
    m2 = get_adapter("sglang").normalize({**_sample("sglang_result.json"), "request_rate": 8.0})
    assert m2.as_metrics()["extra_request_rate"] == 8.0


def test_load_benchmark_result_reads_last_jsonl_record(tmp_path):
    """SGLang bench_serving APPENDS one JSON object per run; the loader must take the LAST record
    (most recent run), and still accept a plain single-object JSON file."""
    jsonl = tmp_path / "sglang_result.jsonl"
    jsonl.write_text('{"request_throughput": 1.0, "completed": 10}\n'
                     '{"request_throughput": 2.0, "completed": 20}\n', encoding="utf-8")
    assert load_benchmark_result(str(jsonl))["request_throughput"] == 2.0   # last record
    single = tmp_path / "one.json"
    single.write_text('{"request_throughput": 9.0}', encoding="utf-8")
    assert load_benchmark_result(str(single))["request_throughput"] == 9.0


def test_accuracy_from_eval_sglang_and_lm_eval(tmp_path):
    sg = tmp_path / "mmlu.json"; sg.write_text('{"score": 0.83, "latency": 12.0}', encoding="utf-8")
    assert accuracy_from_eval(str(sg)) == (0.83, None, "score")
    rep = tmp_path / "rep.json"; rep.write_text('{"mean_score": 0.77, "scores": ["a"]}', encoding="utf-8")
    assert accuracy_from_eval(str(rep)) == (0.77, None, "mean_score")
    lme = tmp_path / "lm.json"
    lme.write_text('{"results": {"gsm8k": {"exact_match,none": 0.61, "alias": "gsm8k"}}}', encoding="utf-8")
    assert accuracy_from_eval(str(lme)) == (0.61, "gsm8k", "exact_match,none")  # task+metric reported
    bad = tmp_path / "bad.json"; bad.write_text('{"nope": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        accuracy_from_eval(str(bad))


def test_accuracy_from_eval_rejects_multi_task_and_bool(tmp_path):
    """A run's 'accuracy' must have one defined meaning — multiple lm-eval tasks are rejected, and a
    JSON bool is not a valid score (isinstance(True,int) would otherwise leak 1.0)."""
    multi = tmp_path / "multi.json"
    multi.write_text('{"results": {"gsm8k": {"acc,none": 0.6}, "mmlu": {"acc,none": 0.4}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        accuracy_from_eval(str(multi))
    b = tmp_path / "b.json"; b.write_text('{"score": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        accuracy_from_eval(str(b))


def test_load_benchmark_result_errors_cleanly(tmp_path):
    empty = tmp_path / "e.jsonl"; empty.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_result(str(empty))
    partial = tmp_path / "p.jsonl"; partial.write_text('{"a":1}\n{"b": 2, "trunc', encoding="utf-8")
    with pytest.raises(ValueError):                       # last line is a truncated write, not silent
        load_benchmark_result(str(partial))


def test_non_scalar_extra_not_tagged(tmp_path):
    """SGLang verbose mode dumps per-request arrays; a list/dict extra must NOT become a giant tag
    that trips the store's tag-length limit — only scalars are tagged, and a non-finite numeric
    (request_rate=inf) is preserved as a tag rather than lost."""
    import boto3
    from moto import mock_aws as _mock
    with _mock():
        s3 = boto3.client("s3", region_name="us-east-1"); s3.create_bucket(Bucket="mcp-traces-test")
        io = MlflowIO(tracking_uri=f"file://{tmp_path}/mlruns")
        store = ExperimentStore(trace_bucket="mcp-traces-test", s3_client=s3, mlflow=io)
        raw = {**_sample("sglang_result.json"), "request_rate": float("inf"),
               "itls": [1.0] * 5000, "generated_texts": ["x" * 100] * 50}
        rid = log_benchmark_iteration(store, alias="a", chip="gpu", region="us-east-1",
                                      workload_id="w", framework="sglang", result=raw, run_no=1)
        r = store.resolve(rid, by="id")[0]
        assert r.tags.get("extra_request_rate") == "inf"          # non-finite preserved as tag
        assert "extra_itls" not in r.tags and "extra_generated_texts" not in r.tags  # arrays skipped


@mock_aws
def test_tool_selects_adapter_framework_is_the_tag(tmp_path):
    """Measuring a vLLM server with SGLang's client: adapter (schema) = sglang, but the searchable
    framework tag = vllm. The two must be separable or cross-framework compare is corrupted."""
    s3 = boto3.client("s3", region_name="us-east-1"); s3.create_bucket(Bucket="mcp-traces-test")
    io = MlflowIO(tracking_uri=f"file://{tmp_path}/mlruns")
    store = ExperimentStore(trace_bucket="mcp-traces-test", s3_client=s3, mlflow=io)
    rid = log_benchmark_iteration(store, alias="x", chip="gpu", region="us-east-1", workload_id="w",
                                  framework="vllm", tool="sglang", result=_sample("sglang_result.json"),
                                  run_no=1)
    r = store.resolve(rid, by="id")[0]
    assert r.tags["framework"] == "vllm"                       # tag = serving framework
    assert r.metrics["total_token_throughput_tok_s"] == 1600.0  # parsed by the sglang adapter


@mock_aws
def test_log_iteration_records_searchable_run(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1"); s3.create_bucket(Bucket="mcp-traces-test")
    io = MlflowIO(tracking_uri=f"file://{tmp_path}/mlruns")
    store = ExperimentStore(trace_bucket="mcp-traces-test", s3_client=s3, mlflow=io, namespace="ddp")
    rid = log_benchmark_iteration(
        store, alias="llama3-serving", chip="gpu", region="us-east-1", workload_id="sharegpt",
        framework="vllm", result=_sample("vllm_result.json"), run_no=10,
        tags={"sweep": "qps", "qps": "8"}, accuracy=0.812,
        artifacts=[os.path.join(_HERE, "samples", "vllm_result.json")])
    r = store.resolve(rid, by="id")[0]
    assert r.tags["run_no"] == "10" and r.tags["framework"] == "vllm" and r.tags["namespace"] == "ddp"
    assert r.metrics["ttft_ms_p99"] == 210.4 and r.metrics["accuracy"] == 0.812
    assert r.artifacts_uri and r.artifacts_uri.endswith(f"/{rid}/")
    # the iteration is retrievable by its number and by the sweep tag
    assert len(store.search(alias="llama3-serving", filter_string="tags.run_no = '10'")) == 1
    assert len(store.search(filter_string="tags.sweep = 'qps'")) == 1
