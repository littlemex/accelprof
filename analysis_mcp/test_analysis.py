"""Unit tests for the analysis MCP service + analyzers (fakes; a temp dir stands in for the
S3 Files mount)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest

from . import analyzers as A
from . import config as C
from .service import AnalysisService


# --- config ---------------------------------------------------------------------------------

def test_load_config_happy():
    cfg = C.load_config({
        "MCP_MLFLOW_TRACKING_URI": "arn:...", "MCP_AWS_REGION": "ap-northeast-1",
        "MCP_TRACE_BUCKET": "mcp-traces-x", "MCP_MOUNT_BASE": "/traces",
    })
    assert cfg.trace_bucket == "mcp-traces-x" and cfg.mount_base == "/traces"


def test_load_config_mount_base_optional():
    """A3: the mount is opt-in — a metadata-only (list_runs/compare) deployment boots without it."""
    cfg = C.load_config({
        "MCP_MLFLOW_TRACKING_URI": "arn:...", "MCP_AWS_REGION": "r", "MCP_TRACE_BUCKET": "b",
    })
    assert cfg.mount_base is None


@pytest.mark.parametrize("missing", ["MCP_MLFLOW_TRACKING_URI", "MCP_AWS_REGION", "MCP_TRACE_BUCKET"])
def test_load_config_requires_each(missing):
    env = {"MCP_MLFLOW_TRACKING_URI": "arn:...", "MCP_AWS_REGION": "r", "MCP_TRACE_BUCKET": "b"}
    del env[missing]
    with pytest.raises(ValueError):
        C.load_config(env)


def test_load_config_parses_analyzers():
    cfg = C.load_config({
        "MCP_MLFLOW_TRACKING_URI": "arn:x", "MCP_AWS_REGION": "r", "MCP_TRACE_BUCKET": "b",
        "MCP_ANALYZERS": '{"nsys-stats": ["nsys","stats","{file:*.nsys-rep}"],'
                         ' "neuron": {"type":"server","start":["srv","-n","{file:*.neff}"],'
                         ' "ready_port":3002,"query":["curl","http://127.0.0.1:3002/x"]}}',
    })
    assert isinstance(cfg.analyzers["nsys-stats"], A.CommandAnalyzer)
    assert cfg.analyzers["nsys-stats"].argv_template == ("nsys", "stats", "{file:*.nsys-rep}")
    assert isinstance(cfg.analyzers["neuron"], A.ServerQueryAnalyzer)
    assert cfg.analyzers["neuron"].ready_port == 3002


def test_build_analyzer_rejects_bad_server_spec():
    for bad in ({"type": "server", "start": ["s"], "query": ["q"]},          # no ready_port
                {"type": "server", "start": [], "query": ["q"], "ready_port": 1},  # empty start
                {"type": "nope", "argv": ["x"]}):                            # unknown type
        with pytest.raises(ValueError):
            A.build_analyzer("x", bad)


def test_server_query_analyzer_end_to_end(tmp_path):
    d = _dir_with(tmp_path, "f")
    import shutil as _sh
    if not _sh.which("curl"):
        import pytest as _p; _p.skip("curl not available")
    port = 18931
    sq = A.ServerQueryAnalyzer(
        name="probe",
        start_template=("python3", "-c",
                        f"import http.server,socketserver; socketserver.TCPServer(('127.0.0.1',{port}),"
                        f"http.server.SimpleHTTPRequestHandler).serve_forever()"),
        ready_port=port,
        query_template=("curl", "-sS", f"http://127.0.0.1:{port}/"),
        ready_timeout_s=15)
    out = sq(d, 20)
    assert "<" in out  # got an HTTP directory-listing body back from the started server


def test_load_config_bad_analyzers_raises():
    base = {"MCP_MLFLOW_TRACKING_URI": "arn:x", "MCP_AWS_REGION": "r", "MCP_TRACE_BUCKET": "b"}
    for bad in ("{not json", '["a"]', '{"x": "notalist"}', '{"x": []}', '{"x": [1,2]}',
                '{"x": ["nsys","{flie:*.nsys-rep}"]}',       # keyword typo
                '{"x": ["nsys","--out={file:*.nsys-rep}"]}'):  # prefixed embed
        with pytest.raises(ValueError):
            C.load_config({**base, "MCP_ANALYZERS": bad})


def test_load_config_region_fallback_and_strip():
    cfg = C.load_config({"MCP_MLFLOW_TRACKING_URI": " arn:x ", "AWS_DEFAULT_REGION": "eu-west-1",
                         "MCP_TRACE_BUCKET": " b "})
    assert cfg.region == "eu-west-1" and cfg.tracking_uri == "arn:x" and cfg.trace_bucket == "b"


# --- analyzers ------------------------------------------------------------------------------

def test_inventory_analyzer(tmp_path):
    (tmp_path / "model.neff").write_bytes(b"abc")
    sub = tmp_path / "d"; sub.mkdir(); (sub / "run.ntff").write_bytes(b"de")
    out = A.inventory_analyzer(str(tmp_path), 10)
    assert "model.neff\t3" in out
    assert os.path.join("d", "run.ntff") in out or "d/run.ntff" in out
    assert "total_bytes=5" in out


def test_inventory_analyzer_missing_dir():
    with pytest.raises(FileNotFoundError):
        A.inventory_analyzer("/no/such/dir", 10)


def test_command_analyzer_argv_no_shell(tmp_path):
    (tmp_path / "f").write_bytes(b"x")
    ca = A.CommandAnalyzer(name="echo", argv_template=("/bin/echo", "analyzed:{dir}"))
    out = ca(str(tmp_path), 10)
    assert f"analyzed:{tmp_path}" in out


def test_command_analyzer_reports_nonzero_exit(tmp_path):
    ca = A.CommandAnalyzer(name="false", argv_template=("/bin/sh", "-c", "exit 3"))
    # note: this uses /bin/sh only because the TEST wants a nonzero exit; the template tokens are
    # still passed as argv (no interpolation of {dir} into a shell string).
    out = ca(str(tmp_path), 10)
    assert "exited 3" in out


def _dir_with(tmp_path, *names):
    d = tmp_path / "run"; d.mkdir()
    for n in names:
        (d / n).write_bytes(b"x")
    return str(d)


def test_find_matches_recursive(tmp_path):
    d = _dir_with(tmp_path, "a.nsys-rep", "b.neff", "c.ntff")
    (tmp_path / "run" / "sub").mkdir(); (tmp_path / "run" / "sub" / "d.neff").write_bytes(b"y")
    assert [p.rsplit("/", 1)[-1] for p in A.find_matches(d, "*.neff")] == ["b.neff", "d.neff"]


def test_command_analyzer_file_token_single(tmp_path):
    d = _dir_with(tmp_path, "prof.nsys-rep", "notes.txt")
    ca = A.CommandAnalyzer(name="echo", argv_template=("/bin/echo", "{file:*.nsys-rep}"))
    out = ca(d, 10)
    assert out.strip().endswith("prof.nsys-rep")


def test_command_analyzer_file_token_ambiguous_raises(tmp_path):
    d = _dir_with(tmp_path, "a.neff", "b.neff")
    ca = A.CommandAnalyzer(name="echo", argv_template=("/bin/echo", "{file:*.neff}"))
    with pytest.raises(FileNotFoundError):
        ca(d, 10)  # 2 matches for a {file:} token = ambiguous, loud error


def test_command_analyzer_files_token_multiple(tmp_path):
    d = _dir_with(tmp_path, "k.neff", "p.ntff")
    ca = A.CommandAnalyzer(name="echo", argv_template=("/bin/echo", "{files:*.n*ff}"))
    out = ca(d, 10)
    assert "k.neff" in out and "p.ntff" in out


def test_command_analyzer_tmp_token_is_writable(tmp_path):
    """{tmp} expands to a writable scratch dir; a tool can write its export there even though the
    run dir is read-only (R1)."""
    d = _dir_with(tmp_path, "prof.nsys-rep")
    ca = A.CommandAnalyzer(name="sh", argv_template=("/bin/sh", "-c", "echo hi > {tmp}/out.txt && cat {tmp}/out.txt"))
    out = ca(d, 10)
    assert "hi" in out


def test_command_analyzer_drops_noise_lines(tmp_path):
    d = _dir_with(tmp_path, "f")
    ca = A.CommandAnalyzer(name="x", argv_template=("/bin/sh", "-c", "echo 'SKIPPED: a'; echo real; echo 'Processing [b]'"),
                           drop_line_prefixes=("SKIPPED:", "Processing ["))
    out = ca(d, 10)
    assert out.strip() == "real"


def test_command_analyzer_annotates_empty_success(tmp_path):
    d = _dir_with(tmp_path, "f")
    ca = A.CommandAnalyzer(name="quiet", argv_template=("/bin/sh", "-c", "echo 'SKIPPED: x'"),
                           drop_line_prefixes=("SKIPPED:",))
    out = ca(d, 10)
    assert "no findings" in out and "quiet" in out


def test_builtin_analyzers_registered():
    for name in ("inventory", "nsys-stats", "nsys-analyze", "neuron-summary"):
        assert A.resolve_analyzer(name) is not None
    # neuron-summary targets the .neff + .ntff pair via {file:} tokens
    assert A.resolve_analyzer("neuron-summary").argv_template[:2] == ("neuron-explorer", "view")


def test_validate_argv_template_rejects_typo_and_embed():
    A.validate_argv_template(["nsys", "stats", "{file:*.nsys-rep}", "{dir}", "{tmp}/x"])  # ok
    for bad in (["{flie:*.neff}"], ["--input={file:*.nsys-rep}"], ["{files:*.neff"], ["{file}"]):
        with pytest.raises(ValueError):
            A.validate_argv_template(bad)


def test_resolve_artifacts_by_run_id(tmp_path):
    svc = _service_with_run(tmp_path)  # aliasA/run1 has model.neff + run.ntff
    out = svc.resolve_artifacts("run1", pattern="*.neff")
    assert len(out["matches"]) == 1 and out["matches"][0].endswith("/aliasA/run1/model.neff")
    assert out["chip"] == "neuron"


def test_resolve_artifacts_by_alias_chip(tmp_path):
    svc = _service_with_run(tmp_path)
    out = svc.resolve_artifacts(alias="aliasA", chip="neuron", pattern="*")
    assert {m.rsplit("/", 1)[-1] for m in out["matches"]} == {"model.neff", "run.ntff"}


def test_resolve_artifacts_needs_selector(tmp_path):
    svc = _service_with_run(tmp_path)
    with pytest.raises(ValueError):
        svc.resolve_artifacts()  # neither run_id nor alias+chip


def test_resolve_analyzer_unknown():
    with pytest.raises(ValueError):
        A.resolve_analyzer("nope")


# --- service (fake store) -------------------------------------------------------------------

@dataclass
class FakeRun:
    run_id: str
    chip: str
    region: str
    workload_id: str
    artifacts_uri: str
    metrics: dict = field(default_factory=dict)
    start_time: int = 0
    tags: dict = field(default_factory=dict)


@dataclass
class FakeStore:
    """Stands in for ExperimentStore: resolve() + locate() over a temp 'mount'."""
    mount_base: str
    runs: dict = field(default_factory=dict)          # run_id -> FakeRun
    by_alias: dict = field(default_factory=dict)       # alias -> [run_id]

    def resolve(self, alias_or_id, by="alias"):
        if by == "id":
            return [self.runs[alias_or_id]] if alias_or_id in self.runs else []
        return [self.runs[r] for r in self.by_alias.get(alias_or_id, [])]

    def search(self, *, alias=None, filter_string="", order_by=None, max_results=1000):
        # minimal fake: scope by alias if given, else all; support a single tags.K = 'V' filter
        pool = ([self.runs[r] for r in self.by_alias.get(alias, [])] if alias
                else list(self.runs.values()))
        if filter_string.startswith("tags."):
            k, v = filter_string[len("tags."):].split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            pool = [r for r in pool if r.tags.get(k) == v]
        return pool[:max_results]

    def locate(self, run):
        # mimic experiment_store.local_dir_for_mount: <mount_base>/<key-prefix>
        from urllib.parse import urlparse
        key = urlparse(run.artifacts_uri).path.lstrip("/")
        return os.path.join(self.mount_base, key)


def _service_with_run(tmp_path, chip="neuron"):
    mount = tmp_path / "mount"
    rundir = mount / "aliasA" / "run1"
    rundir.mkdir(parents=True)
    (rundir / "model.neff").write_bytes(b"NEFFDATA")
    (rundir / "run.ntff").write_bytes(b"NT")
    run = FakeRun(run_id="run1", chip=chip, region="ap-northeast-1", workload_id="w",
                  artifacts_uri="s3://mcp-traces-x/aliasA/run1/", metrics={"cosine": 0.999})
    store = FakeStore(mount_base=str(mount), runs={"run1": run}, by_alias={"aliasA": ["run1"]})
    return AnalysisService(store, analyzer_timeout_s=10)


def test_stage_lists_files_in_place(tmp_path):
    svc = _service_with_run(tmp_path)
    out = svc.stage("run1")
    assert out["count"] == 2
    assert set(out["files"]) == {"model.neff", "run.ntff"}
    assert out["dir"].rstrip("/").endswith("/aliasA/run1")


def test_stage_unknown_run(tmp_path):
    svc = _service_with_run(tmp_path)
    with pytest.raises(LookupError):
        svc.stage("nope")


def test_analyze_inventory_returns_advice_not_bytes(tmp_path):
    svc = _service_with_run(tmp_path)
    out = svc.analyze("run1", analyzer="inventory")
    assert out["analyzer"] == "inventory"
    assert "model.neff\t8" in out["advice"]
    # advice must not contain the raw artifact bytes
    assert "NEFFDATA" not in out["advice"]
