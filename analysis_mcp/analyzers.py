"""Pluggable analyzers: turn a run's Pod-local artifact directory into ADVICE (text).

The platform is tool-agnostic (fixed IDs, open content), so analyzers are pluggable and the
external ones are configured, not hardcoded to a specific accuracy metric:

- ``inventory`` (builtin, dependency-free): walks the staged dir and reports the files + sizes.
  Always available; used to verify the stage->analyze->advice path end to end without a heavy
  tool image, and as a safe default.
- external command analyzers (e.g. ``nsys stats`` for GPU, ``neuron-explorer view --ingest-only``
  for Neuron): registered as an argv TEMPLATE with a ``{dir}`` placeholder. Run with subprocess
  WITHOUT a shell (argv list, no shell interpolation) so a crafted path/filename cannot inject a
  command; stdout is returned as advice, truncated to a bound.

An analyzer receives the run's local directory (on the S3 Files mount) and returns a string. It
must never return raw artifact bytes — only findings/advice.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess  # noqa: S404 - argv-only, no shell; see run_command_analyzer
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_MAX_ADVICE_CHARS = 200_000  # bound the advice payload returned over MCP
_FILE_TOKEN = re.compile(r"^\{file:(.+)\}$")    # exactly one match required
_FILES_TOKEN = re.compile(r"^\{files:(.+)\}$")  # one-or-more matches, each its own argv item


def find_matches(run_dir: str, pattern: str) -> list[str]:
    """Absolute paths of files under ``run_dir`` (recursive) whose name matches the glob
    ``pattern`` (e.g. ``*.nsys-rep``), sorted. The single place run artifacts are located by
    pattern — shared by the analyzer templating below and by AnalysisService.resolve_artifacts."""
    return sorted(str(p) for p in Path(run_dir).rglob(pattern) if p.is_file())


def validate_argv_template(argv: tuple[str, ...] | list[str]) -> None:
    """Reject any token with a brace that is not a recognized placeholder, so a typo ("{flie:*}",
    "{file}" without a glob), a prefixed embed ("--sqlite={file:*}"), or an unbalanced brace fails
    loudly at REGISTRATION rather than reaching the tool as a silent literal arg.
    ``{file:GLOB}``/``{files:GLOB}`` are whole-token only; ``{dir}``/``{tmp}`` may be embedded."""
    for tok in argv:
        if _FILE_TOKEN.match(tok) or _FILES_TOKEN.match(tok):
            continue  # valid whole-token file selector
        residual = tok.replace("{dir}", "").replace("{tmp}", "")  # strip valid embeds
        if "{" in residual or "}" in residual:
            raise ValueError(
                f"analyzer token {tok!r} has a brace that is not a valid placeholder "
                f"({{dir}}, {{tmp}}, {{file:GLOB}}, {{files:GLOB}}) — fix the typo/embedding")


def _expand_token(tok: str, run_dir: str, tmp_dir: str) -> list[str]:
    """Expand one argv-template token. ``{dir}`` -> the run dir; ``{tmp}`` -> a per-call writable
    scratch dir (for a tool's export/output on an otherwise read-only mount); ``{file:GLOB}`` -> the
    single matching file (error on 0 or >1 — ambiguity is loud); ``{files:GLOB}`` -> every match as
    its own argv item (e.g. a Neuron .neff + .ntff pair). Any other token is literal. No shell."""
    if "{dir}" in tok or "{tmp}" in tok:
        return [tok.replace("{dir}", run_dir).replace("{tmp}", tmp_dir)]
    m = _FILE_TOKEN.match(tok)
    if m:
        matches = find_matches(run_dir, m.group(1))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{tok}: expected exactly one file matching {m.group(1)!r} under {run_dir}, "
                f"found {len(matches)}: {matches}")
        return matches
    m = _FILES_TOKEN.match(tok)
    if m:
        matches = find_matches(run_dir, m.group(1))
        if not matches:
            raise FileNotFoundError(f"{tok}: no file matches {m.group(1)!r} under {run_dir}")
        return matches
    return [tok]


def _bounded(text: str) -> str:
    """Cap advice length, but NEVER silently: if clipped, say so, so an LLM consuming the advice
    knows it is partial rather than treating a truncated `nsys stats` as complete."""
    if len(text) <= _MAX_ADVICE_CHARS:
        return text
    marker = "\n…[truncated {} chars]"
    budget = _MAX_ADVICE_CHARS - len(marker.format(0)) - 8  # slack for the count digits
    dropped = len(text) - budget
    return text[:budget] + marker.format(dropped)


def inventory_analyzer(run_dir: str, timeout_s: int) -> str:
    """Builtin: list files (relative path + size) under the staged dir. No external tool."""
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"staged dir not found: {run_dir}")
    lines: list[str] = [f"inventory of {run_dir}:"]
    total = 0
    count = 0
    for root, _dirs, files in os.walk(run_dir):  # single walk (readdir is costly on the S3 mount)
        for name in sorted(files):
            fp = os.path.join(root, name)
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = -1
            total += max(size, 0)
            count += 1
            lines.append(f"  {os.path.relpath(fp, run_dir)}\t{size}")
    lines.append(f"total_files={count} total_bytes={total}")
    return _bounded("\n".join(lines))


# --- shared analyzer plumbing (used by every analyzer TYPE) -----------------------------------
# The unifying contract is just Analyzer = (run_dir, timeout) -> advice; the MCP tool analyze() is
# identical for all types. What varies is HOW the tool is driven (run-to-completion vs server+query)
# — each type owns that, but they share token expansion + result shaping so the UX is consistent.

def _expand_argv(template: tuple[str, ...], run_dir: str, tmp_dir: str) -> list[str]:
    return [item for tok in template for item in _expand_token(tok, run_dir, tmp_dir)]


def _finalize(name: str, returncode: int, stdout: str, stderr: str,
              drop_prefixes: tuple[str, ...]) -> str:
    """Common result shaping: non-zero exit surfaces stderr+stdout (marker FIRST so the length
    bound can't clip the failure into a fake success); a successful run's noise lines are dropped
    and an empty result is annotated ('no findings' != 'broke'); everything is length-bounded."""
    out = stdout or ""
    if returncode != 0:
        return _bounded(f"[analyzer '{name}' exited {returncode}]\n{stderr or ''}\n---\n{out}")
    if drop_prefixes:
        out = "\n".join(ln for ln in out.splitlines() if not ln.lstrip().startswith(drop_prefixes))
    if not out.strip():
        return (f"[analyzer '{name}' ran successfully but produced no findings — "
                f"likely no matching activity in this trace]")
    return _bounded(out)


def _port_open(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


@dataclass(frozen=True)
class CommandAnalyzer:
    """Run-to-completion tool analyzer (nsys/ncu and any CLI): expand the argv template, run once,
    return stdout as advice. Tokens (NO shell): ``{dir}`` = run dir; ``{tmp}`` = writable scratch
    (the run dir is mounted READ-ONLY, so a tool that writes an export — nsys' SQLite — must target
    {tmp}; cwd is also {tmp}); ``{file:GLOB}`` = the single match; ``{files:GLOB}`` = each match as
    its own argv item (a Neuron .neff + .ntff pair). ``drop_line_prefixes`` strips a tool's noise.
    Example: ["nsys","stats","--sqlite","{tmp}/e.sqlite","{file:*.nsys-rep}"]."""
    name: str
    argv_template: tuple[str, ...]
    drop_line_prefixes: tuple[str, ...] = ()

    def __call__(self, run_dir: str, timeout_s: int) -> str:
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"staged dir not found: {run_dir}")
        tmp_dir = tempfile.mkdtemp(prefix="analyzer-")
        try:
            argv = _expand_argv(self.argv_template, run_dir, tmp_dir)
            try:
                proc = subprocess.run(  # noqa: S603 - argv list, no shell
                    argv, capture_output=True, text=True, errors="replace",
                    timeout=timeout_s, check=False, cwd=tmp_dir)
            except subprocess.TimeoutExpired as e:
                partial = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
                return _bounded(f"[analyzer '{self.name}' timed out after {timeout_s}s]\n---\n{partial or ''}")
            return _finalize(self.name, proc.returncode, proc.stdout, proc.stderr, self.drop_line_prefixes)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@dataclass(frozen=True)
class ServerQueryAnalyzer:
    """Server-then-query tool analyzer (neuron-explorer and any tool that serves results rather than
    printing them): start ``start_template`` as a background process, wait until it accepts TCP on
    ``ready_port``, run ``query_template`` (whose stdout is the advice), then tear the server down.
    Both templates use the SAME token DSL as CommandAnalyzer, so file-targeting/scratch behave
    identically — the difference from a command analyzer is the execution strategy only, not the UX.

    NOT concurrency-safe: ``ready_port`` is fixed, so two server-type analyze() calls in the same Pod
    would race on the port (the second could connect to the FIRST server and query the wrong run's
    data — a silent wrong answer). The MCP runs analyze sequentially and the chart runs replicas=1,
    so this does not arise today; if that changes, serialize server-type analyzers or template a
    dynamic ``{port}``.
    Example (neuron-explorer's REST DB on :3002):
      start: ["neuron-explorer","view","-n","{file:*.neff}","-s","{file:*.ntff}","--disable-ui"]
      query: ["curl","-sS","-X","POST","http://127.0.0.1:3002/api/v1/db/x/_search","-d","<sql json>"]"""
    name: str
    start_template: tuple[str, ...]
    ready_port: int
    query_template: tuple[str, ...]
    drop_line_prefixes: tuple[str, ...] = ()
    ready_timeout_s: int = 60

    def __call__(self, run_dir: str, timeout_s: int) -> str:
        import time
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"staged dir not found: {run_dir}")
        tmp_dir = tempfile.mkdtemp(prefix="analyzer-")
        server = None
        try:
            start = _expand_argv(self.start_template, run_dir, tmp_dir)
            query = _expand_argv(self.query_template, run_dir, tmp_dir)
            server = subprocess.Popen(  # noqa: S603 - argv list, no shell
                start, cwd=tmp_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace")
            deadline = time.monotonic() + min(self.ready_timeout_s, timeout_s)
            while time.monotonic() < deadline:
                if server.poll() is not None:  # server died before becoming ready
                    log = server.stdout.read() if server.stdout else ""
                    return _finalize(self.name, server.returncode or 1, "",
                                     f"server exited before opening port {self.ready_port}:\n{log}", ())
                if _port_open("127.0.0.1", self.ready_port):
                    break
                time.sleep(0.5)
            else:
                return _bounded(f"[analyzer '{self.name}' server never opened port "
                                f"{self.ready_port} within {min(self.ready_timeout_s, timeout_s)}s]")
            try:
                proc = subprocess.run(  # noqa: S603 - argv list, no shell
                    query, capture_output=True, text=True, errors="replace",
                    timeout=timeout_s, check=False, cwd=tmp_dir)
            except subprocess.TimeoutExpired as e:
                partial = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
                return _bounded(f"[analyzer '{self.name}' query timed out after {timeout_s}s]\n---\n{partial or ''}")
            return _finalize(self.name, proc.returncode, proc.stdout, proc.stderr, self.drop_line_prefixes)
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
            shutil.rmtree(tmp_dir, ignore_errors=True)


# name -> analyzer callable(run_dir, timeout_s) -> advice
Analyzer = Callable[[str, int], str]


def build_analyzer(name: str, spec) -> Analyzer:
    """Construct an Analyzer from a config spec (used by MCP_ANALYZERS). A bare argv list is sugar
    for a command analyzer; a dict is discriminated by ``type`` ('command' | 'server'), each with
    its own fields — so both types register uniformly while keeping their own flexibility."""
    if isinstance(spec, (list, tuple)):
        spec = {"type": "command", "argv": list(spec)}
    if not isinstance(spec, dict):
        raise ValueError(f"analyzer {name!r} spec must be an argv list or an object, got {type(spec).__name__}")
    kind = spec.get("type", "command")
    drop = tuple(spec.get("drop_line_prefixes", ()))
    if kind == "command":
        argv = spec.get("argv")
        if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
            raise ValueError(f"analyzer {name!r}: 'argv' must be a non-empty list of strings")
        validate_argv_template(argv)
        return CommandAnalyzer(name=name, argv_template=tuple(argv), drop_line_prefixes=drop)
    if kind == "server":
        start, query = spec.get("start"), spec.get("query")
        for key, val in (("start", start), ("query", query)):
            if not (isinstance(val, list) and val and all(isinstance(a, str) for a in val)):
                raise ValueError(f"analyzer {name!r}: '{key}' must be a non-empty list of strings")
            validate_argv_template(val)
        port = spec.get("ready_port")
        if not isinstance(port, int) or isinstance(port, bool):
            raise ValueError(f"analyzer {name!r}: 'ready_port' (int) is required for a server analyzer")
        rt = spec.get("ready_timeout_seconds", 60)
        return ServerQueryAnalyzer(name=name, start_template=tuple(start), ready_port=port,
                                   query_template=tuple(query), drop_line_prefixes=drop,
                                   ready_timeout_s=rt)
    raise ValueError(f"analyzer {name!r}: unknown type {kind!r} (expected 'command' or 'server')")


# Zero-config tool analyzers, pre-wired so a user does not hand-craft the {tmp}/--sqlite incantation
# (a dogfooding pain point) and does not drown in nsys' SKIPPED-section noise. They need the tool
# binary in the image (build FROM the base + the tool, e.g. Dockerfile.analysis-mcp-nsys); if the
# binary is absent the subprocess raises a clear "No such file" — they are inert on the base image.
_NSYS_DROP = ("Processing [", "SKIPPED:", "Generating SQLite")
BUILTIN_ANALYZERS: dict[str, Analyzer] = {
    "inventory": inventory_analyzer,
    "nsys-stats": CommandAnalyzer(
        name="nsys-stats",
        argv_template=("nsys", "stats", "--force-export=true", "--sqlite", "{tmp}/e.sqlite",
                       "{file:*.nsys-rep}"),
        drop_line_prefixes=_NSYS_DROP),
    "nsys-analyze": CommandAnalyzer(
        name="nsys-analyze",
        argv_template=("nsys", "analyze", "--force-export=true", "--sqlite", "{tmp}/a.sqlite",
                       "{file:*.nsys-rep}"),
        drop_line_prefixes=_NSYS_DROP),
    # Neuron runtime profile analysis. `neuron-explorer view --output-format summary-text` prints a
    # per-NeuronCore report to stdout (engine active times, FLOPS, MFU/MBU/HFU, DMA, arithmetic
    # intensity, cycle counts) — a run-to-completion command, NOT the server model (that is only the
    # 'db' output-format). Verified real-machine: a CPU-compiled .neff captured to .ntff on trn2,
    # then this exact command produced the report. Needs neuron-explorer (aws-neuronx-tools) + HOME
    # writable (the chart sets HOME=/tmp). Use summary-json for machine-readable output.
    "neuron-summary": CommandAnalyzer(
        name="neuron-summary",
        argv_template=("neuron-explorer", "view", "-n", "{file:*.neff}", "-s", "{file:*.ntff}",
                       "--output-format", "summary-text")),
}


def resolve_analyzer(name: str, extra: dict[str, Analyzer] | None = None) -> Analyzer:
    reg = {**BUILTIN_ANALYZERS, **(extra or {})}
    if name not in reg:
        raise ValueError(f"unknown analyzer {name!r}; available: {sorted(reg)}")
    return reg[name]
