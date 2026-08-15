"""The analysis service (no MCP-protocol dependency; unit-testable with fakes).

Given an experiment alias, it resolves the runs (MLflow, via experiment_store), locates each
run's artifacts on the mounted trace bucket (S3 Files — no copy), runs a per-run analyzer over
those Pod-local files, and returns ADVICE. Nothing but advice/metadata is returned — artifact
bytes stay on the Pod.
"""
from __future__ import annotations

import os
from typing import Any

from experiment_store import ExperimentStore

from .analyzers import Analyzer, find_matches, resolve_analyzer


class AnalysisService:
    def __init__(self, store: ExperimentStore, analyzer_timeout_s: int,
                 extra_analyzers: dict[str, Analyzer] | None = None):
        self._store = store
        self._timeout = analyzer_timeout_s
        self._extra = extra_analyzers or {}

    def _run_by_id(self, run_id: str):
        # resolve(by="id") filters to FINISHED, so an existing-but-FAILED run is also "not found here"
        runs = self._store.resolve(run_id, by="id")
        if not runs:
            raise LookupError(f"run {run_id} not found or not FINISHED")
        return runs[0]

    def _select_run(self, run_id: str | None, alias: str | None, chip: str | None):
        """Pick the run by explicit run_id, or the latest FINISHED run of a chip under an alias
        (the same latest-per-chip rule as compare) — the platform's id/alias -> run mapping."""
        if run_id:
            return self._run_by_id(run_id)
        if not (alias and chip):
            raise ValueError("provide run_id, or both alias and chip")
        runs = [r for r in self._store.resolve(alias) if r.chip == chip]
        if not runs:
            raise LookupError(f"no FINISHED {chip!r} run under alias {alias!r}")
        return max(runs, key=lambda r: getattr(r, "start_time", 0) or 0)

    def _staged_dir(self, run) -> str:
        """The run's Pod-local artifact dir on the S3 Files mount (no copy). Raises if it is not a
        directory: a down/misconfigured mount (or a PV bound to the wrong AZ) would otherwise make
        globbing/os.walk yield nothing, indistinguishable from a legitimately artifact-less run."""
        local = self._store.locate(run)  # <mount_base>/<alias>/<run_id>/  (raises if mount unset)
        if not os.path.isdir(local):
            raise FileNotFoundError(
                f"staged dir {local!r} for run {run.run_id} is not a directory; the S3 Files mount "
                f"is likely absent/misconfigured (or the PV is bound to a different AZ)")
        return local

    def stage(self, run_id: str) -> dict[str, Any]:
        """Return the Pod-local dir of the run's artifacts + a file inventory, without copying.
        Traverses the dir so S3 Files imports the metadata before an analyzer reads (first-access
        import can otherwise miss a freshly-synced object)."""
        run = self._run_by_id(run_id)
        local = self._staged_dir(run)
        files: list[str] = []
        for root, _dirs, fs in os.walk(local):  # traverse => triggers S3 Files import
            for f in fs:
                files.append(os.path.relpath(os.path.join(root, f), local))
        return {"run_id": run_id, "chip": run.chip, "dir": local,
                "files": sorted(files), "count": len(files)}

    def resolve_artifacts(self, run_id: str | None = None, *, alias: str | None = None,
                          chip: str | None = None, pattern: str = "*") -> dict[str, Any]:
        """Map an MLflow identity (run_id, or alias+chip) to the concrete Pod-local path(s) of the
        matching profile file(s) on the S3 Files mount — the platform's id/alias -> file-path
        contract. Returns metadata only (paths + names, never bytes). ``pattern`` is a glob over the
        run's files (e.g. ``*.nsys-rep``, ``*.neff``). Both this MCP tool (so the laptop can hand an
        external analyzer an absolute path) and CommandAnalyzer's ``{file:}``/``{files:}`` tokens
        resolve paths through the same globbing, so the mapping lives in exactly one place."""
        run = self._select_run(run_id, alias, chip)
        local = self._staged_dir(run)
        return {"run_id": run.run_id, "chip": run.chip, "dir": local,
                "pattern": pattern, "matches": find_matches(local, pattern)}

    def analyze(self, run_id: str, analyzer: str = "inventory") -> dict[str, Any]:
        """Run an analyzer over the run's staged (mounted) dir and return advice text."""
        staged = self.stage(run_id)
        fn = resolve_analyzer(analyzer, self._extra)
        advice = fn(staged["dir"], self._timeout)
        return {"run_id": run_id, "chip": staged["chip"], "analyzer": analyzer,
                "dir": staged["dir"], "advice": advice}
