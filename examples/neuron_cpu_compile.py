"""Produce an AWS Neuron .neff on CPU (no Trainium/Inferentia) — a producer-side example.

torch_neuronx.trace() can't compile without a device, but neuronx-cc (the compiler) is CPU-only and
accepts an XLA HLO. So we lower the model to HLO on the CPU XLA backend (which HAS a device) and
hand that HLO to neuronx-cc. Run inside the CPU compiler image (infra/bench/Dockerfile.neuron-cc):

    python3 neuron_cpu_compile.py --out /work/model.neff

The resulting .neff is a real compiled Neuron artifact; upload it to the trace bucket like any other
profiler output (experiment_store.log(..., artifacts=[neff])) so the analysis-mcp can serve it.
NOTE: a runtime PROFILE (.ntff) still requires executing on a Neuron device — this gives the
compiled graph, not a runtime trace.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile


def build_model():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 64)).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model.neff")
    ap.add_argument("--target", default="trn2")
    args = ap.parse_args()

    dump = tempfile.mkdtemp(prefix="hlo-")
    # CPU XLA backend => a device exists => we can lower to HLO without any Neuron hardware.
    os.environ["PJRT_DEVICE"] = "CPU"
    os.environ["XLA_FLAGS"] = f"--xla_dump_to={dump} --xla_dump_hlo_as_proto"

    import torch
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    model = build_model().to(dev)
    x = torch.rand(4, 128).to(dev)
    _ = model(x)
    xm.mark_step()  # forces the graph to be traced + dumped

    hlos = sorted(glob.glob(os.path.join(dump, "*before_optimizations.hlo.pb")))
    if not hlos:
        raise SystemExit(f"no HLO proto dumped under {dump}")
    hlo = hlos[0]
    print(f"HLO: {os.path.basename(hlo)} ({os.path.getsize(hlo)} bytes)")

    proc = subprocess.run(
        ["neuronx-cc", "compile", "--framework", "XLA", "--target", args.target, hlo,
         "--output", args.out],
        check=False)
    if proc.returncode != 0 or not os.path.exists(args.out):
        raise SystemExit(f"neuronx-cc failed (rc={proc.returncode})")
    print(f"NEFF: {args.out} ({os.path.getsize(args.out)} bytes) — Compiler status PASS")


if __name__ == "__main__":
    main()
