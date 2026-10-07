"""Validate kernels/coulomb_device.cu on the host, without CUDA.

Compiles the kernel source as C++ (device qualifiers stubbed out), runs the
coulomb_batch_device kernel over every point of
tests/reference_data/coulomb_mpmath.json and checks F, G, F', G' (signed)
against the 1e-12 relative-error target. On a CUDA machine the same check runs
on the GPU as `cargo test -p nucrust-gpu --features cuda gpu_coulomb_mpmath`.

    uv run check_coulomb_device_host.py            # from scripts/
    uv run check_coulomb_device_host.py --no-fma   # disable FMA contraction
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-12

HARNESS = r"""
#include <cmath>
#include <cstdio>
#include <vector>
#define __device__
#define __global__
#define __restrict__
struct Dim { int x; };
static Dim blockIdx{0}, blockDim{1}, threadIdx{0};
#include "coulomb_device.cu"
int main() {
  std::vector<double> eta, rho;
  std::vector<int> l;
  double e, r;
  int li;
  while (scanf("%d %lf %lf", &li, &e, &r) == 3) {
    l.push_back(li); eta.push_back(e); rho.push_back(r);
  }
  int n = (int)l.size();
  std::vector<double> f(n), g(n), fp(n), gp(n);
  std::vector<int> st(n);
  for (int i = 0; i < n; i++) {
    blockIdx.x = i;
    coulomb_batch_device(eta.data(), rho.data(), l.data(), f.data(), g.data(),
                         fp.data(), gp.data(), st.data(), n);
  }
  for (int i = 0; i < n; i++)
    printf("%d %.17e %.17e %.17e %.17e\n", st[i], f[i], g[i], fp[i], gp[i]);
  return 0;
}
"""


def rel_err(computed: float, reference: float) -> float:
    if abs(reference) < 1e-300:
        return abs(computed)
    return abs((computed - reference) / reference)


def main() -> int:
    fma = "--no-fma" not in sys.argv[1:]
    ref = json.loads((ROOT / "tests/reference_data/coulomb_mpmath.json").read_text())
    points = [p for group in ref["data"].values() for p in group if p.get("error") is None]

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "harness.cpp"
        exe = Path(tmp) / "harness"
        src.write_text(HARNESS)
        cxx = os.environ.get("CXX", "c++")
        subprocess.run(
            [cxx, "-std=c++17", "-O2", f"-ffp-contract={'fast' if fma else 'off'}",
             "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "kernels"), str(src), "-o", str(exe)],
            check=True,
        )
        stdin = "".join(f"{p['l']} {p['eta']!r} {p['rho']!r}\n" for p in points)
        out = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, check=True)

    worst = dict.fromkeys(["F", "G", "Fp", "Gp", "W"], 0.0)
    failures = 0
    for p, line in zip(points, out.stdout.splitlines(), strict=True):
        status, *vals = line.split()
        f, g, fp, gp = map(float, vals)
        errs = {k: rel_err(v, p[k]) for k, v in zip(["F", "G", "Fp", "Gp"], [f, g, fp, gp])}
        errs["W"] = abs(fp * g - f * gp - 1.0)
        for k, v in errs.items():
            worst[k] = max(worst[k], v)
        if int(status) != 0 or not all(v < TOL for v in errs.values()):
            failures += 1
            print(f"FAIL l={p['l']} eta={p['eta']} rho={p['rho']} status={status} "
                  + " ".join(f"{k}={v:.1e}" for k, v in errs.items()) + f" [{p['label']}]")

    print(f"{len(points)} points, {failures} failures (FMA {'on' if fma else 'off'}); max rel err "
          + ", ".join(f"{k}={v:.2e}" for k, v in worst.items()))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
