# nucrust

[![CI](https://github.com/so-kid/nucrust/actions/workflows/ci.yml/badge.svg)](https://github.com/so-kid/nucrust/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**nucrust** is a GPU-accelerated nuclear reaction rate calculation framework
written in Rust, targeting nuclear astrophysics. It implements the
Hauser-Feshbach statistical model and R-matrix theory, aiming for 50–500x
speedups over existing Fortran codes (TALYS, AZURE2) via CUDA batch execution.

> **Status:** v0.1.0 (pre-release). All core physics (Phases 1–4) is
> implemented and unit-tested; quantitative validation against TALYS/AZURE2
> and some CLI features (batch/fit/export) are still in progress. See
> [CHANGELOG.md](CHANGELOG.md) and [task.md](task.md).

## Features

- **Hauser-Feshbach statistical model** with width fluctuation corrections (Moldauer, GOE)
- **R-matrix theory** with Lane-Thomas and Brune parameterizations, LM/MCMC fitting
- **Optical model** transmission coefficients via Numerov integration
  (Koning-Delaroche, McFadden-Satchler, Avrigeanu, custom)
- **Pure-Rust Coulomb wave functions** (Thompson-Barnett COULCC algorithm)
- **Astrophysics outputs**: MACS, reaction rates, S-factors, stellar
  enhancement factors, REACLIB 7-parameter fits
- **GPU acceleration** (CUDA via `cudarc`): batched Numerov, HF summation,
  R-matrix solve, MACS integration
- **Data I/O**: RIPL-3, REACLIB, HDF5, TOML configuration
- **CPU parallelism** (rayon) and SIMD (`wide`, AVX2/NEON) — the fast path on macOS
- **Python bindings** via PyO3 + rust-numpy (abi3 wheel, CPython ≥ 3.9)

## Build

Requires Rust 1.75+.

```bash
cargo build --workspace --release
cargo test --workspace
```

Optional features live on individual crates: `cuda` (nucrust-gpu, requires
CUDA Toolkit 12+), `hdf5_io` (nucrust-data, requires HDF5 1.10.x), and the
CPU acceleration features `parallel` / `simd` (forwarded by the `nucrust`
crate, e.g. `cargo build --release -p nucrust --features "parallel simd"`).
See the user guide's Installation chapter for details.

## Quick Start

```bash
# CLI: transmission coefficients for n + ⁵⁶Fe (see book/src/quickstart.md)
target/release/nucrust calc --config fe56_neutron.toml

# Rust API example
cargo run --release -p nucrust --example quickstart
```

## Documentation

- **User guide** (mdBook): `mdbook build` → `book/build/index.html`,
  sources in [`book/src/`](book/src/SUMMARY.md)
- **API docs** (rustdoc): `cargo doc --workspace --no-deps --open`
- [`CHANGELOG.md`](CHANGELOG.md) — implemented functionality
- [`docs/`](docs/) — original design documents (frozen historical record;
  the user guide and rustdoc are authoritative for current behavior)

## Workspace Layout

| Crate | Purpose |
|-------|---------|
| `nucrust-core` | Domain types (`Nuclide`, `SpinParity`, `Channel`), units, Wigner symbols |
| `nucrust-special` | Coulomb / Bessel special functions |
| `nucrust-data` | RIPL-3 / REACLIB parsers, TOML config, HDF5 I/O |
| `nucrust-optical` | Optical model, Numerov transmission coefficients |
| `nucrust-hf` | Hauser-Feshbach model, NLD/GSF, WFC, cascade |
| `nucrust-rmatrix` | R-matrix cross sections, Brune transform, fitting |
| `nucrust-astro` | MACS, reaction rates, S-factors, SEF |
| `nucrust-gpu` | CUDA backend (feature `cuda`) |
| `nucrust-python` | PyO3 bindings (module `nucrust_python`) |
| `nucrust` | Aggregator crate + CLI binary |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
