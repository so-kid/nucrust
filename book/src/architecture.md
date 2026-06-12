# Crate Architecture

nucrust is organized as a Cargo workspace with 10 crates, each handling a specific domain.

## Dependency Graph

Arrows point from each crate to its internal dependencies:

```text
nucrust-core ──────── types, traits, constants (no internal deps)

nucrust-special ───── Coulomb wave functions     ← core
nucrust-data ──────── RIPL-3 / REACLIB / HDF5    ← core

nucrust-optical ───── optical model, Numerov     ← core, special, data
nucrust-hf ────────── Hauser-Feshbach model      ← core, optical, data
nucrust-rmatrix ───── R-matrix theory, fitting   ← core, special, data

nucrust-astro ─────── MACS, rates, S-factors     ← core, data, hf, rmatrix
nucrust-gpu ───────── CUDA backend (standalone)  ← core, special

nucrust-python ────── PyO3 bindings              ← all CPU crates
nucrust ───────────── aggregator + CLI           ← all CPU crates
```

Note that `nucrust-gpu` is an independent compute backend: it depends only on
`nucrust-core` and `nucrust-special`, and is *not* re-exported by the
aggregate `nucrust` crate (enable it directly with its `cuda` feature).

## Crate Summaries

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| `nucrust-core` | Shared types and traits | `Nuclide`, `Channel`, `SpinParity`, `ComputeBackend` |
| `nucrust-special` | Mathematical special functions | `CoulombResult`, `coulomb_wave()` |
| `nucrust-data` | Nuclear data I/O | `parse_discrete_levels()`, `ReaclibEntry` |
| `nucrust-optical` | Optical model calculations | `KoningDelaroche`, `compute_transmission_coeffs()` |
| `nucrust-hf` | Hauser-Feshbach model | `HfCalculation`, `cascade_calculation()` |
| `nucrust-rmatrix` | R-matrix theory | `rmatrix_cross_section()`, `levenberg_marquardt()` |
| `nucrust-astro` | Astrophysical quantities | `compute_macs()`, `compute_reaction_rate()` |
| `nucrust-gpu` | GPU acceleration | `GpuBackend`, CUDA kernel management |
| `nucrust-python` | Python interface | PyO3 wrapper functions |
| `nucrust` | Aggregator + CLI | `CpuBackend`, `clap` CLI |

## Design Principles

- **Type-driven**: Strong types for physical quantities prevent unit errors
- **Zero-cost abstractions**: Physics models are traits with monomorphic dispatch
- **Backend transparency**: CPU and GPU backends share the `ComputeBackend` trait
- **Modular**: Each crate can be used independently
