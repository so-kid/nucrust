# Crate Architecture

nucrust is organized as a Cargo workspace with 10 crates, each handling a specific domain.

## Dependency Graph

```
nucrust-core          (types, traits, constants)
    ↓
nucrust-special       (Coulomb wave functions)
    ↓
nucrust-data          (RIPL-3, REACLIB, HDF5 parsers)
nucrust-optical       (optical model, Numerov integration)
    ↓
nucrust-hf            (Hauser-Feshbach statistical model)
nucrust-rmatrix       (R-matrix theory, fitting)
    ↓
nucrust-astro         (MACS, reaction rates, S-factors)
    ↓
nucrust-gpu           (CUDA kernels, GPU backend)
    ↓
nucrust-python        (PyO3 bindings)
nucrust               (aggregator crate, CLI)
```

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
