# Core Concepts

nucrust is built around strongly-typed representations of nuclear physics concepts. This chapter introduces the key types and abstractions.

## Type-Driven Design

All physical quantities are represented by dedicated types with validation:

- **`Nuclide`** — a nucleus identified by (Z, A) with range validation
- **`Projectile`** — incident particle (n, p, d, t, ³He, α, γ)
- **`Channel`** — a reaction channel (projectile + target + Q-value)
- **`SpinParity`** — quantum numbers (J, π) with selection rule helpers
- **`EnergyGrid`** — validated energy point collections (sorted, positive)

## Compute Backends

nucrust separates physics logic from execution strategy via the `ComputeBackend` trait:

- **`CpuBackend`** — multi-threaded CPU execution using rayon
- **`GpuBackend`** — CUDA-accelerated batch execution (requires `cuda` feature)

Both backends implement the same interface, allowing transparent switching.

## Calculation Pipeline

A typical calculation flows through these stages:

1. **Data loading** — Parse nuclear data (RIPL-3 level schemes, OMP parameters)
2. **Transmission coefficients** — Solve the optical model via Numerov integration
3. **Cross sections** — Apply Hauser-Feshbach or R-matrix formalism
4. **Reaction rates** — Integrate over Maxwell-Boltzmann distribution
5. **Output** — Write results in desired format (JSON, REACLIB, HDF5)
