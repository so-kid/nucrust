# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

#### Coulomb wave functions (`nucrust-special`)
- G_l for l > 0 in the classically forbidden region had relative errors up to
  0.76. The l = 0 normalization took G_0 from Numerov integration started at a
  point whose sign came from the asymptotic phase. That sign is wrong near the
  turning point, so G_0 had the wrong sign, the Wronskian-derived G'_0 was
  inconsistent, and the upward recurrence carried the error to all l > 0.
- The sign of F (and G) is now exact: it is tracked through the CF1 Lentz
  iteration (Barnett COULFG) instead of being inferred from the asymptotic
  phase, which gave the wrong sign for F and G near the turning point
  (e.g. eta = 5, rho = 10).
- The l = 0 normalization now picks the most accurate of Steed's method (CF1 +
  CF2), the 1F1 power series for F with G = (1 - qF^2)/(F' - pF) from CF2, and
  a rho shift (Steed at rho = 2 eta, high-order Taylor integration of G
  inward). The O(h^4) Numerov fallback is removed. F, G, F' and G' now agree
  with mpmath to < 1e-13 (F, G) and < 1e-12 (F', G') on all 106 reference
  points, including deep in the forbidden region, at small rho, and for eta
  up to 200.
- `gamow_factor` evaluates C_0 in closed form plus an upward recurrence in l.
  The log-gamma route lost ~4 digits at large eta.
- The 1F1 series recurs on the terms A_k rho^k directly; A_k and rho^k no
  longer overflow separately at large rho.
- The SIMD batch path (`simd` feature) shares the scalar normalization. It
  previously applied Steed at l_min with the asymptotic-phase sign and an
  upward F recurrence, and returned unconverged continued fractions silently.

### Added
- mpmath reference data: forbidden-region (l > 0, near the turning point,
  large eta), small-rho, and attractive (eta < 0) groups. Reference F'/G' are
  now exact (from the l recurrence) instead of finite differences. They are
  validated together with F, G and the Wronskian, and values are compared with
  their signs.
- Criterion benchmark target `cargo bench -p nucrust-special` (per-regime
  `coulomb_wave` timings and a SIMD batch).
- Genuine TALYS-2.25 golden reference data for Fe-56(n,x) and U-238(n,x)
  under `tests/reference_data/talys/`: capture, elastic, non-elastic, total
  and reaction cross sections, and spin-averaged / j-split neutron
  transmission coefficients, together with the raw TALYS outputs and inputs
  and `scripts/extract_talys_golden.py`, which regenerates the tables
  byte-identically from the raw outputs

### Fixed
- `ConstantTemperature` level density (and the CT part of `GilbertCameron`)
  returned 0 for excitation energies below the shift `E0`. The CT formula
  ρ(E) = (1/T)·exp((E−E0)/T) now holds for all E ≥ 0 and is zero only below
  the ground state. With a positive E0 (e.g. 1.94 MeV for ⁵⁷Fe) this restores
  the γ final states in 0 ≤ U < E0; Fe-56 σ(n,γ) rises by 20–27 % between
  10 keV and 1 MeV

## [0.1.0] - 2026-10-07

### Added

#### macOS / Apple Silicon optimization
- `parallel` feature (rayon) for `nucrust-special`, `nucrust-optical`, `nucrust-hf`,
  and the `nucrust` aggregate crate: energy-parallel transmission coefficients,
  Hauser-Feshbach summation, and batch Coulomb wave functions (results identical
  to the sequential path)
- Documented `-C target-cpu=apple-m1` codegen tuning for
  `aarch64-apple-darwin` builds
- `simd` feature forwarding on the `nucrust` aggregate crate (NEON via `wide`
  on Apple Silicon)
- macOS (arm64) CI job running the test suite with `parallel`/`simd` features
- Book chapter "macOS / Apple Silicon Optimization"

#### Documentation overhaul
- Root `README.md` with project status, build instructions, and workspace overview
- `LICENSE-MIT` and `LICENSE-APACHE` files matching the declared dual license
- Compile-checked quickstart example (`crates/nucrust/examples/quickstart.rs`)
- mdBook CI job (`mdbook build` + `mdbook test`)
- Freeze notices on the historical design documents in `docs/`

#### Documentation fixes
- Book API examples corrected to match the implemented APIs
  (`HfCalculation`/`HfConfig`, astro function signatures, R-matrix
  `BoundaryCondition` and fitting signatures, GPU `batch_pipeline`,
  `CubicSpline::natural`, `SpinParity`)
- Python bindings chapter rewritten for the actual module
  (`import nucrust_python`, tuple/object return types, neutron-only note)
- CLI reference now documents implementation status (only `calc` with
  `json`/`table` output is functional)
- Repository URLs corrected to `so-kid/nucrust` (Cargo.toml, book.toml,
  installation chapter); `docs/reserch_RQ01-09.md` renamed to
  `docs/research_RQ01-09.md`
- `CLAUDE.md` updated to reflect the implemented state (cudarc 0.16, actual
  dependency graph, parser approach, test infrastructure status)

#### Build and CI fixes
- `nucrust-python`: build against the stable ABI (`abi3-py39`) so the workspace
  builds with any CPython ≥ 3.9 (including 3.14), and add a `build.rs` that
  emits `-undefined dynamic_lookup` so plain `cargo build` links on macOS
- CI: the all-features job now tests every non-CUDA feature (installs
  libhdf5-dev); the `cuda` feature is compile-checked in an `nvidia/cuda`
  container since it needs `nvcc` at build time
- Installation / macOS book chapters, README, and CLAUDE.md updated for the
  `parallel` / `simd` features and package-scoped `--features` commands

#### Phase 0: Project Foundation
- Initial project structure with Cargo workspace and documentation
- Software Requirements Specification (SRS v0.2.1)
- Detailed design document with crate architecture, type definitions, and algorithm pseudocode
- Technical research notes (RQ-01 through RQ-09) covering Coulomb functions, RIPL-3 format, cudarc API, and more
- Prerequisite task list for Phase 3/4 handoff

#### Phase 1: Core Types, Special Functions, and Data Parsers

**nucrust-core**
- `Nuclide` type with Z/A encoding and arithmetic
- `SpinParity` type for nuclear quantum numbers
- Unit system definitions for nuclear physics quantities
- Common error types and result aliases
- `CpuBackend` trait definition for compute abstraction

**nucrust-special**
- Coulomb wave functions (F, G, F', G') via Thompson-Barnett COULCC algorithm
- Continued-fraction (CF1/CF2) evaluation with Steed's method
- Support for arbitrary angular momentum l and Sommerfeld parameter eta
- Criterion benchmarks for performance regression tracking

**nucrust-data**
- RIPL-3 discrete level parser (energies, spins, parities, branching ratios)
- RIPL-3 level density parameter parser
- REACLIB reaction rate format parser
- REACLIB 7-parameter fitting routines
- TOML configuration file support
- HDF5 I/O with round-trip serialization tests

**Testing & Validation (Phase 1)**
- mpmath 50-digit reference data for Coulomb function validation (ACC-01)
- Reference data files: RIPL-3 samples, REACLIB samples
- TALYS comparison script for generating golden-file data
- CI/CD pipeline configuration

#### Phase 2: Optical Model, Hauser-Feshbach, and R-matrix

**nucrust-optical**
- Koning-Delaroche global optical model potential (neutrons, protons)
- Numerov method integrator for radial Schrodinger equation
- Transmission coefficient calculation for all partial waves
- Coupled-channel optical model with full N x N S-matrix extraction
- SIMD-accelerated batch computation of transmission coefficients

**nucrust-hf**
- Hauser-Feshbach statistical model cross section calculation
- Multi-channel support: neutron, proton, alpha, and gamma emission
- Width-fluctuation correction (WFC) via GOE triple integral
- Multi-particle emission cascade framework
- Statistical gamma-ray cascade with E1/M1/E2 multipole competition
- Gilbert-Cameron composite level density model

**nucrust-rmatrix**
- R-matrix formalism (Lane-Thomas parametrization)
- Brune alternative parametrization with level-shift transformation
- Channel radius and boundary condition management
- Collision matrix (U-matrix) construction and cross section extraction

**Aggregator Crate**
- `CpuBackend` implementation in nucrust root crate connecting all subsystems
- End-to-end integration tests (E2E) for full reaction calculation pipeline
- Property-based tests (proptest) verifying physics constraints (cross section >= 0, detailed balance)

#### Phase 3: GPU Acceleration

**nucrust-gpu**
- GPU backend foundation with cudarc v0.16 device management
- Batch Numerov kernel: 2D grid (energy x partial wave) parallel execution
- Hauser-Feshbach summation kernel with warp-reduce optimization
- MACS (Maxwellian-Averaged Cross Section) integration kernel
- R-matrix batched LU solve kernel via cuSOLVER
- GPU Coulomb wave function device routines with lookup table acceleration
- Hybrid CPU/GPU computation strategy with automatic fallback
- Mixed-precision detection (FP32 + iterative refinement, relative error < 1e-5)
- Multi-stream executor for concurrent kernel dispatch
- Warp divergence binning for optimal GPU occupancy
- `ComputeBackend` trait implementation for seamless CPU/GPU switching
- hipify-clang compatibility verified for future ROCm portability

**Benchmarks**
- GPU vs CPU Coulomb function benchmark suite
- U-238 neutron capture cross section benchmark
- Hauser-Feshbach warm-start performance measurement
- End-to-end pipeline throughput benchmarks

#### Phase 4: Astrophysics and Python Bindings

**nucrust-astro**
- Maxwellian-Averaged Cross Section (MACS) calculation
- Stellar Enhancement Factor (SEF) computation
- Astrophysical S-factor extraction
- REACLIB 7-parameter rate fitting from calculated cross sections

**CLI (nucrust binary)**
- `calc` subcommand for single-reaction cross section calculation
- `batch` subcommand for multi-target systematic calculations
- `fit` subcommand for REACLIB parameter fitting
- TOML-based input configuration

**nucrust-python**
- PyO3 bindings for core types (Nuclide, SpinParity)
- rust-numpy integration for array I/O
- Python API for cross section and reaction rate calculation
- maturin build system integration

### Fixed

- Coulomb wave function CF2 algorithm for l_min > 0: rewrote to always compute from l=0 and use forward recurrence
- Koning-Delaroche OMP parameters: corrected transcription from original paper tables
- Hauser-Feshbach total width calculation: now includes all exit channels (n, p, alpha, gamma)
- Clippy warnings resolved across all crates (zero-warning policy)

### Changed

- GPU backend upgraded from initial cudarc integration to full pipeline with Coulomb tables and hybrid strategy
- R-matrix solver enhanced from basic Lane-Thomas to include Brune parametrization
- Optical model extended from spherical to coupled-channel formalism

[Unreleased]: https://github.com/so-kid/nucrust/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/so-kid/nucrust/releases/tag/v0.1.0
