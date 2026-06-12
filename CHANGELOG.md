# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 0.1.0

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
