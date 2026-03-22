# nucrust

**nucrust** is a GPU-accelerated nuclear reaction rate calculation framework written in Rust.

It implements the Hauser-Feshbach statistical model and R-matrix theory for computing nuclear reaction cross sections and astrophysical reaction rates, targeting applications in nuclear astrophysics.

## Key Features

- **Hauser-Feshbach statistical model** with width fluctuation corrections (Moldauer, GOE)
- **R-matrix theory** with Lane-Thomas and Brune parameterizations
- **GPU acceleration** via CUDA for batch calculations (50-500x speedup over Fortran codes)
- **Multiple physics models**: 5 nuclear level density models, 3 gamma-ray strength function models, 4 optical model potentials
- **Astrophysical quantities**: MACS, reaction rates, S-factors, stellar enhancement factors
- **Data I/O**: RIPL-3, REACLIB, HDF5, TOML configuration
- **Python bindings** via PyO3 for integration with existing workflows

## Target Audience

- Nuclear astrophysicists computing reaction rates for nucleosynthesis simulations
- Nuclear physicists analyzing resonance data with R-matrix fitting
- Researchers needing high-throughput cross section calculations across the nuclear chart

## Comparison with Existing Codes

| Feature | TALYS | AZURE2 | nucrust |
|---------|-------|--------|---------|
| Hauser-Feshbach | Yes | No | Yes |
| R-matrix | No | Yes | Yes |
| GPU acceleration | No | No | Yes |
| Language | Fortran | Fortran/C++ | Rust |
| Python API | No | No | Yes |
| REACLIB output | No | No | Yes |
