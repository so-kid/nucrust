//! Parsers for the RIPL-3 (Reference Input Parameter Library) data files.
//!
//! RIPL-3 is the IAEA reference library providing nuclear structure and
//! reaction input parameters. Each submodule handles one category of data
//! files distributed with RIPL-3.

/// Fission barrier parameters (`fission/fission-barriers-*.dat`).
pub mod fission;
/// Gamma-ray strength function data (GDR parameters and tabulated GSF).
pub mod gamma_strength;
/// Nuclear level density parameters (phenomenological and HFB microscopic).
pub mod level_density;
/// Discrete nuclear level schemes (`levels/z???.dat`).
pub mod levels;
/// Nuclear mass table (`masses/mass-*.dat`).
pub mod mass;
/// Optical model parameter database (`om-parameter-u.dat`).
pub mod omp;
/// Average resonance parameters (`resonances/resonances?.dat`).
pub mod resonances;
/// Shell and pairing correction energies.
pub mod shell_corrections;
/// Shared types for RIPL-3 data structures.
pub mod types;

pub use fission::{parse_fission_barriers, FissionBarrier};
pub use gamma_strength::{parse_gdr_params, parse_gsf_table, GdrParams, GsfTable, GsfTableEntry};
pub use level_density::{
    parse_hfb_density_table, parse_level_density_params, HfbDensityEntry, HfbDensityTable,
    LevelDensityParams,
};
pub use levels::parse_discrete_levels;
pub use mass::parse_mass_table;
pub use omp::{parse_omp_database, OmpParameterSet};
pub use resonances::{parse_resonances, ResonanceParams};
pub use shell_corrections::{parse_shell_corrections, ShellCorrection};
pub use types::*;
