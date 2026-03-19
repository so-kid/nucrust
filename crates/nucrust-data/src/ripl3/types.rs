use nucrust_core::{Nuclide, Parity};

/// Discrete level data for a single isotope (from RIPL-3 levels/z???.dat).
#[derive(Debug, Clone)]
pub struct IsotopeData {
    pub nuclide: Nuclide,
    /// Number of levels in the decay scheme.
    pub n_levels: usize,
    /// Maximum level number considered complete.
    pub n_max: usize,
    /// Maximum level with unique spin/parity assignment.
    pub n_unique: usize,
    /// Neutron separation energy (MeV).
    pub sn: Option<f64>,
    /// Proton separation energy (MeV).
    pub sp: Option<f64>,
    /// Discrete levels.
    pub levels: Vec<DiscreteLevel>,
}

/// A single discrete nuclear level.
#[derive(Debug, Clone)]
pub struct DiscreteLevel {
    /// Level index (1-based).
    pub index: u32,
    /// Excitation energy (MeV).
    pub energy: f64,
    /// Spin J (None if unknown; RIPL-3 uses -1.0 for unknown).
    pub spin: Option<f64>,
    /// Parity (None if unknown; RIPL-3 uses 0 for unknown).
    pub parity: Option<Parity>,
    /// Half-life in seconds (None if unknown or stable; RIPL-3 uses -1.0).
    pub half_life: Option<f64>,
    /// Number of gamma transitions from this level.
    pub n_gammas: u32,
    /// Gamma transitions from this level.
    pub gammas: Vec<GammaTransition>,
}

/// A gamma-ray transition between levels.
#[derive(Debug, Clone)]
pub struct GammaTransition {
    /// Final (daughter) level index.
    pub final_level: u32,
    /// Gamma-ray energy (MeV).
    pub energy: f64,
    /// Photon emission probability.
    pub branching_ratio: f64,
    /// Total electromagnetic transition probability (gamma + ICC + pair).
    pub total_transition_prob: Option<f64>,
    /// Internal conversion coefficient.
    pub icc: Option<f64>,
}

/// Mass table entry from RIPL-3 masses/mass-*.dat.
#[derive(Debug, Clone)]
pub struct MassEntry {
    pub nuclide: Nuclide,
    /// Mass excess (MeV).
    pub mass_excess: f64,
    /// Binding energy per nucleon (MeV).
    pub binding_energy_per_a: Option<f64>,
    /// Beta-decay energy (MeV).
    pub beta_decay_energy: Option<f64>,
    /// Atomic mass (micro-u).
    pub atomic_mass_micro_u: Option<f64>,
}
