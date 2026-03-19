use crate::{Channel, EnergyGrid};

/// Cross section table sigma(E).
///
/// All values in millibarns (mb).
#[derive(Debug, Clone)]
pub struct CrossSection {
    pub energy: EnergyGrid,
    /// Total cross section (mb).
    pub sigma_total: Vec<f64>,
    /// Elastic scattering cross section (mb).
    pub sigma_elastic: Vec<f64>,
    /// Reaction (non-elastic) cross section (mb).
    pub sigma_reaction: Vec<f64>,
    /// Channel-resolved partial cross sections.
    pub partial: Vec<PartialCrossSection>,
}

/// Partial cross section for a specific exit channel.
#[derive(Debug, Clone)]
pub struct PartialCrossSection {
    pub channel: Channel,
    /// Cross section values (mb), corresponding to the parent CrossSection's energy grid.
    pub sigma: Vec<f64>,
}
