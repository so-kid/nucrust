/// Astrophysical reaction rate result.
#[derive(Debug, Clone)]
pub struct ReactionRate {
    /// Temperatures (GK).
    pub temperatures: Vec<f64>,
    /// N_A <sigma v> (cm^3/mol/s).
    pub na_sigma_v: Vec<f64>,
    /// Maxwellian-averaged cross section (mb), per kT.
    pub macs: Option<Vec<f64>>,
    /// Astrophysical S-factor (MeV*b), charged particles only.
    pub s_factor: Option<Vec<f64>>,
    /// Stellar enhancement factor.
    pub sef: Option<Vec<f64>>,
}
