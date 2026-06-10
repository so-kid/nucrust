// Internal unit conventions (all modules):
// - Energy: MeV, Length: fm, Time: s
// - Cross section: mb, Temperature: GK, Reaction rate: cm^3/mol/s, Mass: amu

// Physical constants (CODATA 2022)

/// hbar*c (MeV*fm)
pub const HBAR_C: f64 = 197.3269804;
/// 1 amu in MeV/c^2
pub const AMU_MEV: f64 = 931.49410372;
/// Fine-structure constant alpha
pub const FINE_STRUCTURE: f64 = 7.2973525643e-3;
/// Boltzmann constant (MeV/GK)
pub const BOLTZMANN_MEV: f64 = 8.617333262e-2;
/// Avogadro's number (1/mol) -- exact since 2019 SI redefinition
pub const AVOGADRO: f64 = 6.02214076e23;

/// NA<sigma*v> prefactor: NA * sqrt(8/(pi*m_u*k_B)) in CGS units.
///
/// Converts MACS to a reaction rate via the Fowler-Caughlan-Zimmerman formula
/// NA<sigma*v> = NA_SIGMA_V_PREFACTOR / sqrt(mu * T9) * MACS,
/// in cm^3/(mol*s*mb) for the reduced mass mu in amu and T9 in GK.
pub const NA_SIGMA_V_PREFACTOR: f64 = 3.7318e10;

/// Sommerfeld parameter: eta = Z1*Z2*alpha*sqrt(mu*c^2 / (2*E_cm))
///
/// where alpha is the fine-structure constant.
pub fn sommerfeld_parameter(z1: f64, z2: f64, mu_amu: f64, e_cm_mev: f64) -> f64 {
    // eta = Z1 * Z2 * e^2 / (hbar * v)
    //     = Z1 * Z2 * alpha * sqrt(mu * c^2 / (2 * E_cm))
    z1 * z2 * FINE_STRUCTURE * (mu_amu * AMU_MEV / (2.0 * e_cm_mev)).sqrt()
}

/// Wave number k (fm^-1).
///
/// k = sqrt(2 * mu * E_cm) / hbar_c
pub fn wave_number(mu_amu: f64, e_cm_mev: f64) -> f64 {
    (2.0 * mu_amu * AMU_MEV * e_cm_mev).sqrt() / HBAR_C
}

/// Reduced mass (amu).
pub fn reduced_mass(m1_amu: f64, m2_amu: f64) -> f64 {
    m1_amu * m2_amu / (m1_amu + m2_amu)
}

/// Convert lab energy to center-of-mass energy (MeV).
pub fn cm_energy(e_lab: f64, m_proj: f64, m_targ: f64) -> f64 {
    e_lab * m_targ / (m_proj + m_targ)
}

/// Convert MeV to keV.
#[inline]
pub fn mev_to_kev(e: f64) -> f64 {
    e * 1e3
}

/// Convert barn to millibarn.
#[inline]
pub fn barn_to_mb(s: f64) -> f64 {
    s * 1e3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_mass_symmetric() {
        let mu = reduced_mass(1.0, 1.0);
        assert!((mu - 0.5).abs() < 1e-15);
    }

    #[test]
    fn reduced_mass_heavy_target() {
        // mu ~ m_proj when m_targ >> m_proj
        let mu = reduced_mass(1.0, 1e6);
        assert!((mu - 1.0).abs() < 1e-3);
    }

    #[test]
    fn cm_energy_conversion() {
        // For equal masses, E_cm = E_lab / 2
        let e_cm = cm_energy(10.0, 1.0, 1.0);
        assert!((e_cm - 5.0).abs() < 1e-15);
    }

    #[test]
    fn sommerfeld_positive_for_charged() {
        let eta = sommerfeld_parameter(1.0, 26.0, 0.97, 1.0);
        assert!(eta > 0.0);
    }

    #[test]
    fn sommerfeld_zero_for_neutron() {
        let eta = sommerfeld_parameter(0.0, 26.0, 0.97, 1.0);
        assert_eq!(eta, 0.0);
    }

    #[test]
    fn wave_number_positive() {
        let k = wave_number(0.97, 1.0);
        assert!(k > 0.0);
    }

    #[test]
    fn unit_conversions() {
        assert!((mev_to_kev(1.0) - 1000.0).abs() < 1e-15);
        assert!((barn_to_mb(1.0) - 1000.0).abs() < 1e-15);
    }
}
