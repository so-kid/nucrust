//! Astrophysical reaction rate NA<σv>(T) computation.
//!
//! NA<σv> = NA · (8/πμ)^{1/2} · (kT)^{-1/2} · MACS(kT)

use nucrust_core::units::BOLTZMANN_MEV;
use nucrust_core::{CoreError, ReactionRate};

/// Compute astrophysical reaction rate NA<σv>(T) from MACS values.
///
/// NA<σv> = NA · (8/πμ)^{1/2} · (kT)^{-1/2} · MACS(kT)
///
/// where μ is the reduced mass in amu, kT in MeV, MACS in mb.
/// Result in cm³/mol/s.
pub fn compute_reaction_rate(
    macs_mb: &[f64],
    temperatures_t9: &[f64],
    reduced_mass_amu: f64,
) -> Result<ReactionRate, CoreError> {
    if macs_mb.len() != temperatures_t9.len() {
        return Err(CoreError::InvalidParameter {
            name: "temperatures",
            value: temperatures_t9.len() as f64,
            reason: "macs and temperatures must have same length",
        });
    }

    // Prefactor: NA · sqrt(8/(π·μ)) in appropriate units
    // NA = 6.022e23 /mol
    // sqrt(8/(π·μ·c²)) with μ in MeV → need unit conversion
    // Working in natural units: ℏ = c = 1, then convert
    //
    // NA<σv> = NA · sqrt(8/(π·μ)) · (kT)^{-1/2} · MACS
    // with μ in MeV/c², kT in MeV, MACS in mb = 1e-27 cm²
    //
    // velocity units: v/c = sqrt(2E/μc²), so sqrt(8/πμ) has units of 1/sqrt(MeV)
    // Need: [cm³/mol/s] = [1/mol] · [cm/s] · [cm²]
    //
    // Conversion factor: ℏc = 197.327 MeV·fm, 1 fm = 1e-13 cm
    // σ in cm²: 1 mb = 1e-27 cm²
    // v = c · sqrt(2E/μc²)
    let na_sigma_v: Vec<f64> = macs_mb
        .iter()
        .zip(temperatures_t9.iter())
        .map(|(&macs, &t9)| {
            let kt = BOLTZMANN_MEV * t9; // MeV
            if kt < 1e-15 || macs <= 0.0 {
                return 0.0;
            }

            // NA<σv> = NA · sqrt(8/(π·μ_MeV)) · (kT)^{-1/2} · MACS
            // Units: need to convert to cm³/mol/s
            //
            // Standard formula (Fowler, Caughlan, Zimmerman):
            // NA<σv> = 3.7318e10 · (μ·T9)^{-1/2} · MACS(mb) · exp_factor
            // where μ is in amu, T9 in GK
            //
            // The constant 3.7318e10 = NA · sqrt(8/(π·m_u·k_B)) in CGS
            // with m_u in g, k_B in erg/K
            let constant = 3.7318e10; // cm³/(mol·s·mb) for μ in amu, T9 in GK
            constant / (reduced_mass_amu * t9).sqrt() * macs
        })
        .collect();

    Ok(ReactionRate {
        temperatures: temperatures_t9.to_vec(),
        na_sigma_v,
        macs: Some(macs_mb.to_vec()),
        s_factor: None,
        sef: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reaction_rate_positive() {
        let macs = vec![100.0, 80.0, 60.0]; // mb
        let temps = vec![0.3, 1.0, 3.0]; // GK
        let mu = 0.97; // amu (roughly neutron on Fe-56)

        let rate = compute_reaction_rate(&macs, &temps, mu).unwrap();
        assert_eq!(rate.na_sigma_v.len(), 3);
        for &r in &rate.na_sigma_v {
            assert!(r > 0.0 && r.is_finite(), "rate = {}", r);
        }
    }

    #[test]
    fn reaction_rate_increases_with_macs() {
        let macs1 = vec![100.0];
        let macs2 = vec![200.0];
        let temps = vec![1.0];
        let mu = 1.0;

        let r1 = compute_reaction_rate(&macs1, &temps, mu).unwrap();
        let r2 = compute_reaction_rate(&macs2, &temps, mu).unwrap();
        assert!(r2.na_sigma_v[0] > r1.na_sigma_v[0]);
    }

    #[test]
    fn reaction_rate_stores_macs() {
        let macs = vec![50.0, 40.0];
        let temps = vec![0.5, 1.0];
        let rate = compute_reaction_rate(&macs, &temps, 1.0).unwrap();
        assert!(rate.macs.is_some());
        assert_eq!(rate.macs.unwrap(), macs);
    }
}
