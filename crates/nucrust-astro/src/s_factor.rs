//! Astrophysical S-factor computation.
//!
//! S(E) = σ(E) · E · exp(2πη)
//!
//! Removes the dominant Coulomb barrier penetration factor to reveal
//! the nuclear physics content of the cross section.

use nucrust_core::{Channel, CoreError, CrossSection};

/// Compute the astrophysical S-factor for a charged-particle reaction.
///
/// S(E) = σ(E) · E · exp(2πη)
///
/// where η is the Sommerfeld parameter.
/// Returns S(E) in MeV·b for each energy point.
pub fn compute_s_factor(
    cross_section: &CrossSection,
    channel: &Channel,
) -> Result<Vec<f64>, CoreError> {
    let z1 = channel.projectile.z() as f64;
    let z2 = channel.target.z() as f64;

    if z1 == 0.0 || z2 == 0.0 {
        return Err(CoreError::InvalidParameter {
            name: "channel",
            value: 0.0,
            reason: "S-factor is only defined for charged-particle reactions",
        });
    }

    let mu =
        nucrust_core::units::reduced_mass(channel.projectile.mass_amu(), channel.target.a() as f64);

    let energies = cross_section.energy.as_slice();
    let sigmas = &cross_section.sigma_reaction;

    let s_factors: Vec<f64> = energies
        .iter()
        .zip(sigmas.iter())
        .map(|(&e, &sigma)| {
            if e <= 0.0 || sigma <= 0.0 {
                return 0.0;
            }
            let eta = nucrust_core::units::sommerfeld_parameter(z1, z2, mu, e);
            // σ in mb, convert to b: σ_b = σ_mb * 1e-3
            // S in MeV·b = σ_b · E · exp(2πη)
            let sigma_b = sigma * 1e-3;
            sigma_b * e * (2.0 * std::f64::consts::PI * eta).exp()
        })
        .collect();

    Ok(s_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::{EnergyGrid, Nuclide, Projectile};

    #[test]
    fn s_factor_for_proton_capture() {
        let ch = Channel {
            projectile: Projectile::Proton,
            target: Nuclide::new(4, 7).unwrap(), // 7Be
            q_value: 0.137,
        };
        let energies = EnergyGrid::from_values(vec![0.1, 0.5, 1.0]).unwrap();
        let xs = CrossSection {
            energy: energies,
            sigma_total: vec![0.001, 0.1, 1.0], // mb
            sigma_elastic: vec![0.0; 3],
            sigma_reaction: vec![0.001, 0.1, 1.0],
            partial: vec![],
        };

        let sf = compute_s_factor(&xs, &ch).unwrap();
        assert_eq!(sf.len(), 3);
        for &s in &sf {
            assert!(s > 0.0 && s.is_finite(), "S = {}", s);
        }
        // S-factor should be more slowly varying than cross section
    }

    #[test]
    fn s_factor_error_for_neutrons() {
        let ch = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        };
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let xs = CrossSection {
            energy: energies,
            sigma_total: vec![100.0],
            sigma_elastic: vec![0.0],
            sigma_reaction: vec![100.0],
            partial: vec![],
        };
        assert!(compute_s_factor(&xs, &ch).is_err());
    }
}
