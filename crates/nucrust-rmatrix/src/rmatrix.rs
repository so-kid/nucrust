//! Core R-matrix cross section computation (Lane-Thomas formalism).
//!
//! Implements the Lane-Thomas R-matrix formalism:
//! - Build R-matrix R_{cc'}(E) = sum_lambda gamma_{lc} * gamma_{lc'} / (E_lambda - E)
//! - Compute (1 - R*L^0)^{-1} using faer LU decomposition
//! - Build collision matrix U with penetrability P_c and Coulomb phase Omega_c
//! - Compute cross sections from U-matrix

use faer::complex_native::c64;
use faer::prelude::*;
use nucrust_core::units::{sommerfeld_parameter, wave_number};
use nucrust_core::{
    Channel, CollisionMatrix, CoreError, CrossSection, EnergyGrid, PartialCrossSection,
};
use nucrust_special::coulomb_wave;
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::types::{BoundaryCondition, RMatrixParams, RMatrixResult};

/// Compute R-matrix cross sections over a range of energies.
///
/// For each energy E:
/// 1. Build R-matrix: R_{cc'}(E) = sum_lambda gamma_{lc} * gamma_{lc'} / (E_lambda - E)
/// 2. Compute L^0 diagonal matrix (shift - boundary + i*penetrability)
/// 3. Solve (1 - R*L^0)^{-1} via LU decomposition
/// 4. A = (1 - R*L^0)^{-1} * R
/// 5. Build collision matrix U_{cc'} = Omega_c * [delta_{cc'} + 2i * sqrt(P_c) * A_{cc'} * sqrt(P_c')] * Omega_c'
/// 6. Compute cross sections from U
pub fn rmatrix_cross_section(
    params: &RMatrixParams,
    energies: &[f64],
) -> Result<RMatrixResult, CoreError> {
    let n_ch = params.n_channels();
    let n_e = energies.len();

    if n_ch == 0 {
        return Err(CoreError::InvalidParameter {
            name: "channels",
            value: 0.0,
            reason: "must have at least one channel",
        });
    }
    if n_e == 0 {
        return Err(CoreError::InvalidParameter {
            name: "energies",
            value: 0.0,
            reason: "must have at least one energy point",
        });
    }

    // Validate levels have correct number of reduced widths
    for (i, level) in params.levels.iter().enumerate() {
        if level.reduced_widths.len() != n_ch {
            return Err(CoreError::InvalidParameter {
                name: "reduced_widths",
                value: level.reduced_widths.len() as f64,
                reason: "each level must have one reduced width per channel",
            });
        }
        // Check level index is valid (just use i for context)
        let _ = i;
    }

    // Validate boundary condition
    if let BoundaryCondition::Standard { b } = &params.boundary_condition {
        if b.len() != n_ch {
            return Err(CoreError::InvalidParameter {
                name: "boundary_condition.b",
                value: b.len() as f64,
                reason: "boundary condition vector must have one value per channel",
            });
        }
    }

    // Precompute channel properties for all energies
    let mut u_matrix = vec![Complex64::new(0.0, 0.0); n_e * n_ch * n_ch];
    let mut sigma_total = vec![0.0_f64; n_e];
    let mut sigma_elastic = vec![0.0_f64; n_e];
    let mut sigma_reaction = vec![0.0_f64; n_e];

    for (ei, &energy) in energies.iter().enumerate() {
        // Compute channel quantities: k, eta, rho, P, S, Omega for each channel
        let channel_data = compute_channel_data(params, energy)?;

        // Build R-matrix: R_{cc'}(E) = sum_lambda gamma_{lc} * gamma_{lc'} / (E_lambda - E)
        let r_matrix = build_r_matrix(params, energy);

        // Build L^0 diagonal: L^0_c = S_c - B_c + i*P_c
        let l0_diag = build_l0_diagonal(params, &channel_data);

        // Solve (1 - R*L^0)^{-1} * R = A
        let a_matrix = solve_channel_matrix(n_ch, &r_matrix, &l0_diag)?;

        // Build collision matrix U
        let u_at_e = build_collision_matrix(n_ch, &a_matrix, &channel_data);

        // Store U-matrix
        for c in 0..n_ch {
            for cp in 0..n_ch {
                u_matrix[ei * n_ch * n_ch + c * n_ch + cp] = u_at_e[c * n_ch + cp];
            }
        }

        // Compute cross sections from U-matrix
        // For the entrance channel (channel 0):
        // sigma_total = (pi/k^2) * (2J+1) / ((2s1+1)(2s2+1)) * sum_c (delta_{0c} - Re(U_{0c}))  -- not standard
        // Actually: sigma_el = pi/k^2 * |1 - U_{00}|^2 (for single J-pi)
        // sigma_reaction = pi/k^2 * (1 - |U_{00}|^2)
        // sigma_cc' = pi/k^2 * |delta_{cc'} - U_{cc'}|^2
        if let Some(cd) = channel_data.first() {
            let k = cd.k;
            let prefactor = PI / (k * k); // fm^2
            let fm2_to_mb = 10.0; // 1 fm^2 = 10 mb

            // Elastic: sigma_el = pi/k^2 * |1 - U_{00}|^2
            let u00 = u_at_e[0];
            let one_minus_u00 = Complex64::new(1.0, 0.0) - u00;
            sigma_elastic[ei] = prefactor * one_minus_u00.norm_sqr() * fm2_to_mb;

            // Reaction (non-elastic): sigma_reaction = pi/k^2 * (1 - |U_{00}|^2)
            sigma_reaction[ei] = prefactor * (1.0 - u00.norm_sqr()) * fm2_to_mb;

            // Total: sigma_total = sigma_elastic + sigma_reaction
            // Or equivalently: sigma_total = pi/k^2 * 2*(1 - Re(U_{00}))
            sigma_total[ei] = prefactor * 2.0 * (1.0 - u00.re) * fm2_to_mb;
        }
    }

    let energy_grid = EnergyGrid::from_values(energies.to_vec())?;
    let collision_matrix = CollisionMatrix {
        energies: energies.to_vec(),
        n_channels: n_ch,
        u_matrix,
    };

    // Build partial cross sections for each non-entrance channel
    let mut partials = Vec::new();
    if n_ch > 1 {
        for cp in 1..n_ch {
            let ch = &params.channels[cp];
            let channel = Channel {
                projectile: ch.pair.light,
                target: ch.pair.heavy,
                q_value: ch.pair.q_value,
            };
            let sigma: Vec<f64> = (0..n_e)
                .map(|ei| {
                    let u_0cp = collision_matrix.get(ei, 0, cp);
                    if let Some(cd) = compute_channel_data(params, energies[ei])
                        .ok()
                        .and_then(|v| v.first().cloned())
                    {
                        let k = cd.k;
                        let prefactor = PI / (k * k) * 10.0; // fm^2 -> mb
                        prefactor * u_0cp.norm_sqr()
                    } else {
                        0.0
                    }
                })
                .collect();
            partials.push(PartialCrossSection { channel, sigma });
        }
    }

    let cross_sections = CrossSection {
        energy: energy_grid,
        sigma_total,
        sigma_elastic,
        sigma_reaction,
        partial: partials,
    };

    Ok(RMatrixResult {
        collision_matrix,
        cross_sections,
    })
}

/// Per-channel data computed at a specific energy.
///
/// `eta`, `rho`, and `is_open` are currently only used for diagnostics
/// (via the `Debug` impl), hence the per-field `dead_code` allowances.
#[derive(Debug, Clone)]
pub(crate) struct ChannelData {
    /// Wave number k (fm^-1).
    pub k: f64,
    /// Sommerfeld parameter eta.
    #[allow(dead_code)]
    pub eta: f64,
    /// Dimensionless parameter rho = k * a.
    #[allow(dead_code)]
    pub rho: f64,
    /// Penetrability P_c.
    pub penetrability: f64,
    /// Shift function S_c.
    pub shift: f64,
    /// Coulomb phase factor Omega_c = exp(i * sigma_l).
    pub omega: Complex64,
    /// Whether the channel is open (E_cm > 0).
    #[allow(dead_code)]
    pub is_open: bool,
}

/// Compute channel data (k, eta, rho, P, S, Omega) for all channels at a given energy.
pub(crate) fn compute_channel_data(
    params: &RMatrixParams,
    energy: f64,
) -> Result<Vec<ChannelData>, CoreError> {
    let mut data = Vec::with_capacity(params.n_channels());

    for ch in &params.channels {
        let pair = &ch.pair;
        let mu = pair.reduced_mass_amu();
        let z1 = pair.light.z() as f64;
        let z2 = pair.heavy.z() as f64;

        // Center-of-mass energy for this channel
        // For entrance channel: E_cm = E
        // For exit channel: E_cm = E + Q
        let e_cm = energy + pair.q_value;

        if e_cm <= 0.0 {
            // Closed channel
            data.push(ChannelData {
                k: 0.0,
                eta: 0.0,
                rho: 0.0,
                penetrability: 0.0,
                shift: 0.0,
                omega: Complex64::new(1.0, 0.0),
                is_open: false,
            });
            continue;
        }

        let k = wave_number(mu, e_cm);
        let eta = sommerfeld_parameter(z1, z2, mu, e_cm);
        let rho = k * ch.radius;

        // Compute Coulomb wave functions at channel radius
        let (penetrability, shift, omega) = if rho > 1e-10 {
            let cw = coulomb_wave(eta, rho, ch.l, 1).map_err(coulomb_convergence_err)?;

            let (s, p) = shift_penetrability_from_waves(cw.f[0], cw.g[0], cw.fp[0], cw.gp[0], rho);

            // Omega_c = exp(i * sigma_l)
            let sigma_l = cw.sigma[0];
            let omega = Complex64::new(sigma_l.cos(), sigma_l.sin());

            (p, s, omega)
        } else {
            // Very small rho: use analytic limits
            let (s, p) = shift_penetrability_small_rho(ch.l, rho);
            (p, s, Complex64::new(1.0, 0.0))
        };

        data.push(ChannelData {
            k,
            eta,
            rho,
            penetrability,
            shift,
            omega,
            is_open: true,
        });
    }

    Ok(data)
}

/// Build the R-matrix: R_{cc'}(E) = sum_lambda gamma_{lc} * gamma_{lc'} / (E_lambda - E)
fn build_r_matrix(params: &RMatrixParams, energy: f64) -> Vec<Complex64> {
    let n_ch = params.n_channels();
    let mut r = vec![Complex64::new(0.0, 0.0); n_ch * n_ch];

    for level in &params.levels {
        let denom = level.energy - energy;
        if denom.abs() < 1e-30 {
            // Near a pole: use a small regularization
            continue;
        }
        let inv_denom = 1.0 / denom;

        for c in 0..n_ch {
            for cp in 0..n_ch {
                r[c * n_ch + cp] += Complex64::new(
                    level.reduced_widths[c] * level.reduced_widths[cp] * inv_denom,
                    0.0,
                );
            }
        }
    }

    r
}

/// Build the L^0 diagonal: L^0_c = S_c - B_c + i*P_c
fn build_l0_diagonal(params: &RMatrixParams, channel_data: &[ChannelData]) -> Vec<Complex64> {
    let n_ch = params.n_channels();
    let mut l0 = vec![Complex64::new(0.0, 0.0); n_ch];

    for (c, cd) in channel_data.iter().enumerate() {
        let b_c = match &params.boundary_condition {
            BoundaryCondition::Standard { b } => b[c],
            BoundaryCondition::Brune => cd.shift, // B_c = S_c(E_lambda) but at current E
        };
        l0[c] = Complex64::new(cd.shift - b_c, cd.penetrability);
    }

    l0
}

/// Solve (1 - R*L^0)^{-1} * R = A using faer LU decomposition.
fn solve_channel_matrix(
    n_ch: usize,
    r_matrix: &[Complex64],
    l0_diag: &[Complex64],
) -> Result<Vec<Complex64>, CoreError> {
    // Build (1 - R*L^0)
    let mut m = faer::Mat::<c64>::zeros(n_ch, n_ch);

    for c in 0..n_ch {
        for cp in 0..n_ch {
            let rl0 = r_matrix[c * n_ch + cp] * l0_diag[cp];
            let delta = if c == cp { 1.0 } else { 0.0 };
            let val = Complex64::new(delta, 0.0) - rl0;
            m[(c, cp)] = c64::new(val.re, val.im);
        }
    }

    // Build RHS: R-matrix as a faer matrix
    let mut rhs = faer::Mat::<c64>::zeros(n_ch, n_ch);
    for c in 0..n_ch {
        for cp in 0..n_ch {
            let r = r_matrix[c * n_ch + cp];
            rhs[(c, cp)] = c64::new(r.re, r.im);
        }
    }

    // Solve (1 - R*L^0) * A = R  =>  A = (1 - R*L^0)^{-1} * R
    let lu = m.partial_piv_lu();
    let a_mat = lu.solve(&rhs);

    // Convert back to Vec<Complex64>
    let mut a = vec![Complex64::new(0.0, 0.0); n_ch * n_ch];
    for c in 0..n_ch {
        for cp in 0..n_ch {
            let v = a_mat[(c, cp)];
            a[c * n_ch + cp] = Complex64::new(v.re, v.im);
        }
    }

    Ok(a)
}

/// Build the collision matrix:
/// U_{cc'} = Omega_c * [delta_{cc'} + 2i * sqrt(P_c) * A_{cc'} * sqrt(P_c')] * Omega_c'
fn build_collision_matrix(
    n_ch: usize,
    a_matrix: &[Complex64],
    channel_data: &[ChannelData],
) -> Vec<Complex64> {
    let mut u = vec![Complex64::new(0.0, 0.0); n_ch * n_ch];
    let two_i = Complex64::new(0.0, 2.0);

    for c in 0..n_ch {
        let omega_c = channel_data[c].omega;
        let sqrt_p_c = channel_data[c].penetrability.sqrt();

        for cp in 0..n_ch {
            let omega_cp = channel_data[cp].omega;
            let sqrt_p_cp = channel_data[cp].penetrability.sqrt();

            let delta = if c == cp {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };

            let a_ccp = a_matrix[c * n_ch + cp];

            u[c * n_ch + cp] = omega_c * (delta + two_i * sqrt_p_c * a_ccp * sqrt_p_cp) * omega_cp;
        }
    }

    u
}

/// Map a Coulomb wave function failure onto the core error type.
fn coulomb_convergence_err<E>(_: E) -> CoreError {
    CoreError::ConvergenceFailure {
        algorithm: "coulomb_wave",
        iterations: 0,
        residual: 0.0,
    }
}

/// Compute (S_c, P_c) from Coulomb wave function values at the channel radius:
/// - P_c = rho / (F^2 + G^2)
/// - S_c = rho * (F*F' + G*G') / (F^2 + G^2)
fn shift_penetrability_from_waves(f: f64, g: f64, fp: f64, gp: f64, rho: f64) -> (f64, f64) {
    let f2_g2 = f * f + g * g;
    if f2_g2 > 1e-300 {
        (rho * (f * fp + g * gp) / f2_g2, rho / f2_g2)
    } else {
        (0.0, 0.0)
    }
}

/// Analytic small-rho limits: for l=0, P = rho and S = 0; for l>0, P ~ 0 and S ~ -(l+1).
fn shift_penetrability_small_rho(l: u32, rho: f64) -> (f64, f64) {
    if l == 0 {
        (0.0, rho)
    } else {
        (-(l as f64 + 1.0), 0.0)
    }
}

/// Compute the shift and penetrability functions from Coulomb wave functions.
///
/// Returns (S_c, P_c) where:
/// - P_c = rho / (F^2 + G^2)
/// - S_c = rho * (F*F' + G*G') / (F^2 + G^2)
pub fn shift_penetrability(l: u32, eta: f64, rho: f64) -> Result<(f64, f64), CoreError> {
    if rho < 1e-15 {
        return Ok(shift_penetrability_small_rho(l, rho));
    }

    let cw = coulomb_wave(eta, rho, l, 1).map_err(coulomb_convergence_err)?;
    Ok(shift_penetrability_from_waves(
        cw.f[0], cw.g[0], cw.fp[0], cw.gp[0], rho,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use nucrust_core::{Nuclide, Parity, Projectile, SpinParity};

    /// Create a simple 1-level, 1-channel neutron scattering problem.
    fn single_level_single_channel() -> RMatrixParams {
        let pair = ParticlePair {
            light: Projectile::Neutron,
            heavy: Nuclide::new(6, 12).unwrap(),
            q_value: 0.0,
            separation_energy: 4.946,
        };
        let channel = RMatrixChannel {
            pair,
            l: 0,
            s: 0.5,
            j: SpinParity::new(1, Parity::Positive).unwrap(),
            radius: 5.0,
        };
        RMatrixParams {
            channels: vec![channel],
            levels: vec![RMatrixLevel {
                energy: 1.0,               // 1 MeV resonance
                reduced_widths: vec![0.3], // sqrt(MeV)
            }],
            boundary_condition: BoundaryCondition::Standard { b: vec![0.0] },
        }
    }

    #[test]
    fn test_single_level_breit_wigner() {
        // For a single level, single channel, the cross section should follow
        // the Breit-Wigner form near the resonance energy.
        let params = single_level_single_channel();
        let energies: Vec<f64> = (1..=100).map(|i| i as f64 * 0.05).collect();

        let result = rmatrix_cross_section(&params, &energies).unwrap();

        // Cross sections should be non-negative
        for &s in &result.cross_sections.sigma_total {
            assert!(s >= 0.0, "Total cross section must be >= 0, got {}", s);
        }
        for &s in &result.cross_sections.sigma_elastic {
            assert!(s >= 0.0, "Elastic cross section must be >= 0, got {}", s);
        }
        for &s in &result.cross_sections.sigma_reaction {
            assert!(
                s >= -1e-10,
                "Reaction cross section must be >= 0, got {}",
                s
            );
        }

        // There should be a peak near E = 1 MeV (resonance energy)
        let peak_idx = result
            .cross_sections
            .sigma_elastic
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let peak_energy = energies[peak_idx];
        // The peak should be within ~0.5 MeV of the resonance energy (shifted by boundary condition)
        assert!(
            (peak_energy - 1.0).abs() < 1.0,
            "Peak at {} MeV, expected near 1.0 MeV",
            peak_energy
        );
    }

    #[test]
    fn test_smatrix_unitarity() {
        // For a single-channel problem, |U_{00}|^2 should equal 1 (unitarity).
        let params = single_level_single_channel();
        let energies: Vec<f64> = (1..=50).map(|i| i as f64 * 0.1).collect();

        let result = rmatrix_cross_section(&params, &energies).unwrap();

        for (ei, &e) in energies.iter().enumerate() {
            let u00 = result.collision_matrix.get(ei, 0, 0);
            let norm_sq = u00.norm_sqr();
            assert!(
                (norm_sq - 1.0).abs() < 1e-8,
                "S-matrix unitarity violated at E={} MeV: |U_00|^2 = {} (err = {:.2e})",
                e,
                norm_sq,
                (norm_sq - 1.0).abs()
            );
        }
    }

    #[test]
    fn test_two_channel_unitarity() {
        // Two-channel problem: sum_c |U_{ac}|^2 = 1 for each entrance channel a.
        let pair1 = ParticlePair {
            light: Projectile::Neutron,
            heavy: Nuclide::new(6, 12).unwrap(),
            q_value: 0.0,
            separation_energy: 4.946,
        };
        let pair2 = ParticlePair {
            light: Projectile::Neutron,
            heavy: Nuclide::new(6, 12).unwrap(),
            q_value: -0.5,
            separation_energy: 4.446,
        };
        let j = SpinParity::new(1, Parity::Positive).unwrap();
        let ch1 = RMatrixChannel {
            pair: pair1,
            l: 0,
            s: 0.5,
            j,
            radius: 5.0,
        };
        let ch2 = RMatrixChannel {
            pair: pair2,
            l: 1,
            s: 0.5,
            j,
            radius: 5.0,
        };
        let params = RMatrixParams {
            channels: vec![ch1, ch2],
            levels: vec![RMatrixLevel {
                energy: 2.0,
                reduced_widths: vec![0.3, 0.2],
            }],
            boundary_condition: BoundaryCondition::Standard { b: vec![0.0, 0.0] },
        };

        // Use energies above the Q-value threshold so both channels are open
        let energies: Vec<f64> = (1..=20).map(|i| 0.5 + i as f64 * 0.2).collect();
        let result = rmatrix_cross_section(&params, &energies).unwrap();

        for (ei, &e) in energies.iter().enumerate() {
            for a in 0..2 {
                let mut row_sum = 0.0;
                for c in 0..2 {
                    let u_ac = result.collision_matrix.get(ei, a, c);
                    row_sum += u_ac.norm_sqr();
                }
                // Only check if channel is open (E_cm > 0 for both channels)
                let e_cm2 = e + params.channels[a].pair.q_value;
                if e_cm2 > 0.0 {
                    assert!(
                        (row_sum - 1.0).abs() < 1e-6,
                        "Unitarity violated at E={} ch={}: sum|U|^2 = {} (err = {:.2e})",
                        e,
                        a,
                        row_sum,
                        (row_sum - 1.0).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn test_shift_penetrability_neutron_l0() {
        // For neutron (eta=0), l=0: P_0 = rho, S_0 = 0
        let rho = 1.0;
        let (s, p) = shift_penetrability(0, 0.0, rho).unwrap();
        assert!(
            (s).abs() < 1e-8,
            "S_0(eta=0, rho=1) should be ~0, got {}",
            s
        );
        assert!(
            (p - rho).abs() < 1e-8,
            "P_0(eta=0, rho=1) should be ~rho, got {}",
            p
        );
    }

    #[test]
    fn test_shift_penetrability_neutron_l1() {
        // For neutron (eta=0), l=1:
        // P_1 = rho^3 / (1 + rho^2)
        // S_1 = -1 / (1 + rho^2)
        let rho: f64 = 2.0;
        let expected_p = rho.powi(3) / (1.0 + rho * rho);
        let expected_s = -1.0 / (1.0 + rho * rho);

        let (s, p) = shift_penetrability(1, 0.0, rho).unwrap();
        assert!(
            (p - expected_p).abs() < 1e-6,
            "P_1(eta=0, rho=2): got {}, expected {}",
            p,
            expected_p
        );
        assert!(
            (s - expected_s).abs() < 1e-6,
            "S_1(eta=0, rho=2): got {}, expected {}",
            s,
            expected_s
        );
    }

    #[test]
    fn test_empty_channels_error() {
        let params = RMatrixParams {
            channels: vec![],
            levels: vec![],
            boundary_condition: BoundaryCondition::Standard { b: vec![] },
        };
        assert!(rmatrix_cross_section(&params, &[1.0]).is_err());
    }

    #[test]
    fn test_empty_energies_error() {
        let params = single_level_single_channel();
        assert!(rmatrix_cross_section(&params, &[]).is_err());
    }
}
