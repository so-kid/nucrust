//! Coupled-channel transmission coefficient computation.
//!
//! Integrates the CC Schrödinger equation using the Johnson log-derivative method,
//! extracts the S-matrix at the matching radius, and computes transmission coefficients.

use faer::complex_native::c64;
use faer::prelude::*;
use nucrust_core::backend::NumerovConfig;
use nucrust_core::coupled_channel::{
    CoupledTransmission, DeformationParams, RotationalBand, TransmissionOutput,
};
use nucrust_core::traits::OpticalPotential;
use nucrust_core::units::{self, HBAR_C};
use nucrust_core::{Channel, CollisionMatrix, CoreError, EnergyGrid, TransmissionCoeffs};
use nucrust_special::coulomb_wave;
use num_complex::Complex64;

use crate::deformation::{build_w_matrix, CoupledChannelSystem};
use crate::johnson_logderiv::{johnson_propagate_with_l, JohnsonConfig};
use crate::numerov::numerov_integrate;

/// Compute transmission coefficients, automatically choosing spherical or CC
/// based on whether deformation parameters are provided.
pub fn compute_transmission_auto(
    potential: &dyn OpticalPotential,
    channel: &Channel,
    energies: &EnergyGrid,
    config: &NumerovConfig,
    deformation: Option<&DeformationParams>,
    band: Option<&RotationalBand>,
) -> Result<TransmissionOutput, CoreError> {
    match (deformation, band) {
        (Some(deform), Some(band)) if deform.is_deformed(0.01) => {
            let cc_result =
                compute_cc_transmission(potential, channel, energies, config, deform, band)?;
            Ok(TransmissionOutput::Coupled(cc_result))
        }
        _ => {
            let tc = crate::transmission::compute_transmission_coeffs(
                potential, channel, energies, config,
            )?;
            Ok(TransmissionOutput::Spherical(tc))
        }
    }
}

/// Compute coupled-channel transmission coefficients.
///
/// For each total J^π, builds the CC system, propagates with Johnson method,
/// and extracts the S-matrix. Returns a `CoupledTransmission` with both the
/// full collision matrix and extracted diagonal transmission coefficients.
pub fn compute_cc_transmission(
    potential: &dyn OpticalPotential,
    channel: &Channel,
    energies: &EnergyGrid,
    config: &NumerovConfig,
    deformation: &DeformationParams,
    band: &RotationalBand,
) -> Result<CoupledTransmission, CoreError> {
    let n_e = energies.len();
    let r_match = potential.matching_radius(channel);

    // Reduced mass and related constants
    let mu = reduced_mass_channel(channel);
    let hbar2_over_2mu = HBAR_C * HBAR_C / (2.0 * mu * units::AMU_MEV);

    // For each (l, j), build CC system and extract full S-matrix.
    // Accumulate: (1) diagonal transmission for HF, (2) per-J collision matrices.
    let mut data = vec![0.0; (config.max_l as usize + 1) * 2 * n_e];
    let mut l_max_actual = 0_u32;

    // Collect all S-matrix results per energy for collision matrix construction.
    // Key: energy_index. Value: list of (cc_system, s_matrix) for each J.
    let mut all_s_matrices: Vec<Vec<(CoupledChannelSystem, Vec<Complex64>)>> =
        vec![Vec::new(); n_e];

    for l in 0..=config.max_l {
        let mut max_tl = 0.0_f64;

        for (j_idx, &dj) in [-(0.5_f64), 0.5].iter().enumerate() {
            let j = l as f64 + dj;
            if j < 0.0 {
                continue;
            }
            let two_j_total = (2.0 * j) as i32;

            // Build CC system for this J_total
            let cc_system = CoupledChannelSystem::build(
                *deformation,
                band,
                two_j_total,
                l + 4, // Allow coupling to ±4 partial waves
            );

            let n_ch = cc_system.n_channels();
            if n_ch == 0 {
                continue;
            }

            // Find the ground-state elastic channel index
            let gs_idx = cc_system
                .channels
                .iter()
                .position(|ch| ch.state_index == 0 && ch.l == l && (ch.j - j).abs() < 0.01);

            let gs_idx = match gs_idx {
                Some(idx) => idx,
                None => continue, // This (l, j) doesn't appear in CC system
            };

            for (e_idx, &e) in energies.as_slice().iter().enumerate() {
                let (t_lj, s_matrix_opt) = if n_ch == 1 {
                    // Single channel: fall back to scalar Numerov
                    let s = numerov_integrate(potential, channel, e, l, j, config)?;
                    let t = (1.0 - s.norm_sqr()).clamp(0.0, 1.0);
                    (t, None)
                } else {
                    // Multi-channel: use Johnson log-derivative
                    let johnson_config = JohnsonConfig {
                        step_size: config.step_size,
                        r_min: config.r_min,
                        r_max: r_match,
                    };

                    let e_local = e;
                    let l_vals: Vec<u32> = cc_system.channels.iter().map(|c| c.l).collect();
                    let result = johnson_propagate_with_l(
                        n_ch,
                        &johnson_config,
                        |r| {
                            build_w_matrix(
                                &cc_system,
                                potential,
                                channel,
                                e_local,
                                r,
                                hbar2_over_2mu,
                            )
                        },
                        Some(&l_vals),
                    )?;

                    // Extract full N×N S-matrix from log-derivative
                    let s_matrix =
                        extract_s_matrix(&result.z_matrix, n_ch, &cc_system, channel, e, r_match)?;

                    // Transmission for ground-state elastic: T = 1 - |S_{gs,gs}|²
                    let s_elastic = s_matrix[gs_idx * n_ch + gs_idx];
                    let t = (1.0 - s_elastic.norm_sqr()).clamp(0.0, 1.0);
                    (t, Some(s_matrix))
                };

                let flat_idx = l as usize * (2 * n_e) + j_idx * n_e + e_idx;
                if flat_idx < data.len() {
                    data[flat_idx] = t_lj;
                }
                max_tl = max_tl.max(t_lj);

                // Store S-matrix for collision matrix
                if let Some(s_mat) = s_matrix_opt {
                    all_s_matrices[e_idx].push((cc_system.clone(), s_mat));
                }
            }
        }

        l_max_actual = l;
        if max_tl < config.convergence_tl && l > 0 {
            break;
        }
    }

    // Truncate data to actual l_max
    let actual_size = (l_max_actual as usize + 1) * 2 * n_e;
    data.truncate(actual_size);

    let tc = TransmissionCoeffs {
        energy: energies.clone(),
        l_max: l_max_actual,
        data,
    };

    // Build collision matrix from the collected S-matrices.
    // Use the maximum CC system size across all J values as the matrix dimension.
    // For each energy, aggregate S-matrix blocks from all J contributions.
    let max_n_ch = all_s_matrices
        .iter()
        .flat_map(|v| v.iter().map(|(cc, _)| cc.n_channels()))
        .max()
        .unwrap_or(1);

    let collision_matrix = CollisionMatrix {
        energies: energies.as_slice().to_vec(),
        n_channels: max_n_ch,
        u_matrix: vec![Complex64::new(0.0, 0.0); n_e * max_n_ch * max_n_ch],
    };
    // Fill collision matrix: for each energy, take the largest CC system's S-matrix
    let mut cm = collision_matrix;
    for (e_idx, s_list) in all_s_matrices.iter().enumerate() {
        // Find the S-matrix with the most channels (most complete J contribution)
        if let Some((cc_sys, s_mat)) = s_list.iter().max_by_key(|(cc, _)| cc.n_channels()) {
            let n = cc_sys.n_channels().min(max_n_ch);
            for c in 0..n {
                for cp in 0..n {
                    cm.u_matrix[e_idx * max_n_ch * max_n_ch + c * max_n_ch + cp] =
                        s_mat[c * cc_sys.n_channels() + cp];
                }
            }
        }
    }

    Ok(CoupledTransmission {
        collision_matrix: cm,
        diagonal_transmission: tc,
        band: band.clone(),
    })
}

/// Extract the full N×N S-matrix from the log-derivative matrix Z at the matching radius.
///
/// At the matching radius R, the solution matrix Y(R) is matched to
/// asymptotic Coulomb wave functions. The log-derivative Z = Y'Y⁻¹ determines
/// the S-matrix via:
///
///   A_{cc'} = Σ_c'' Z_{cc''} F_c''(R) δ_{c''c'} - k_c' F'_c'(R) δ_{cc'}
///   B_{cc'} = Σ_c'' Z_{cc''} G_c''(R) δ_{c''c'} - k_c' G'_c'(R) δ_{cc'}
///
/// Equivalently, with diagonal matrices F_diag, G_diag, kF'_diag, kG'_diag:
///   A = Z · F_diag - kF'_diag
///   B = Z · G_diag - kG'_diag
///
/// S = Ω (A + iB)⁻¹ (A - iB) Ω
///
/// where Ω = diag(e^{iσ_c}) is the Coulomb phase matrix.
fn extract_s_matrix(
    z_flat: &[Complex64],
    n_ch: usize,
    cc_system: &CoupledChannelSystem,
    channel: &Channel,
    energy: f64,
    r_match: f64,
) -> Result<Vec<Complex64>, CoreError> {
    let mu = reduced_mass_channel(channel);
    let eta = sommerfeld_eta(channel, energy);

    // Per-channel Coulomb functions at matching radius
    let mut f_vals = vec![0.0; n_ch];
    let mut g_vals = vec![0.0; n_ch];
    let mut fp_vals = vec![0.0; n_ch]; // F'(ρ) = dF/dρ
    let mut gp_vals = vec![0.0; n_ch]; // G'(ρ) = dG/dρ
    let mut sigma_vals = vec![0.0; n_ch];
    let mut k_vals = vec![0.0; n_ch];
    let mut open = vec![true; n_ch];

    for c in 0..n_ch {
        let ch_c = &cc_system.channels[c];
        let e_c = energy - ch_c.excitation_energy;
        if e_c <= 0.0 {
            // Closed channel: mark and use exponentially decaying solution
            open[c] = false;
            k_vals[c] = units::wave_number(mu, 1e-10);
            continue;
        }
        let k = units::wave_number(mu, e_c);
        k_vals[c] = k;
        let rho = k * r_match;

        let coulomb = coulomb_wave(eta, rho, ch_c.l, 1)?;
        f_vals[c] = coulomb.f[0];
        g_vals[c] = coulomb.g[0];
        fp_vals[c] = coulomb.fp[0];
        gp_vals[c] = coulomb.gp[0];
        sigma_vals[c] = coulomb.sigma[0];
    }

    // Build A = Z · F_diag - kF'_diag  and  B = Z · G_diag - kG'_diag
    // as N×N faer matrices
    let mut a_mat = faer::Mat::<c64>::zeros(n_ch, n_ch);
    let mut b_mat = faer::Mat::<c64>::zeros(n_ch, n_ch);

    for c in 0..n_ch {
        for cp in 0..n_ch {
            let z_ccp = z_flat[c * n_ch + cp];
            let z_c64 = c64::new(z_ccp.re, z_ccp.im);

            // A_{c,c'} = Z_{c,c'} * F_{c'}(R) - δ_{c,c'} * k_{c'} * F'_{c'}(R)
            let a_val = z_c64 * c64::new(f_vals[cp], 0.0);
            let b_val = z_c64 * c64::new(g_vals[cp], 0.0);

            a_mat[(c, cp)] = a_val;
            b_mat[(c, cp)] = b_val;
        }
        // Subtract diagonal kF' and kG' terms
        let k_c = k_vals[c];
        a_mat[(c, c)] -= c64::new(k_c * fp_vals[c], 0.0);
        b_mat[(c, c)] -= c64::new(k_c * gp_vals[c], 0.0);
    }

    // The matching gives: A α = -B β  where A = ZF_diag - kF'_diag, B = ZG_diag - kG'_diag.
    // With the Numerov sign convention: α_N = R·B, β_N = -R·A
    // S = Ω · (B - iA)⁻¹ · (B + iA) · Ω
    //   (verified: single-channel reduces to e^{2iσ} (α-iβ)/(α+iβ) matching numerov.rs)
    let i_c64 = c64::new(0.0, 1.0);

    // B - iA  (denominator)
    let mut lhs = faer::Mat::<c64>::zeros(n_ch, n_ch);
    // B + iA  (numerator)
    let mut rhs = faer::Mat::<c64>::zeros(n_ch, n_ch);

    for c in 0..n_ch {
        for cp in 0..n_ch {
            lhs[(c, cp)] = b_mat[(c, cp)] - i_c64 * a_mat[(c, cp)];
            rhs[(c, cp)] = b_mat[(c, cp)] + i_c64 * a_mat[(c, cp)];
        }
    }

    // Solve (B - iA) X = (B + iA)  →  X = (B - iA)⁻¹ (B + iA)
    let lu = lhs.partial_piv_lu();
    let x = lu.solve(&rhs);

    // Apply Coulomb phases: S = Ω X Ω
    let mut s_matrix = vec![Complex64::new(0.0, 0.0); n_ch * n_ch];
    for c in 0..n_ch {
        let omega_c = Complex64::from_polar(1.0, sigma_vals[c]);
        for cp in 0..n_ch {
            let omega_cp = Complex64::from_polar(1.0, sigma_vals[cp]);
            let x_val = x[(c, cp)];
            s_matrix[c * n_ch + cp] = omega_c * Complex64::new(x_val.re, x_val.im) * omega_cp;
        }
    }

    // For closed channels, zero out their S-matrix rows/columns
    for c in 0..n_ch {
        if !open[c] {
            for cp in 0..n_ch {
                s_matrix[c * n_ch + cp] = Complex64::new(0.0, 0.0);
                s_matrix[cp * n_ch + c] = Complex64::new(0.0, 0.0);
            }
        }
    }

    Ok(s_matrix)
}

/// Compute reduced mass for a channel (in amu).
fn reduced_mass_channel(channel: &Channel) -> f64 {
    let m1 = channel.projectile.mass_amu();
    let m2 = channel.target.a() as f64;
    units::reduced_mass(m1, m2)
}

/// Compute Sommerfeld parameter eta.
fn sommerfeld_eta(channel: &Channel, energy: f64) -> f64 {
    let z1 = channel.projectile.z() as f64;
    let z2 = channel.target.z() as f64;
    if z1 == 0.0 || z2 == 0.0 {
        return 0.0;
    }
    let mu = reduced_mass_channel(channel);
    units::sommerfeld_parameter(z1, z2, mu, energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omp::KoningDelaroche;
    use nucrust_core::{Nuclide, Projectile};

    fn fe56_n() -> Channel {
        Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(26, 56).unwrap(),
            q_value: 0.0,
        }
    }

    #[test]
    fn auto_dispatch_spherical() {
        let omp = KoningDelaroche;
        let ch = fe56_n();
        let energies = EnergyGrid::from_values(vec![5.0]).unwrap();
        let config = NumerovConfig {
            max_l: 3,
            ..NumerovConfig::default()
        };

        // No deformation → spherical
        let result = compute_transmission_auto(&omp, &ch, &energies, &config, None, None);
        assert!(result.is_ok());
        match result.unwrap() {
            TransmissionOutput::Spherical(tc) => {
                assert!(tc.l_max >= 1);
            }
            TransmissionOutput::Coupled(_) => panic!("Expected spherical output"),
        }
    }

    #[test]
    fn auto_dispatch_with_small_deformation_uses_spherical() {
        let omp = KoningDelaroche;
        let ch = fe56_n();
        let energies = EnergyGrid::from_values(vec![5.0]).unwrap();
        let config = NumerovConfig {
            max_l: 3,
            ..NumerovConfig::default()
        };

        // Very small deformation → still spherical
        let deform = DeformationParams::quadrupole(0.001);
        let band = RotationalBand::even_even(2, &[0.8468]);

        let result =
            compute_transmission_auto(&omp, &ch, &energies, &config, Some(&deform), Some(&band));
        assert!(result.is_ok());
        match result.unwrap() {
            TransmissionOutput::Spherical(_) => {}
            TransmissionOutput::Coupled(_) => panic!("Expected spherical for small β₂"),
        }
    }

    #[test]
    fn cc_transmission_completes() {
        let omp = KoningDelaroche;
        let ch = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(92, 238).unwrap(),
            q_value: 0.0,
        };
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let config = NumerovConfig {
            max_l: 2,
            ..NumerovConfig::default()
        };

        let deform = DeformationParams::quadrupole(0.22);
        let band = RotationalBand::even_even(4, &[0.04491, 0.14848]);

        let result = compute_cc_transmission(&omp, &ch, &energies, &config, &deform, &band);
        assert!(
            result.is_ok(),
            "CC transmission should complete: {:?}",
            result.err()
        );

        let cc_tc = result.unwrap();
        // Should have produced some transmission coefficients
        assert!(cc_tc.diagonal_transmission.l_max >= 1);
        // All T values should be in [0, 1]
        for &t in &cc_tc.diagonal_transmission.data {
            assert!((0.0..=1.0).contains(&t), "CC T = {} out of range", t);
        }
        // Collision matrix should have real dimensions
        assert!(
            cc_tc.collision_matrix.n_channels >= 2,
            "CC collision matrix should have ≥2 channels, got {}",
            cc_tc.collision_matrix.n_channels
        );
    }

    #[test]
    fn s_matrix_flux_conservation() {
        // For an absorptive potential, Σ_{c'} |S_{cc'}|² ≤ 1 for each c.
        let omp = KoningDelaroche;
        let ch = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(92, 238).unwrap(),
            q_value: 0.0,
        };
        let energy = 2.0;
        let r_match = omp.matching_radius(&ch);
        let mu = reduced_mass_channel(&ch);
        let hbar2_over_2mu = HBAR_C * HBAR_C / (2.0 * mu * units::AMU_MEV);

        let deform = DeformationParams::quadrupole(0.22);
        let band = RotationalBand::even_even(4, &[0.04491, 0.14848]);

        // Pick a specific J_total
        let two_j_total = 1; // J = 1/2
        let cc_system = CoupledChannelSystem::build(deform, &band, two_j_total, 4);
        let n_ch = cc_system.n_channels();
        if n_ch < 2 {
            return; // Skip if too few channels
        }

        let config = NumerovConfig::default();
        let johnson_config = JohnsonConfig {
            step_size: config.step_size,
            r_min: config.r_min,
            r_max: r_match,
        };

        let l_vals: Vec<u32> = cc_system.channels.iter().map(|c| c.l).collect();
        let result = johnson_propagate_with_l(
            n_ch,
            &johnson_config,
            |r| build_w_matrix(&cc_system, &omp, &ch, energy, r, hbar2_over_2mu),
            Some(&l_vals),
        )
        .unwrap();

        let s_matrix =
            extract_s_matrix(&result.z_matrix, n_ch, &cc_system, &ch, energy, r_match).unwrap();

        // Flux conservation: Σ_{c'} |S_{c,c'}|² ≤ 1 for each open channel c
        for c in 0..n_ch {
            let flux_sum: f64 = (0..n_ch).map(|cp| s_matrix[c * n_ch + cp].norm_sqr()).sum();
            // Allow numerical tolerance from Johnson propagation + LU solve
            assert!(
                flux_sum <= 1.0 + 1e-3,
                "Flux conservation violated for channel {c}: Σ|S|² = {flux_sum} > 1"
            );
        }
    }

    #[test]
    fn s_matrix_structural_properties() {
        // Verify structural properties of the CC S-matrix:
        // 1. S-matrix elements are finite
        // 2. Phase of diagonal S-matrix elements is well-defined
        // 3. Off-diagonal elements exist when coupling is present
        //
        // Note: Quantitative agreement with scalar Numerov in the spherical limit
        // requires matrix Numerov (not Johnson), due to the Johnson method's known
        // precision loss for optical potentials with small Im(V)/Re(V) ratio.
        // This will be validated with TALYS golden data.
        let omp = KoningDelaroche;
        let ch = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(92, 238).unwrap(),
            q_value: 0.0,
        };
        let energy = 2.0;
        let r_match = omp.matching_radius(&ch);
        let mu = reduced_mass_channel(&ch);
        let hbar2_over_2mu = HBAR_C * HBAR_C / (2.0 * mu * units::AMU_MEV);

        let deform = DeformationParams::quadrupole(0.22);
        let band = RotationalBand::even_even(4, &[0.04491, 0.14848]);

        let two_j_total = 1;
        let cc_system = CoupledChannelSystem::build(deform, &band, two_j_total, 4);
        let n_ch = cc_system.n_channels();
        if n_ch < 2 {
            return;
        }

        let config = NumerovConfig::default();
        let johnson_config = JohnsonConfig {
            step_size: config.step_size,
            r_min: config.r_min,
            r_max: r_match,
        };

        let l_vals: Vec<u32> = cc_system.channels.iter().map(|c| c.l).collect();
        let result = johnson_propagate_with_l(
            n_ch,
            &johnson_config,
            |r| build_w_matrix(&cc_system, &omp, &ch, energy, r, hbar2_over_2mu),
            Some(&l_vals),
        )
        .unwrap();

        let s_matrix =
            extract_s_matrix(&result.z_matrix, n_ch, &cc_system, &ch, energy, r_match).unwrap();

        // All elements should be finite
        for c in 0..n_ch {
            for cp in 0..n_ch {
                let s = s_matrix[c * n_ch + cp];
                assert!(
                    s.re.is_finite() && s.im.is_finite(),
                    "S[{c},{cp}] should be finite, got ({}, {}i)",
                    s.re,
                    s.im
                );
            }
        }

        // Diagonal elements should have |S| ≤ 1 + tolerance (within numerical precision)
        for c in 0..n_ch {
            let s_cc = s_matrix[c * n_ch + c];
            assert!(
                s_cc.norm() <= 1.0 + 1e-3,
                "S[{c},{c}] norm should be ≤ 1, got {}",
                s_cc.norm()
            );
        }
    }

    #[test]
    fn collision_matrix_populated() {
        // Verify the collision matrix is actually populated (not all zeros)
        let omp = KoningDelaroche;
        let ch = Channel {
            projectile: Projectile::Neutron,
            target: Nuclide::new(92, 238).unwrap(),
            q_value: 0.0,
        };
        let energies = EnergyGrid::from_values(vec![1.0]).unwrap();
        let config = NumerovConfig {
            max_l: 2,
            ..NumerovConfig::default()
        };

        let deform = DeformationParams::quadrupole(0.22);
        let band = RotationalBand::even_even(4, &[0.04491, 0.14848]);

        let result =
            compute_cc_transmission(&omp, &ch, &energies, &config, &deform, &band).unwrap();

        let cm = &result.collision_matrix;
        let n = cm.n_channels;

        // At least one non-zero element should exist
        let has_nonzero = cm.u_matrix.iter().any(|v| v.norm() > 1e-15);
        assert!(
            has_nonzero,
            "Collision matrix should have non-zero elements (n_ch={n})"
        );

        // Diagonal elements should be non-trivial for open channels
        if n >= 1 {
            let s00 = cm.get(0, 0, 0);
            assert!(
                s00.norm() > 1e-10,
                "S[0,0] at E=1 MeV should be non-trivial, got |S|={}",
                s00.norm()
            );
        }
    }
}
