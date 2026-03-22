//! SIMD-accelerated batch Coulomb wave function computation.
//!
//! Uses `wide::f64x4` to evaluate 4 (eta, rho) pairs simultaneously.
//! Parameters are binned by computation region (oscillatory/forbidden/small-rho)
//! to minimize SIMD lane divergence.
//!
//! When the `simd` feature is enabled, CF1 and CF2 are evaluated natively
//! with f64x4, processing 4 independent (eta, rho) pairs in parallel.

#[cfg(feature = "simd")]
use wide::f64x4;

use crate::coulomb::{coulomb_wave, CoulombResult};
use crate::error::SpecialError;

/// Maximum CF iterations.
#[cfg(feature = "simd")]
const MAX_ITER: u32 = 20_000;
/// Convergence threshold.
#[cfg(feature = "simd")]
const CF_EPS: f64 = 1e-15;

/// Computation region for binning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComputeRegion {
    /// rho > 0.5 and oscillatory (rho > 0.9 * rho_tp)
    Oscillatory,
    /// rho > 0.5 and forbidden (rho < 0.9 * rho_tp)
    Forbidden,
    /// rho <= 0.5
    SmallRho,
}

fn classify_region(eta: f64, rho: f64, l: u32) -> ComputeRegion {
    if rho <= 0.5 {
        return ComputeRegion::SmallRho;
    }
    let lf = l as f64;
    let rho_tp = eta + (eta * eta + lf * (lf + 1.0)).sqrt();
    if rho < 0.9 * rho_tp {
        ComputeRegion::Forbidden
    } else {
        ComputeRegion::Oscillatory
    }
}

/// SIMD-accelerated batch computation of Coulomb wave functions.
///
/// Bins input parameters by computation region, then processes each bin.
/// Within a bin, groups of 4 are processed with f64x4 SIMD for CF evaluation.
///
/// When the `simd` feature is disabled, this is equivalent to `coulomb_wave_batch`.
pub fn coulomb_wave_batch_simd(
    eta: &[f64],
    rho: &[f64],
    l_min: u32,
    n_l: u32,
) -> Result<Vec<CoulombResult>, SpecialError> {
    assert_eq!(eta.len(), rho.len(), "eta and rho must have same length");
    let n = eta.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Build index + region classification
    let mut indexed: Vec<(usize, ComputeRegion)> = (0..n)
        .map(|i| (i, classify_region(eta[i], rho[i], l_min)))
        .collect();

    // Sort by region for better SIMD utilization
    indexed.sort_by_key(|&(_, region)| match region {
        ComputeRegion::Oscillatory => 0,
        ComputeRegion::Forbidden => 1,
        ComputeRegion::SmallRho => 2,
    });

    // Allocate output
    let mut results: Vec<Option<CoulombResult>> = (0..n).map(|_| None).collect();

    // Process in groups
    #[cfg(feature = "simd")]
    {
        process_simd_groups(&indexed, eta, rho, l_min, n_l, &mut results)?;
    }

    #[cfg(not(feature = "simd"))]
    {
        process_scalar(&indexed, eta, rho, l_min, n_l, &mut results)?;
    }

    // Unwrap results (all should be Some)
    Ok(results.into_iter().map(|r| r.unwrap()).collect())
}

/// Scalar fallback: process all points sequentially.
#[cfg_attr(feature = "simd", allow(dead_code))]
fn process_scalar(
    indexed: &[(usize, ComputeRegion)],
    eta: &[f64],
    rho: &[f64],
    l_min: u32,
    n_l: u32,
    results: &mut [Option<CoulombResult>],
) -> Result<(), SpecialError> {
    for &(orig_idx, _) in indexed {
        let result = coulomb_wave(eta[orig_idx], rho[orig_idx], l_min, n_l)?;
        results[orig_idx] = Some(result);
    }
    Ok(())
}

// =============================================================================
// SIMD-native CF evaluation (feature = "simd")
// =============================================================================

#[cfg(feature = "simd")]
mod simd_cf {
    use crate::error::SpecialError;
    use wide::{f64x4, CmpLe, CmpLt};

    use super::{CF_EPS, MAX_ITER};

    const SMALL: f64 = 1e-50;

    /// f64x4 helper: element-wise reciprocal 1/x.
    #[inline]
    fn recip_x4(x: f64x4) -> f64x4 {
        f64x4::ONE / x
    }

    /// f64x4 helper: element-wise absolute value.
    #[inline]
    fn abs_x4(x: f64x4) -> f64x4 {
        x.abs()
    }

    /// f64x4 helper: clamp small absolute values to SMALL.
    #[inline]
    fn clamp_small(x: f64x4) -> f64x4 {
        let small = f64x4::splat(SMALL);
        // If |x| < SMALL, replace with SMALL
        let too_small = abs_x4(x).cmp_lt(small);
        too_small.blend(small, x)
    }

    /// SIMD CF1: evaluate 4 CF1 values simultaneously.
    ///
    /// CF1 gives f_l = F'_l / F_l for 4 different (l, eta, rho) combinations.
    /// All lanes use the same l but different eta and rho.
    pub fn cf1_x4(l: f64, eta4: f64x4, rho4: f64x4) -> Result<[f64; 4], SpecialError> {
        let l1 = f64x4::splat(l + 1.0);
        let rho_inv = recip_x4(rho4);

        // b0 = (l+1)/rho + eta/(l+1)
        let mut h = l1 * rho_inv + eta4 / l1;
        h = clamp_small(h);

        let mut d = f64x4::ZERO;
        let mut c = h;

        let eps = f64x4::splat(CF_EPS);
        let one = f64x4::ONE;

        for n in 1..=MAX_ITER {
            let nf = f64x4::splat(n as f64);
            let ln = f64x4::splat(l) + nf; // l + n

            // a_n = -(1 + (eta/ln)^2)
            let eta_over_ln = eta4 / ln;
            let an = -(one + eta_over_ln * eta_over_ln);

            // b_n = ln/rho + eta/ln + (ln+1)/rho + eta/(ln+1)
            let ln1 = ln + one;
            let bn = ln * rho_inv + eta4 / ln + ln1 * rho_inv + eta4 / ln1;

            d = clamp_small(bn + an * d);
            c = clamp_small(bn + an / c);
            d = recip_x4(d);
            let delta = c * d;
            h *= delta;

            // Check convergence: all lanes must satisfy |delta - 1| < eps
            let converged = abs_x4(delta - one).cmp_lt(eps);
            if converged.all() {
                return Ok(h.to_array());
            }
        }

        // Return whatever we have - some lanes may have converged
        Ok(h.to_array())
    }

    /// SIMD CF2: evaluate 4 CF2 values simultaneously using Steed algorithm.
    ///
    /// Returns ([p0..p3], [q0..q3]) where p + iq = H^{+'}_{l} / H^+_l.
    pub fn cf2_x4(l: f64, eta4: f64x4, rho4: f64x4) -> Result<([f64; 4], [f64; 4]), SpecialError> {
        let one = f64x4::ONE;
        let two = f64x4::splat(2.0);
        let eps = f64x4::splat(CF_EPS);

        let lf = f64x4::splat(l);
        let wi = two * eta4;
        let x_inv = recip_x4(rho4);
        let e2mm1 = eta4 * eta4 + lf * (lf + one);

        let mut ar = -e2mm1;
        let mut ai = eta4;
        let br = two * (rho4 - eta4);
        let mut bi = two;

        // d = 1/(br + i*bi) = (br - i*bi)/(br^2 + bi^2)
        let denom = br * br + bi * bi;
        let denom_inv = recip_x4(denom);
        let mut dr = br * denom_inv;
        let mut di = -bi * denom_inv;

        // First correction: dp + i*dq = (i/rho) * (ar + i*ai) / (br + i*bi)
        let mut dp = -x_inv * (ar * di + ai * dr);
        let mut dq = x_inv * (ar * dr - ai * di);

        let mut p = f64x4::ZERO;
        let mut q = one - eta4 * x_inv;

        let mut pk = f64x4::ZERO;

        for _ in 1..=MAX_ITER {
            p += dp;
            q += dq;
            pk += two;
            ar += pk;
            ai += wi;
            bi += two;

            let d_re = ar * dr - ai * di + br;
            let d_im = ai * dr + ar * di + bi;
            let c_inv = recip_x4(d_re * d_re + d_im * d_im);
            dr = c_inv * d_re;
            di = -c_inv * d_im;

            let a = br * dr - bi * di - one;
            let b = bi * dr + br * di;
            let new_dp = dp * a - dq * b;
            dq = dp * b + dq * a;
            dp = new_dp;

            // Convergence: |dp| + |dq| <= (|p| + |q|) * eps
            let lhs = abs_x4(dp) + abs_x4(dq);
            let rhs = (abs_x4(p) + abs_x4(q)) * eps;
            if lhs.cmp_le(rhs).all() {
                return Ok((p.to_array(), q.to_array()));
            }
        }

        // Return whatever we have
        Ok((p.to_array(), q.to_array()))
    }
}

/// SIMD processing: evaluate 4 points at a time within the same region.
///
/// Uses f64x4-native CF1 and CF2 evaluators for the oscillatory and forbidden
/// regions. Falls back to scalar for the small-rho region (power series).
#[cfg(feature = "simd")]
fn process_simd_groups(
    indexed: &[(usize, ComputeRegion)],
    eta: &[f64],
    rho: &[f64],
    l_min: u32,
    n_l: u32,
    results: &mut [Option<CoulombResult>],
) -> Result<(), SpecialError> {
    let mut i = 0;
    while i < indexed.len() {
        let current_region = indexed[i].1;

        // Find the extent of this region bin
        let bin_end = indexed[i..]
            .iter()
            .position(|&(_, r)| r != current_region)
            .map_or(indexed.len(), |pos| i + pos);

        let bin = &indexed[i..bin_end];

        // For SmallRho region, use scalar (power series doesn't benefit from SIMD CF)
        if current_region == ComputeRegion::SmallRho {
            for &(idx, _) in bin {
                let result = coulomb_wave(eta[idx], rho[idx], l_min, n_l)?;
                results[idx] = Some(result);
            }
            i = bin_end;
            continue;
        }

        // Process groups of 4 with SIMD CF
        let mut j = 0;
        while j + 4 <= bin.len() {
            let idxs = [bin[j].0, bin[j + 1].0, bin[j + 2].0, bin[j + 3].0];

            let eta4 = f64x4::new([eta[idxs[0]], eta[idxs[1]], eta[idxs[2]], eta[idxs[3]]]);
            let rho4 = f64x4::new([rho[idxs[0]], rho[idxs[1]], rho[idxs[2]], rho[idxs[3]]]);

            // Compute CF1 and CF2 for l=0 (Steed method starting point) using SIMD
            let l0 = l_min as f64;
            let f_ratio = simd_cf::cf1_x4(l0, eta4, rho4)?;
            let (p_arr, q_arr) = simd_cf::cf2_x4(l0, eta4, rho4)?;

            // Now assemble results per lane using scalar post-processing
            // (recurrences and normalization are inherently serial per-point)
            for (lane, &idx) in idxs.iter().enumerate() {
                let result = assemble_from_cf(
                    eta[idx],
                    rho[idx],
                    l_min,
                    n_l,
                    f_ratio[lane],
                    p_arr[lane],
                    q_arr[lane],
                );
                match result {
                    Ok(r) => results[idx] = Some(r),
                    Err(_) => {
                        // Fallback to full scalar computation on SIMD failure
                        let r = coulomb_wave(eta[idx], rho[idx], l_min, n_l)?;
                        results[idx] = Some(r);
                    }
                }
            }

            j += 4;
        }

        // Handle remainder (< 4 points)
        while j < bin.len() {
            let idx = bin[j].0;
            let result = coulomb_wave(eta[idx], rho[idx], l_min, n_l)?;
            results[idx] = Some(result);
            j += 1;
        }

        i = bin_end;
    }

    Ok(())
}

/// Assemble a CoulombResult from pre-computed CF1 and CF2 values.
///
/// This handles the Steed method combination and l-recurrence using
/// scalar arithmetic, since these are inherently serial per-point.
#[cfg(feature = "simd")]
fn assemble_from_cf(
    eta: f64,
    rho: f64,
    l_min: u32,
    n_l: u32,
    f_ratio: f64,
    p: f64,
    q: f64,
) -> Result<CoulombResult, SpecialError> {
    use crate::gamma::coulomb_phase_shift;

    if q.abs() < 1e-300 {
        return Err(SpecialError::NumericalOverflow {
            context: "SIMD Steed: q ~ 0",
        });
    }

    let l0 = l_min as f64;

    // F^2 = q / ((f-p)^2 + q^2)
    let fmp = f_ratio - p;
    let denom = fmp * fmp + q * q;
    let f_sq = (q / denom).abs();

    // Sign from asymptotic phase
    let sign = f_sign_scalar(l0, eta, rho);
    let f_l0 = sign * f_sq.sqrt();
    let fp_l0 = f_ratio * f_l0;
    let gamm = fmp / q;
    let g_l0 = gamm * f_l0;
    let gp_l0 = p * g_l0 - q * f_l0;

    // Build result with recurrence if n_l > 1
    let mut f_vals = vec![0.0; n_l as usize];
    let mut g_vals = vec![0.0; n_l as usize];
    let mut fp_vals = vec![0.0; n_l as usize];
    let mut gp_vals = vec![0.0; n_l as usize];
    let mut sigma_vals = vec![0.0; n_l as usize];

    f_vals[0] = f_l0;
    g_vals[0] = g_l0;
    fp_vals[0] = fp_l0;
    gp_vals[0] = gp_l0;
    sigma_vals[0] = coulomb_phase_shift(l_min, eta);

    // Upward recurrence for G, downward for F
    for k in 1..n_l as usize {
        let l = l_min as f64 + k as f64;
        let r = (1.0 + (eta / l).powi(2)).sqrt();
        let s = l / rho + eta / l;

        // G upward: G_{l} = (S_l * G_{l-1} - G'_{l-1}) / R_l
        g_vals[k] = (s * g_vals[k - 1] - gp_vals[k - 1]) / r;
        gp_vals[k] = s * g_vals[k] - r * g_vals[k - 1];

        sigma_vals[k] = coulomb_phase_shift(l_min + k as u32, eta);
    }

    // F from Wronskian: F_l = 1/(G_l * f_ratio_l - G'_l) if we have f_ratio at each l
    // Or use downward recurrence from F at l_min
    // For simplicity when n_l > 1, use Wronskian: F_l * G'_l - F'_l * G_l = 1
    for k in 1..n_l as usize {
        let l = l_min as f64 + k as f64;
        let r = (1.0 + (eta / l).powi(2)).sqrt();
        let s = l / rho + eta / l;

        // F upward from Wronskian
        f_vals[k] = (s * f_vals[k - 1] - fp_vals[k - 1]) / r;
        fp_vals[k] = s * f_vals[k] - r * f_vals[k - 1];
    }

    Ok(CoulombResult {
        f: f_vals,
        g: g_vals,
        fp: fp_vals,
        gp: gp_vals,
        sigma: sigma_vals,
        exponent: 0.0,
    })
}

/// Determine sign of F_l at l=0 using asymptotic phase.
#[cfg(feature = "simd")]
fn f_sign_scalar(l: f64, eta: f64, rho: f64) -> f64 {
    use crate::gamma::coulomb_phase_shift;

    let rho_tp = eta + (eta * eta + l * (l + 1.0)).sqrt();
    if rho < rho_tp {
        return 1.0;
    }
    let sigma = coulomb_phase_shift(l as u32, eta);
    let theta = rho - eta * (2.0 * rho).ln() - l * std::f64::consts::FRAC_PI_2 + sigma;
    theta.sin().signum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_simd_matches_scalar() {
        let etas = vec![0.0, 1.0, 2.0, 0.5, 5.0, 0.0, 1.0, 3.0];
        let rhos = vec![3.0, 5.0, 7.0, 0.3, 2.0, 10.0, 1.0, 8.0];

        let simd_results = coulomb_wave_batch_simd(&etas, &rhos, 0, 1).unwrap();

        for (i, (&e, &r)) in etas.iter().zip(rhos.iter()).enumerate() {
            let scalar = coulomb_wave(e, r, 0, 1).unwrap();
            let rel_err_f = if scalar.f[0].abs() > 1e-300 {
                ((simd_results[i].f[0] - scalar.f[0]) / scalar.f[0]).abs()
            } else {
                (simd_results[i].f[0] - scalar.f[0]).abs()
            };
            assert!(
                rel_err_f < 1e-14,
                "mismatch at i={} (eta={}, rho={}): simd={}, scalar={}",
                i,
                e,
                r,
                simd_results[i].f[0],
                scalar.f[0]
            );
        }
    }

    #[test]
    fn batch_simd_empty() {
        let results = coulomb_wave_batch_simd(&[], &[], 0, 1).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn batch_simd_single() {
        let results = coulomb_wave_batch_simd(&[1.0], &[5.0], 0, 1).unwrap();
        assert_eq!(results.len(), 1);
        let scalar = coulomb_wave(1.0, 5.0, 0, 1).unwrap();
        assert!((results[0].f[0] - scalar.f[0]).abs() < 1e-14);
    }

    #[test]
    fn batch_simd_region_binning() {
        // Mix of all three regions
        let etas = vec![0.0, 10.0, 0.0, 1.0]; // oscillatory, forbidden, small_rho, oscillatory
        let rhos = vec![5.0, 1.0, 0.1, 3.0];

        let results = coulomb_wave_batch_simd(&etas, &rhos, 0, 1).unwrap();
        assert_eq!(results.len(), 4);

        // All should produce finite results
        for (i, r) in results.iter().enumerate() {
            assert!(r.f[0].is_finite(), "F[{}] not finite", i);
        }
    }

    #[cfg(feature = "simd")]
    mod simd_tests {
        use super::super::simd_cf;
        use wide::f64x4;

        #[test]
        fn cf1_x4_matches_scalar() {
            // Test CF1 with 4 different eta values at same rho
            let eta4 = f64x4::new([0.0, 1.0, 2.0, 5.0]);
            let rho4 = f64x4::new([5.0, 5.0, 5.0, 5.0]);

            let results = simd_cf::cf1_x4(0.0, eta4, rho4).unwrap();

            // Compare with scalar CF1 via coulomb_wave
            for (i, (&eta, &rho)) in [0.0, 1.0, 2.0, 5.0]
                .iter()
                .zip([5.0, 5.0, 5.0, 5.0].iter())
                .enumerate()
            {
                let scalar = crate::coulomb::coulomb_wave(eta, rho, 0, 1).unwrap();
                // CF1 gives f_ratio = F'/F
                let scalar_ratio = scalar.fp[0] / scalar.f[0];
                let rel_err = if scalar_ratio.abs() > 1e-10 {
                    ((results[i] - scalar_ratio) / scalar_ratio).abs()
                } else {
                    (results[i] - scalar_ratio).abs()
                };
                assert!(
                    rel_err < 1e-10,
                    "CF1 mismatch at lane {}: simd={}, scalar={}, rel_err={}",
                    i,
                    results[i],
                    scalar_ratio,
                    rel_err
                );
            }
        }

        #[test]
        fn cf2_x4_produces_finite() {
            let eta4 = f64x4::new([0.0, 1.0, 2.0, 3.0]);
            let rho4 = f64x4::new([5.0, 5.0, 7.0, 10.0]);

            let (p, q) = simd_cf::cf2_x4(0.0, eta4, rho4).unwrap();
            for i in 0..4 {
                assert!(p[i].is_finite(), "p[{}] not finite", i);
                assert!(q[i].is_finite(), "q[{}] not finite", i);
                assert!(q[i].abs() > 1e-30, "q[{}] too small: {}", i, q[i]);
            }
        }

        #[test]
        fn batch_simd_oscillatory_group_of_4() {
            // 4 oscillatory points (same region) should use SIMD path
            let etas = vec![0.0, 0.5, 1.0, 1.5];
            let rhos = vec![5.0, 6.0, 7.0, 8.0];

            let simd_results =
                crate::simd_batch::coulomb_wave_batch_simd(&etas, &rhos, 0, 1).unwrap();

            for (i, (&e, &r)) in etas.iter().zip(rhos.iter()).enumerate() {
                let scalar = crate::coulomb::coulomb_wave(e, r, 0, 1).unwrap();
                let rel_err_f = if scalar.f[0].abs() > 1e-300 {
                    ((simd_results[i].f[0] - scalar.f[0]) / scalar.f[0]).abs()
                } else {
                    (simd_results[i].f[0] - scalar.f[0]).abs()
                };
                assert!(
                    rel_err_f < 1e-10,
                    "mismatch at i={}: simd={}, scalar={}, err={}",
                    i,
                    simd_results[i].f[0],
                    scalar.f[0],
                    rel_err_f
                );
            }
        }
    }
}
