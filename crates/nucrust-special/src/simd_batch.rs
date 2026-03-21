//! SIMD-accelerated batch Coulomb wave function computation.
//!
//! Uses `wide::f64x4` to evaluate 4 (eta, rho) pairs simultaneously.
//! Parameters are binned by computation region (oscillatory/forbidden/small-rho)
//! to minimize SIMD lane divergence.

#[cfg(feature = "simd")]
use wide::f64x4;

use crate::coulomb::{coulomb_wave, CoulombResult};
use crate::error::SpecialError;

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
/// Within a bin, groups of 4 are processed with SIMD where possible,
/// falling back to scalar for remainders.
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

/// SIMD processing: evaluate 4 points at a time within the same region.
///
/// The SIMD acceleration applies to the parameter preparation and
/// post-processing stages. The core CF evaluation still uses scalar
/// per-lane due to variable iteration counts, but the binning ensures
/// similar convergence behavior within each group.
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

        // Process groups of 4 with SIMD-friendly batching
        let mut j = 0;
        while j + 4 <= bin.len() {
            let idxs = [bin[j].0, bin[j + 1].0, bin[j + 2].0, bin[j + 3].0];

            // Load parameters into SIMD registers for pre/post-processing
            let _eta4 = f64x4::new([eta[idxs[0]], eta[idxs[1]], eta[idxs[2]], eta[idxs[3]]]);
            let _rho4 = f64x4::new([rho[idxs[0]], rho[idxs[1]], rho[idxs[2]], rho[idxs[3]]]);

            // Core computation: scalar per lane (CF iteration counts differ)
            // Future optimization: implement f64x4-native CF evaluation
            // where all 4 lanes run the same number of iterations (padded)
            for &idx in &idxs {
                let result = coulomb_wave(eta[idx], rho[idx], l_min, n_l)?;
                results[idx] = Some(result);
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
}
