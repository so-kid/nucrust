//! Warp divergence mitigation via 2-stage binning (T-3A.6, §10.2.3).
//!
//! Strategy: sort tasks by (l, energy_bin) so that threads within the same warp
//! execute similar iteration counts in the Numerov kernel.
//!
//! Stage 1: Sort by l (dominant factor for iteration count difference)
//! Stage 2: Within each l group, sort by energy (similar convergence behavior)

/// Bin tasks for GPU Numerov to minimize warp divergence.
///
/// Reorders (l_values, e_indices) arrays so that adjacent threads
/// have similar computation cost.
///
/// Returns (sorted_l_values, sorted_e_indices, original_indices)
/// where original_indices maps sorted position → original position.
pub fn bin_numerov_tasks(
    l_values: &[i32],
    e_indices: &[i32],
    energies: &[f64],
) -> (Vec<i32>, Vec<i32>, Vec<usize>) {
    let n = l_values.len();
    assert_eq!(n, e_indices.len());

    // Create index array and sort by (l, energy)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let la = l_values[a];
        let lb = l_values[b];
        la.cmp(&lb).then_with(|| {
            let ea = energies[e_indices[a] as usize];
            let eb = energies[e_indices[b] as usize];
            ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let sorted_l: Vec<i32> = indices.iter().map(|&i| l_values[i]).collect();
    let sorted_e: Vec<i32> = indices.iter().map(|&i| e_indices[i]).collect();

    (sorted_l, sorted_e, indices)
}

/// Unsort results back to original task order.
pub fn unsort_results(sorted_results: &[f64], original_indices: &[usize]) -> Vec<f64> {
    let n = sorted_results.len();
    let mut results = vec![0.0; n];
    for (sorted_pos, &orig_pos) in original_indices.iter().enumerate() {
        results[orig_pos] = sorted_results[sorted_pos];
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_sorts_by_l() {
        let l_vals = vec![2, 0, 1, 0, 2, 1];
        let e_idxs = vec![0, 1, 2, 0, 1, 2];
        let energies = vec![1.0, 2.0, 3.0];

        let (sorted_l, _sorted_e, indices) = bin_numerov_tasks(&l_vals, &e_idxs, &energies);

        // All l=0 should come first, then l=1, then l=2
        assert_eq!(sorted_l[0], 0);
        assert_eq!(sorted_l[1], 0);
        assert_eq!(sorted_l[2], 1);
        assert_eq!(sorted_l[3], 1);
        assert_eq!(sorted_l[4], 2);
        assert_eq!(sorted_l[5], 2);

        // Original indices should map back correctly
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn unsort_recovers_original_order() {
        let l_vals = vec![2, 0, 1];
        let e_idxs = vec![0, 0, 0];
        let energies = vec![1.0];

        let (_sorted_l, _sorted_e, indices) = bin_numerov_tasks(&l_vals, &e_idxs, &energies);

        // Simulate sorted results: [result_for_orig_1, result_for_orig_2, result_for_orig_0]
        let sorted_results: Vec<f64> = indices.iter().map(|&i| (i * 10) as f64).collect();
        let original = unsort_results(&sorted_results, &indices);

        assert_eq!(original[0], 0.0); // original index 0 → value 0
        assert_eq!(original[1], 10.0); // original index 1 → value 10
        assert_eq!(original[2], 20.0); // original index 2 → value 20
    }
}
