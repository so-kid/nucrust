//! Hybrid Coulomb function strategy (T-3A.12, §10.6).
//!
//! Combines table lookup (fast, limited range) with on-the-fly computation
//! (slower, arbitrary range). Sorts tasks by computation branch to minimize
//! warp divergence.

/// Strategy selection for Coulomb function computation on GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoulombStrategy {
    /// Use pre-computed table (fastest, within table range).
    Table,
    /// Use on-the-fly __device__ computation (slower, arbitrary range).
    OnTheFly,
}

/// Classify each (eta, rho, l) point into table or on-the-fly.
///
/// Points within the table's (eta, rho, l) range use table lookup;
/// points outside use on-the-fly computation.
pub fn classify_coulomb_tasks(
    eta: &[f64],
    rho: &[f64],
    l: &[i32],
    table_eta_max: f64,
    table_rho_max: f64,
    table_l_max: i32,
) -> Vec<CoulombStrategy> {
    eta.iter()
        .zip(rho.iter())
        .zip(l.iter())
        .map(|((&e, &r), &li)| {
            if e >= 0.0 && e <= table_eta_max && r >= 0.5 && r <= table_rho_max && li <= table_l_max
            {
                CoulombStrategy::Table
            } else {
                CoulombStrategy::OnTheFly
            }
        })
        .collect()
}

/// Sort tasks by strategy for warp-coherent execution.
///
/// Returns (sorted_indices, split_point) where:
/// - sorted_indices[0..split_point] → Table strategy
/// - sorted_indices[split_point..] → OnTheFly strategy
pub fn sort_by_strategy(strategies: &[CoulombStrategy]) -> (Vec<usize>, usize) {
    let mut table_indices = Vec::new();
    let mut onthefly_indices = Vec::new();

    for (i, &s) in strategies.iter().enumerate() {
        match s {
            CoulombStrategy::Table => table_indices.push(i),
            CoulombStrategy::OnTheFly => onthefly_indices.push(i),
        }
    }

    let split = table_indices.len();
    let mut sorted = table_indices;
    sorted.extend(onthefly_indices);
    (sorted, split)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_within_table_range() {
        let eta = vec![0.0, 5.0, 25.0];
        let rho = vec![5.0, 15.0, 5.0];
        let l = vec![0, 5, 0];

        let strategies = classify_coulomb_tasks(&eta, &rho, &l, 20.0, 30.0, 20);

        assert_eq!(strategies[0], CoulombStrategy::Table);
        assert_eq!(strategies[1], CoulombStrategy::Table);
        assert_eq!(strategies[2], CoulombStrategy::OnTheFly); // eta=25 > 20
    }

    #[test]
    fn sort_separates_strategies() {
        let strategies = vec![
            CoulombStrategy::OnTheFly,
            CoulombStrategy::Table,
            CoulombStrategy::Table,
            CoulombStrategy::OnTheFly,
        ];

        let (sorted, split) = sort_by_strategy(&strategies);

        assert_eq!(split, 2); // 2 Table entries
        assert_eq!(sorted.len(), 4);
        // First 2 should be table indices
        assert_eq!(sorted[0], 1);
        assert_eq!(sorted[1], 2);
        // Last 2 should be on-the-fly indices
        assert_eq!(sorted[2], 0);
        assert_eq!(sorted[3], 3);
    }
}
