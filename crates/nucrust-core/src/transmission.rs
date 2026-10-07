use crate::EnergyGrid;

/// Transmission coefficient table T_{l,j}(E).
///
/// SoA layout optimized for GPU batch transfer. Every partial wave occupies
/// two j-slots (including l = 0, whose unphysical j = -1/2 slot is 0):
/// `data[l * 2 * n_e + j_index * n_e + e_index]`
/// where j_index: 0 = l-1/2, 1 = l+1/2.
#[derive(Debug, Clone)]
pub struct TransmissionCoeffs {
    /// Energy grid for the tabulated coefficients.
    pub energy: EnergyGrid,
    /// Maximum orbital angular momentum included.
    pub l_max: u32,
    /// Flat array: `[l0_j0_e0, l0_j0_e1, ..., l0_j1_e0, ..., l1_j0_e0, ...]`
    pub data: Vec<f64>,
}

impl TransmissionCoeffs {
    /// Number of j-slots stored per partial wave l (always 2: j = l-1/2, l+1/2).
    pub const J_SLOTS: usize = 2;

    /// Number of physical j-substates per l: 2 (l-1/2 and l+1/2), except l=0 has only j=1/2.
    ///
    /// Note: the storage layout always reserves [`Self::J_SLOTS`] slots per l.
    #[inline]
    pub fn n_j(l: u32) -> usize {
        if l == 0 {
            1
        } else {
            2
        }
    }

    /// Get T_{l,j}(E) for a specific (l, j_index, energy_index).
    ///
    /// j_index: 0 = l-1/2, 1 = l+1/2. For l = 0 the s-wave value is at
    /// j_index = 1 (j = 1/2); j_index = 0 returns 0.
    #[inline]
    pub fn get(&self, l: u32, j_index: usize, e_index: usize) -> f64 {
        let n_e = self.energy.len();
        let offset = l as usize * Self::J_SLOTS * n_e + j_index * n_e + e_index;
        self.data[offset]
    }

    /// Spin-averaged T_l(E) = Σ_j (2j+1) T_{lj} / [(2s+1)(2l+1)] for a spin-1/2 projectile.
    pub fn get_l_averaged(&self, l: u32, e_index: usize) -> f64 {
        let two_l = 2.0 * l as f64;
        let t_minus = self.get(l, 0, e_index);
        let t_plus = self.get(l, 1, e_index);
        (two_l * t_minus + (two_l + 2.0) * t_plus) / (2.0 * (two_l + 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_matches_producer_layout() {
        // Layout written by nucrust-optical / nucrust-gpu: 2 slots per l, incl. l = 0.
        let energy = EnergyGrid::from_values(vec![1.0, 2.0]).unwrap();
        let tc = TransmissionCoeffs {
            energy,
            l_max: 1,
            // l=0: j=-1/2 (0, 0), j=1/2 (0.9, 0.8); l=1: j=1/2 (0.3, 0.4), j=3/2 (0.5, 0.6)
            data: vec![0.0, 0.0, 0.9, 0.8, 0.3, 0.4, 0.5, 0.6],
        };
        assert_eq!(tc.get(0, 1, 0), 0.9);
        assert_eq!(tc.get(0, 1, 1), 0.8);
        assert_eq!(tc.get(1, 0, 1), 0.4);
        assert_eq!(tc.get(1, 1, 0), 0.5);
        assert!((tc.get_l_averaged(0, 0) - 0.9).abs() < 1e-15);
        // (2*0.3 + 4*0.5) / 6
        assert!((tc.get_l_averaged(1, 0) - 2.6 / 6.0).abs() < 1e-15);
    }
}
