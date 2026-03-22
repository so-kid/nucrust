use crate::EnergyGrid;

/// Transmission coefficient table T_{l,j}(E).
///
/// SoA layout optimized for GPU batch transfer.
/// Index: `data[l * n_j * n_e + j_index * n_e + e_index]`
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
    /// Number of j-substates per l: 2 (l-1/2 and l+1/2), except l=0 has only j=1/2.
    #[inline]
    pub fn n_j(l: u32) -> usize {
        if l == 0 {
            1
        } else {
            2
        }
    }

    /// Get T_{l,j}(E) for a specific (l, j_index, energy_index).
    #[inline]
    pub fn get(&self, l: u32, j_index: usize, e_index: usize) -> f64 {
        let n_e = self.energy.len();
        let offset = self.l_offset(l) + j_index * n_e + e_index;
        self.data[offset]
    }

    /// Compute the flat-array offset for the start of partial wave l.
    fn l_offset(&self, l: u32) -> usize {
        let n_e = self.energy.len();
        let mut offset = 0;
        for ll in 0..l {
            offset += Self::n_j(ll) * n_e;
        }
        offset
    }
}
