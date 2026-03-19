use num_complex::Complex64;

/// Collision matrix U_{cc'}(E).
///
/// SoA layout: `u_matrix[energy_index * n_ch * n_ch + c * n_ch + c']`.
#[derive(Debug, Clone)]
pub struct CollisionMatrix {
    /// Energy points (MeV).
    pub energies: Vec<f64>,
    /// Number of channels.
    pub n_channels: usize,
    /// U-matrix elements in flat SoA layout.
    pub u_matrix: Vec<Complex64>,
}

impl CollisionMatrix {
    /// Access U_{c,c'}(E_i).
    #[inline]
    pub fn get(&self, energy_index: usize, c: usize, c_prime: usize) -> Complex64 {
        let nc = self.n_channels;
        self.u_matrix[energy_index * nc * nc + c * nc + c_prime]
    }
}
