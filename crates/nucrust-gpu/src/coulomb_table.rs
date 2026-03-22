//! GPU Coulomb function lookup table.
//!
//! Phase 1 strategy (§10.6): pre-compute F, G on CPU, upload as texture-like
//! table to GPU global memory, read via `__ldg()` in kernels.
//!
//! Table dimensions: n_eta × n_rho × n_l → ~24MB for typical grids.

use cudarc::driver::{CudaSlice, CudaStream};
use nucrust_core::CoreError;
use nucrust_special::coulomb_wave;
use std::sync::Arc;

/// Pre-computed Coulomb function table on GPU.
pub struct GpuCoulombTable {
    /// F_l(eta, rho) values [n_eta * n_rho * n_l]
    pub d_f: CudaSlice<f64>,
    /// G_l(eta, rho) values [n_eta * n_rho * n_l]
    pub d_g: CudaSlice<f64>,
    /// eta grid [n_eta]
    pub d_eta: CudaSlice<f64>,
    /// rho grid [n_rho]
    pub d_rho: CudaSlice<f64>,
    /// Number of eta grid points.
    pub n_eta: usize,
    /// Number of rho grid points.
    pub n_rho: usize,
    /// Number of l values (l_max + 1).
    pub n_l: usize,
}

/// Configuration for Coulomb table generation.
pub struct CoulombTableConfig {
    /// eta range: [eta_min, eta_max]
    pub eta_min: f64,
    pub eta_max: f64,
    pub n_eta: usize,
    /// rho range: [rho_min, rho_max]
    pub rho_min: f64,
    pub rho_max: f64,
    pub n_rho: usize,
    /// Maximum l
    pub l_max: u32,
}

impl Default for CoulombTableConfig {
    fn default() -> Self {
        Self {
            eta_min: 0.0,
            eta_max: 20.0,
            n_eta: 50,
            rho_min: 0.5,
            rho_max: 30.0,
            n_rho: 100,
            l_max: 20,
        }
    }
}

/// Generate Coulomb function table on CPU and upload to GPU.
pub fn build_coulomb_table(
    stream: &Arc<CudaStream>,
    config: &CoulombTableConfig,
) -> Result<GpuCoulombTable, CoreError> {
    let n_l = (config.l_max + 1) as usize;
    let total = config.n_eta * config.n_rho * n_l;

    // Build eta and rho grids
    let eta_grid: Vec<f64> = (0..config.n_eta)
        .map(|i| {
            if config.n_eta == 1 {
                config.eta_min
            } else {
                config.eta_min
                    + (config.eta_max - config.eta_min) * i as f64 / (config.n_eta - 1) as f64
            }
        })
        .collect();

    let rho_grid: Vec<f64> = (0..config.n_rho)
        .map(|i| {
            if config.n_rho == 1 {
                config.rho_min
            } else {
                config.rho_min
                    + (config.rho_max - config.rho_min) * i as f64 / (config.n_rho - 1) as f64
            }
        })
        .collect();

    // Compute F and G on CPU
    let mut f_table = vec![0.0f64; total];
    let mut g_table = vec![0.0f64; total];

    for (ie, &eta) in eta_grid.iter().enumerate() {
        for (ir, &rho) in rho_grid.iter().enumerate() {
            if rho < 0.01 {
                continue; // Skip very small rho
            }
            match coulomb_wave(eta, rho, 0, config.l_max + 1) {
                Ok(result) => {
                    for l in 0..n_l {
                        let idx = ie * config.n_rho * n_l + ir * n_l + l;
                        f_table[idx] = result.f[l];
                        g_table[idx] = result.g[l];
                    }
                }
                Err(_) => {
                    // Leave as zeros for failed points
                }
            }
        }
    }

    // Upload to GPU
    let mut d_f = stream
        .alloc_zeros::<f64>(total)
        .map_err(|_| CoreError::NumericalOverflow {
            context: "GPU Coulomb table alloc F",
        })?;
    let mut d_g = stream
        .alloc_zeros::<f64>(total)
        .map_err(|_| CoreError::NumericalOverflow {
            context: "GPU Coulomb table alloc G",
        })?;
    let mut d_eta =
        stream
            .alloc_zeros::<f64>(config.n_eta)
            .map_err(|_| CoreError::NumericalOverflow {
                context: "GPU Coulomb table alloc eta",
            })?;
    let mut d_rho =
        stream
            .alloc_zeros::<f64>(config.n_rho)
            .map_err(|_| CoreError::NumericalOverflow {
                context: "GPU Coulomb table alloc rho",
            })?;

    stream
        .memcpy_htod(&f_table, &mut d_f)
        .map_err(|_| CoreError::NumericalOverflow {
            context: "GPU Coulomb table copy F",
        })?;
    stream
        .memcpy_htod(&g_table, &mut d_g)
        .map_err(|_| CoreError::NumericalOverflow {
            context: "GPU Coulomb table copy G",
        })?;
    stream
        .memcpy_htod(&eta_grid, &mut d_eta)
        .map_err(|_| CoreError::NumericalOverflow {
            context: "GPU Coulomb table copy eta",
        })?;
    stream
        .memcpy_htod(&rho_grid, &mut d_rho)
        .map_err(|_| CoreError::NumericalOverflow {
            context: "GPU Coulomb table copy rho",
        })?;

    let table_bytes = total * 2 * std::mem::size_of::<f64>();
    eprintln!(
        "Coulomb table: {}×{}×{} = {} points, {:.1} MB",
        config.n_eta,
        config.n_rho,
        n_l,
        total,
        table_bytes as f64 / 1e6
    );

    Ok(GpuCoulombTable {
        d_f,
        d_g,
        d_eta,
        d_rho,
        n_eta: config.n_eta,
        n_rho: config.n_rho,
        n_l,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    #[test]
    fn build_small_table() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        let config = CoulombTableConfig {
            eta_min: 0.0,
            eta_max: 5.0,
            n_eta: 10,
            rho_min: 0.5,
            rho_max: 10.0,
            n_rho: 20,
            l_max: 5,
        };

        let table = build_coulomb_table(&stream, &config).unwrap();
        assert_eq!(table.n_eta, 10);
        assert_eq!(table.n_rho, 20);
        assert_eq!(table.n_l, 6);
    }
}
