//! GPU-accelerated MACS integration.

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use nucrust_core::{CoreError, CrossSection};
use std::sync::Arc;

const MACS_KERNEL_SRC: &str = include_str!("../../../kernels/macs_integral.cu");

/// Gauss-Laguerre 8-point nodes and weights.
const GL8_NODES: [f64; 8] = [
    0.170_279_632_305,
    0.903_701_776_799,
    2.251_086_629_866,
    4.266_700_170_288,
    7.045_905_402_393,
    10.758_516_010_181,
    15.740_678_641_928,
    22.863_131_736_889,
];
const GL8_WEIGHTS: [f64; 8] = [
    0.369_188_589_342,
    0.418_786_780_814,
    0.175_794_986_637,
    0.033_343_492_261,
    0.002_794_536_235,
    0.000_090_765_688,
    0.000_000_848_574,
    0.000_000_001_049,
];

/// Compute MACS on GPU for a batch of temperatures.
pub fn gpu_macs_integrate(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    xs: &CrossSection,
    temperatures: &[f64],
) -> Result<Vec<f64>, CoreError> {
    let n_e = xs.energy.len();
    let n_t = temperatures.len();
    let n_quad = GL8_NODES.len();

    let ptx =
        cudarc::nvrtc::compile_ptx(MACS_KERNEL_SRC).map_err(|_| CoreError::InvalidParameter {
            name: "nvrtc",
            value: 0.0,
            reason: "failed to compile MACS kernel",
        })?;

    let module = CudaContext::load_module(ctx, ptx).map_err(|_| CoreError::InvalidParameter {
        name: "load_module",
        value: 0.0,
        reason: "failed to load MACS module",
    })?;

    let func = module
        .load_function("macs_integral")
        .map_err(|_| CoreError::DataNotFound {
            description: "macs_integral function not found".to_string(),
        })?;

    // Upload data
    let mut d_temps = stream.alloc_zeros::<f64>(n_t).map_err(gpu_err)?;
    let mut d_energies = stream.alloc_zeros::<f64>(n_e).map_err(gpu_err)?;
    let mut d_sigmas = stream.alloc_zeros::<f64>(n_e).map_err(gpu_err)?;
    let mut d_nodes = stream.alloc_zeros::<f64>(n_quad).map_err(gpu_err)?;
    let mut d_weights = stream.alloc_zeros::<f64>(n_quad).map_err(gpu_err)?;
    let mut d_macs = stream.alloc_zeros::<f64>(n_t).map_err(gpu_err)?;

    stream
        .memcpy_htod(temperatures, &mut d_temps)
        .map_err(gpu_err)?;
    stream
        .memcpy_htod(xs.energy.as_slice(), &mut d_energies)
        .map_err(gpu_err)?;
    stream
        .memcpy_htod(&xs.sigma_reaction, &mut d_sigmas)
        .map_err(gpu_err)?;
    stream
        .memcpy_htod(&GL8_NODES, &mut d_nodes)
        .map_err(gpu_err)?;
    stream
        .memcpy_htod(&GL8_WEIGHTS, &mut d_weights)
        .map_err(gpu_err)?;

    let k_boltzmann = nucrust_core::units::BOLTZMANN_MEV;

    let block_size = 256u32;
    let grid_size = (n_t as u32 + block_size - 1) / block_size;

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&d_temps)
            .arg(&d_energies)
            .arg(&d_sigmas)
            .arg(&(n_e as i32))
            .arg(&d_nodes)
            .arg(&d_weights)
            .arg(&(n_quad as i32))
            .arg(&k_boltzmann)
            .arg(&mut d_macs)
            .arg(&(n_t as i32))
            .launch(LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|_| CoreError::NumericalOverflow {
                context: "MACS kernel launch",
            })?;
    }

    let mut macs = vec![0.0f64; n_t];
    stream.memcpy_dtoh(&d_macs, &mut macs).map_err(gpu_err)?;
    stream.synchronize().map_err(gpu_err)?;

    Ok(macs)
}

fn gpu_err<E>(_: E) -> CoreError {
    CoreError::NumericalOverflow {
        context: "GPU operation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::EnergyGrid;

    #[test]
    fn gpu_macs_runs() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();

        // 1/v cross section
        let energies = EnergyGrid::logarithmic(0.001, 10.0, 100).unwrap();
        let sigmas: Vec<f64> = energies
            .as_slice()
            .iter()
            .map(|&e| 100.0 / e.sqrt())
            .collect();
        let xs = CrossSection {
            energy: energies,
            sigma_total: sigmas.clone(),
            sigma_elastic: vec![0.0; 100],
            sigma_reaction: sigmas,
            partial: vec![],
        };

        let temperatures = vec![0.1, 0.3, 1.0, 3.0];
        let macs = gpu_macs_integrate(&ctx, &stream, &xs, &temperatures).unwrap();

        assert_eq!(macs.len(), 4);
        for (i, &m) in macs.iter().enumerate() {
            assert!(
                m > 0.0 && m.is_finite(),
                "MACS[T9={}] = {} invalid",
                temperatures[i],
                m
            );
        }
    }
}
