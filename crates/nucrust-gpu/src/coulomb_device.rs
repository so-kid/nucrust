//! On-the-fly GPU Coulomb wave function computation (T-3A.11).
//!
//! CUDA __device__ port of the CPU algorithm in `nucrust_special::coulomb_wave`:
//! CF1 (with the exact sign of F) + downward recurrence for F, normalization at
//! l = 0 by Steed's method, the 1F1 power series or a rho shift with Taylor
//! integration (whichever is accurate), and upward recurrence for G.
//! No lookup table needed — each thread computes F, G, F', G' independently.

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use nucrust_core::CoreError;
use std::sync::Arc;

const COULOMB_DEVICE_SRC: &str = include_str!("../../../kernels/coulomb_device.cu");

/// Output of [`gpu_coulomb_batch_full`], one entry per input point.
#[derive(Debug, Clone)]
pub struct GpuCoulombBatch {
    /// F_l(eta, rho), with its true sign.
    pub f: Vec<f64>,
    /// G_l(eta, rho), with its true sign.
    pub g: Vec<f64>,
    /// F'_l(eta, rho).
    pub fp: Vec<f64>,
    /// G'_l(eta, rho).
    pub gp: Vec<f64>,
    /// 0 on success; negative if a continued fraction did not converge
    /// (-1: CF1, -2: CF2), in which case the values are NaN.
    pub status: Vec<i32>,
}

/// Compute Coulomb functions F_l, G_l on GPU for a batch of (eta, rho, l) points.
///
/// See [`gpu_coulomb_batch_full`] for derivatives and per-point status.
pub fn gpu_coulomb_batch(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    eta: &[f64],
    rho: &[f64],
    l: &[i32],
) -> Result<(Vec<f64>, Vec<f64>), CoreError> {
    let out = gpu_coulomb_batch_full(ctx, stream, eta, rho, l)?;
    Ok((out.f, out.g))
}

/// Compute Coulomb functions F_l, G_l, F'_l, G'_l on GPU for a batch of
/// (eta, rho, l) points.
pub fn gpu_coulomb_batch_full(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    eta: &[f64],
    rho: &[f64],
    l: &[i32],
) -> Result<GpuCoulombBatch, CoreError> {
    let n = eta.len();
    assert_eq!(n, rho.len());
    assert_eq!(n, l.len());

    let ptx = cudarc::nvrtc::compile_ptx(COULOMB_DEVICE_SRC).map_err(|_| {
        CoreError::InvalidParameter {
            name: "nvrtc",
            value: 0.0,
            reason: "failed to compile Coulomb device kernel",
        }
    })?;

    let module = CudaContext::load_module(ctx, ptx).map_err(|_| CoreError::InvalidParameter {
        name: "load_module",
        value: 0.0,
        reason: "failed to load Coulomb device module",
    })?;

    let func =
        module
            .load_function("coulomb_batch_device")
            .map_err(|_| CoreError::DataNotFound {
                description: "coulomb_batch_device not found".to_string(),
            })?;

    let mut d_eta = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_rho = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_l = stream.alloc_zeros::<i32>(n).map_err(gpu_err)?;
    let mut d_f = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_g = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_fp = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_gp = stream.alloc_zeros::<f64>(n).map_err(gpu_err)?;
    let mut d_status = stream.alloc_zeros::<i32>(n).map_err(gpu_err)?;

    stream.memcpy_htod(eta, &mut d_eta).map_err(gpu_err)?;
    stream.memcpy_htod(rho, &mut d_rho).map_err(gpu_err)?;
    stream.memcpy_htod(l, &mut d_l).map_err(gpu_err)?;

    let block_size = 256u32;
    let grid_size = (n as u32).div_ceil(block_size);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&d_eta)
            .arg(&d_rho)
            .arg(&d_l)
            .arg(&mut d_f)
            .arg(&mut d_g)
            .arg(&mut d_fp)
            .arg(&mut d_gp)
            .arg(&mut d_status)
            .arg(&(n as i32))
            .launch(LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|_| CoreError::NumericalOverflow {
                context: "Coulomb device kernel launch",
            })?;
    }

    let mut out = GpuCoulombBatch {
        f: vec![0.0; n],
        g: vec![0.0; n],
        fp: vec![0.0; n],
        gp: vec![0.0; n],
        status: vec![0; n],
    };
    stream.memcpy_dtoh(&d_f, &mut out.f).map_err(gpu_err)?;
    stream.memcpy_dtoh(&d_g, &mut out.g).map_err(gpu_err)?;
    stream.memcpy_dtoh(&d_fp, &mut out.fp).map_err(gpu_err)?;
    stream.memcpy_dtoh(&d_gp, &mut out.gp).map_err(gpu_err)?;
    stream
        .memcpy_dtoh(&d_status, &mut out.status)
        .map_err(gpu_err)?;
    stream.synchronize().map_err(gpu_err)?;

    Ok(out)
}

fn gpu_err<E>(_: E) -> CoreError {
    CoreError::NumericalOverflow {
        context: "GPU Coulomb operation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    fn rel_err(computed: f64, reference: f64) -> f64 {
        if reference.abs() < 1e-300 {
            computed.abs()
        } else {
            ((computed - reference) / reference).abs()
        }
    }

    fn setup() -> (Arc<CudaContext>, Arc<CudaStream>) {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        (ctx, stream)
    }

    #[test]
    fn gpu_coulomb_bessel_limit() {
        // eta=0: F_0 = sin(rho), G_0 = cos(rho), signs included.
        let (ctx, stream) = setup();
        let rhos = vec![1.0, 2.0, 3.0, 5.0];
        let out = gpu_coulomb_batch_full(&ctx, &stream, &[0.0; 4], &rhos, &[0; 4]).unwrap();

        for (i, &rho) in rhos.iter().enumerate() {
            assert_eq!(out.status[i], 0);
            for (name, got, want) in [
                ("F", out.f[i], rho.sin()),
                ("G", out.g[i], rho.cos()),
                ("F'", out.fp[i], rho.cos()),
                ("G'", out.gp[i], -rho.sin()),
            ] {
                assert!(
                    (got - want).abs() < 1e-13,
                    "{name}_0(0, {rho}): GPU={got}, expected={want}"
                );
            }
        }
    }

    #[test]
    fn gpu_coulomb_wronskian() {
        // F'G - FG' = 1, including deep in the classically forbidden region.
        let (ctx, stream) = setup();
        let etas = vec![1.0, 2.0, 5.0, 10.0, 30.0, 100.0];
        let rhos = vec![5.0, 8.0, 15.0, 3.0, 10.0, 130.0];
        let ls = vec![0, 1, 2, 10, 5, 0];

        let out = gpu_coulomb_batch_full(&ctx, &stream, &etas, &rhos, &ls).unwrap();
        for i in 0..etas.len() {
            assert_eq!(out.status[i], 0, "status at point {i}");
            let w = out.fp[i] * out.g[i] - out.f[i] * out.gp[i];
            assert!((w - 1.0).abs() < 1e-13, "Wronskian[{i}] = {w}");
        }
    }

    #[test]
    fn gpu_coulomb_large_batch_matches_cpu() {
        let (ctx, stream) = setup();
        let n = 1000;
        let etas: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let rhos: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.03).collect();
        let ls: Vec<i32> = (0..n).map(|i| (i % 5) as i32).collect();

        let out = gpu_coulomb_batch_full(&ctx, &stream, &etas, &rhos, &ls).unwrap();
        for i in 0..n {
            assert_eq!(out.status[i], 0, "status at point {i}");
            let cpu = nucrust_special::coulomb_wave(etas[i], rhos[i], ls[i] as u32, 1).unwrap();
            for (name, got, want) in [
                ("F", out.f[i], cpu.f[0]),
                ("G", out.g[i], cpu.g[0]),
                ("F'", out.fp[i], cpu.fp[0]),
                ("G'", out.gp[i], cpu.gp[0]),
            ] {
                let err = rel_err(got, want);
                assert!(
                    err < 1e-12,
                    "{name}_{}({}, {}): GPU={got}, CPU={want}, rel_err={err:.2e}",
                    ls[i],
                    etas[i],
                    rhos[i]
                );
            }
        }
    }

    #[derive(Deserialize)]
    struct ReferenceFile {
        data: BTreeMap<String, Vec<ReferencePoint>>,
    }

    #[derive(Deserialize)]
    struct ReferencePoint {
        l: i32,
        eta: f64,
        rho: f64,
        label: String,
        #[serde(rename = "F")]
        f: f64,
        #[serde(rename = "G")]
        g: f64,
        #[serde(rename = "Fp")]
        fp: f64,
        #[serde(rename = "Gp")]
        gp: f64,
    }

    #[test]
    fn gpu_coulomb_mpmath_reference() {
        // All signed mpmath points (tests/reference_data/coulomb_mpmath.json),
        // against the SRS target of 1e-12 relative error in FP64.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/reference_data/coulomb_mpmath.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let reference: ReferenceFile = serde_json::from_str(&text).unwrap();
        let points: Vec<&ReferencePoint> = reference.data.values().flatten().collect();
        assert!(
            points.len() >= 100,
            "only {} reference points",
            points.len()
        );

        let etas: Vec<f64> = points.iter().map(|p| p.eta).collect();
        let rhos: Vec<f64> = points.iter().map(|p| p.rho).collect();
        let ls: Vec<i32> = points.iter().map(|p| p.l).collect();
        let (ctx, stream) = setup();
        let out = gpu_coulomb_batch_full(&ctx, &stream, &etas, &rhos, &ls).unwrap();

        for (i, p) in points.iter().enumerate() {
            assert_eq!(out.status[i], 0, "status [{}]", p.label);
            for (name, got, want) in [
                ("F", out.f[i], p.f),
                ("G", out.g[i], p.g),
                ("F'", out.fp[i], p.fp),
                ("G'", out.gp[i], p.gp),
            ] {
                let err = rel_err(got, want);
                assert!(
                    err < 1e-12,
                    "{name}_{}({}, {}): GPU={got}, ref={want}, rel_err={err:.2e} [{}]",
                    p.l,
                    p.eta,
                    p.rho,
                    p.label
                );
            }
        }
    }
}
