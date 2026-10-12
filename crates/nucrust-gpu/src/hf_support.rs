//! Hauser-Feshbach options supported by the GPU kernel.
//!
//! The check is pure and available without the `cuda` feature; the GPU
//! [`ComputeBackend::hf_summation`](nucrust_core::backend::ComputeBackend::hf_summation)
//! calls it before launching the kernel.

use nucrust_core::backend::{HfConfig, WfcModel};
use nucrust_core::CoreError;

/// Reject Hauser-Feshbach options that the GPU kernel (`kernels/hf_summation.cu`)
/// does not implement, instead of silently computing a different model than the
/// CPU backend.
///
/// The kernel has no width fluctuation correction, so only `WfcModel::None` is
/// accepted (`HfConfig::default()` uses `Moldauer`).
pub fn check_hf_config(config: &HfConfig) -> Result<(), CoreError> {
    if config.wfc_model != WfcModel::None {
        return Err(CoreError::InvalidParameter {
            name: "wfc_model",
            value: f64::NAN,
            reason: "the GPU Hauser-Feshbach kernel has no width fluctuation correction; \
                     set wfc_model to WfcModel::None or use the CPU backend",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wfc_models_are_rejected() {
        for wfc_model in [WfcModel::Moldauer, WfcModel::Goe] {
            let config = HfConfig {
                wfc_model,
                ..HfConfig::default()
            };
            let err = check_hf_config(&config).unwrap_err();
            assert!(
                matches!(
                    err,
                    CoreError::InvalidParameter {
                        name: "wfc_model",
                        ..
                    }
                ),
                "{wfc_model:?}: {err}"
            );
        }
        // The default config asks for Moldauer, so it is rejected too.
        assert!(check_hf_config(&HfConfig::default()).is_err());
    }

    #[test]
    fn no_wfc_is_accepted() {
        let config = HfConfig {
            wfc_model: WfcModel::None,
            ..HfConfig::default()
        };
        assert!(check_hf_config(&config).is_ok());
    }
}
