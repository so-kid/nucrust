//! Multi-particle emission cascade calculation.
//!
//! Uses an explicit stack-based approach (not recursion) for GPU compatibility.

use nucrust_core::{CoreError, Nuclide, SpinParity};

/// State of a single cascade stage.
#[derive(Debug, Clone)]
pub struct CascadeState {
    /// Current residual nuclide.
    pub nuclide: Nuclide,
    /// Excitation energy (MeV).
    pub excitation: f64,
    /// Spin-parity of the state.
    pub spin_parity: SpinParity,
    /// Cascade stage number (0 = initial compound nucleus).
    pub stage: u32,
    /// Population weight (branching fraction from parent).
    pub weight: f64,
}

/// Result of a cascade calculation.
#[derive(Debug, Clone, Default)]
pub struct CascadeResult {
    /// Partial cross sections for each exit channel (mb).
    pub channel_cross_sections: Vec<f64>,
    /// Gamma-ray spectrum (E_gamma, intensity) pairs.
    pub gamma_spectrum: Vec<(f64, f64)>,
    /// Final population of ground and isomeric states.
    pub ground_state_population: f64,
    pub isomer_populations: Vec<(Nuclide, f64)>,
}

/// Perform cascade calculation using explicit stack.
///
/// Starting from the compound nucleus, iteratively:
/// 1. Compute HF branching ratios for particle emission
/// 2. If excitation > separation energy, push particle-emission daughters to stack
/// 3. If excitation < separation energy, perform gamma cascade to ground state
pub fn cascade_calculation(
    initial: CascadeState,
    max_stages: u32,
    max_gamma_steps: u32,
) -> Result<CascadeResult, CoreError> {
    let mut stack = vec![initial];
    let mut result = CascadeResult::default();

    while let Some(state) = stack.pop() {
        if state.stage >= max_stages || state.excitation < 0.1 {
            // Terminal: gamma cascade only
            gamma_cascade(&state, max_gamma_steps, &mut result);
            continue;
        }

        // In a full implementation, we would compute HF branching ratios here
        // and push daughter states to the stack. For now, this is a stub
        // that performs gamma cascade for the current state.
        gamma_cascade(&state, max_gamma_steps, &mut result);
    }

    Ok(result)
}

/// Perform gamma cascade from an excited state to the ground state.
///
/// Simplified model: at each step, emit a gamma ray carrying away the excitation
/// energy to the next lower level, until the ground state is reached.
fn gamma_cascade(state: &CascadeState, max_steps: u32, result: &mut CascadeResult) {
    let mut excitation = state.excitation;
    let mut step = 0;

    while excitation > 0.01 && step < max_steps {
        // Simple model: emit gamma with energy = fraction of excitation
        let e_gamma = excitation.min(2.0); // Typical gamma energy ~1-2 MeV
        result.gamma_spectrum.push((e_gamma, state.weight));
        excitation -= e_gamma;
        step += 1;
    }

    result.ground_state_population += state.weight;
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucrust_core::Parity;

    #[test]
    fn cascade_terminates() {
        let initial = CascadeState {
            nuclide: Nuclide::new(26, 57).unwrap(),
            excitation: 7.6, // ~Sn for Fe-57
            spin_parity: SpinParity::new(1, Parity::Positive).unwrap(),
            stage: 0,
            weight: 1.0,
        };

        let result = cascade_calculation(initial, 3, 30).unwrap();
        assert!(result.ground_state_population > 0.0);
        assert!(!result.gamma_spectrum.is_empty());
    }

    #[test]
    fn cascade_conserves_weight() {
        let initial = CascadeState {
            nuclide: Nuclide::new(26, 57).unwrap(),
            excitation: 5.0,
            spin_parity: SpinParity::new(0, Parity::Positive).unwrap(),
            stage: 0,
            weight: 1.0,
        };

        let result = cascade_calculation(initial, 1, 30).unwrap();
        // All weight should end up in ground state
        assert!(
            (result.ground_state_population - 1.0).abs() < 1e-10,
            "gs_pop = {}",
            result.ground_state_population
        );
    }
}
