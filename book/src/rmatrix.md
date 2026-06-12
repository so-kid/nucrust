# R-matrix Calculations

R-matrix theory describes nuclear reactions through resonance parameters, suitable for light nuclei and isolated resonances where the Hauser-Feshbach statistical model is not applicable.

## Lane-Thomas Formalism

The R-matrix is constructed from level energies and reduced width amplitudes:

R_cc'(E) = Σ_λ γ_λc · γ_λc' / (E_λ − E)

From R, the collision matrix U is derived and cross sections computed.

## Basic Usage

```rust,ignore
use nucrust::nucrust_rmatrix::{
    rmatrix_cross_section, BoundaryCondition, RMatrixChannel, RMatrixLevel,
    RMatrixParams,
};

let params = RMatrixParams {
    channels: vec![channel_p, channel_gamma],
    levels: vec![level1, level2, level3],
    // Standard R-matrix with explicit B_c per channel,
    // or BoundaryCondition::Brune for the Brune parameterization
    boundary_condition: BoundaryCondition::Standard { b: vec![0.0, 0.0] },
};

let energies = vec![0.1, 0.5, 1.0, 2.0, 5.0];
let result = rmatrix_cross_section(&params, &energies)?;
```

## Brune Parameterization

The Brune (alternative) parameterization removes the dependence on boundary condition choice, making parameters more physically meaningful:

```rust,ignore
use nucrust::nucrust_rmatrix::{brune_to_standard, standard_to_brune};

let brune_params = standard_to_brune(&standard_params)?;
```

## Parameter Fitting

nucrust provides two fitting methods:

### Levenberg-Marquardt

Gradient-based least-squares minimization:

```rust,ignore
use nucrust::nucrust_rmatrix::levenberg_marquardt;

// params: &mut RMatrixParams (updated in place with fitted values)
// free_params: &[ParamIndex] selecting which parameters to vary
let result = levenberg_marquardt(&mut params, &free_params, &exp_data, &lm_config)?;
```

### MCMC (Markov Chain Monte Carlo)

Affine-invariant ensemble sampler (Goodman-Weare algorithm) for uncertainty quantification:

```rust,ignore
use nucrust::nucrust_rmatrix::mcmc_sample;

let chains = mcmc_sample(&params, &free_params, &exp_data, &mcmc_config)?;
```

MCMC results can be streamed to HDF5 for large sampling runs.
