# R-matrix Calculations

R-matrix theory describes nuclear reactions through resonance parameters, suitable for light nuclei and isolated resonances where the Hauser-Feshbach statistical model is not applicable.

## Lane-Thomas Formalism

The R-matrix is constructed from level energies and reduced width amplitudes:

R_cc'(E) = Σ_λ γ_λc · γ_λc' / (E_λ − E)

From R, the collision matrix U is derived and cross sections computed.

## Basic Usage

```rust
use nucrust::nucrust_rmatrix::{
    RMatrixParams, RMatrixChannel, RMatrixLevel,
    BoundaryCondition, rmatrix_cross_section,
};

let params = RMatrixParams {
    channels: vec![channel_p, channel_gamma],
    levels: vec![level1, level2, level3],
    boundary_condition: BoundaryCondition::ShiftFunction,
};

let energies = vec![0.1, 0.5, 1.0, 2.0, 5.0];
let result = rmatrix_cross_section(&params, &energies);
```

## Brune Parameterization

The Brune (alternative) parameterization removes the dependence on boundary condition choice, making parameters more physically meaningful:

```rust
use nucrust::nucrust_rmatrix::brune::{standard_to_brune, brune_to_standard};

let brune_params = standard_to_brune(&standard_params);
```

## Parameter Fitting

nucrust provides two fitting methods:

### Levenberg-Marquardt

Gradient-based least-squares minimization:

```rust
use nucrust::nucrust_rmatrix::fitting::levenberg_marquardt;

let result = levenberg_marquardt(&data, &initial_params, &config);
```

### MCMC (Markov Chain Monte Carlo)

Affine-invariant ensemble sampler (Goodman-Weare algorithm) for uncertainty quantification:

```rust
use nucrust::nucrust_rmatrix::fitting::mcmc_sample;

let chains = mcmc_sample(&data, &initial_params, &mcmc_config);
```

MCMC results can be streamed to HDF5 for large sampling runs.
