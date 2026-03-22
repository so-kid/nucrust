# Hauser-Feshbach Calculations

The Hauser-Feshbach (HF) statistical model computes compound nucleus reaction cross sections by summing over all spin-parity (Jπ) states:

σ(a,b) = (π/k²) Σ_{J,π} (2J+1)/[(2j_a+1)(2J_A+1)] · T_a · T_b / Σ_c T_c

## Using the HF Module

```rust
use nucrust::nucrust_hf::{HfCalculation, HfConfig};

let config = HfConfig {
    j_max: 30,
    max_l: 20,
    ..Default::default()
};

let mut calc = HfCalculation::new(
    &entrance_channel,
    &transmission_coeffs,
    &exit_channels,
    &nld_model,
    &gsf_model,
    &config,
    None,  // discrete levels (optional)
);

let cross_section = calc.hauser_feshbach();
```

## Width Fluctuation Corrections

WFC accounts for correlations between entrance and exit channels. Three models are available:

- **Moldauer** — 1D numerical integration (fast, good approximation)
- **GOE** — VWZ triple integral (exact, slower)
- **None** — No correction (Weisskopf-Ewing limit)

Set via `HfConfig::wfc_model`.

## Multi-particle Cascade

For reactions where the compound nucleus excitation energy is high enough, sequential particle emission is computed:

```rust
use nucrust::nucrust_hf::cascade::cascade_calculation;

let result = cascade_calculation(&compound, &channels, &nld, &gsf, &config);
// result.cross_sections contains (n,γ), (n,n'), (n,2n), etc.
```

## Discrete vs Continuum Levels

Below the complete level energy E_complete, known discrete levels are used. Above E_complete, the NLD continuum model takes over. This is handled automatically when `DiscreteLevels` are provided to `HfCalculation`.
