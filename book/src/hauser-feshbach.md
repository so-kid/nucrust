# Hauser-Feshbach Calculations

The Hauser-Feshbach (HF) statistical model computes compound nucleus reaction cross sections by summing over all spin-parity (Jπ) states:

σ(a,b) = (π/k²) Σ_{J,π} (2J+1)/[(2j_a+1)(2J_A+1)] · T_a · T_b / Σ_c T_c

## Using the HF Module

`HfCalculation` is a plain context struct (no constructor); fill in its fields
and pass it to the free function `hauser_feshbach`:

```rust,ignore
use nucrust::nucrust_core::Projectile;
use nucrust::nucrust_hf::hf::{hauser_feshbach, HfCalculation, HfConfig};
use nucrust::nucrust_hf::{gsf, nld};

// Physics models (here: ⁵⁷Fe compound nucleus defaults)
let nld = nld::ConstantTemperature { temperature: 0.88, e0: -1.16, a: 6.21 };
let gsf = gsf::StandardLorentzian {
    e_gdr: 16.36,
    gamma_gdr: 4.58,
    sigma_gdr: 136.0,
    m1_params: None,
};

let config = HfConfig {
    two_j_max: 60,                            // stored as 2J, so J_max = 30
    exit_channels: vec![Projectile::Gamma],   // (n,γ) only
    ..HfConfig::default()
};

let calc = HfCalculation {
    entrance: &channel,             // &Channel (e.g. n + ⁵⁶Fe)
    tc_entrance: &tc,               // &TransmissionCoeffs from nucrust-optical
    exit_particle_channels: vec![], // ExitChannelData for particle exits
    nld: &nld,
    gsf: &gsf,
    config: &config,
    discrete_levels: None,
};

let results = hauser_feshbach(&calc)?; // Vec<HfResult>, one per energy point
for r in &results {
    // r.sigma_cn: compound nucleus formation cross section (mb)
    // r.sigma_channels: per-exit-channel cross sections (mb),
    //                   indexed like config.exit_channels
}
```

`HfConfig` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `two_j_max` | `60` | Maximum total angular momentum, in units of 2J (J_max = 30) |
| `wfc_model` | `Moldauer` | Width fluctuation correction model |
| `wfc_quadrature` | `8` | Quadrature points for WFC integration |
| `exit_channels` | n, p, α, γ | Exit channels to include |

## Width Fluctuation Corrections

WFC accounts for correlations between entrance and exit channels. Three models are available:

- **Moldauer** — 1D numerical integration (fast, good approximation)
- **GOE** — VWZ triple integral (exact, slower)
- **None** — No correction (Weisskopf-Ewing limit)

Set via `HfConfig::wfc_model`.

## Multi-particle Cascade

For reactions where the compound nucleus excitation energy is high enough, sequential particle emission is computed with an explicit stack. The cascade needs a `CascadeContext` carrying the physics models and lookup callbacks:

```rust,ignore
use nucrust::nucrust_hf::cascade::{cascade_calculation, CascadeContext, CascadeState};

let ctx = CascadeContext {
    nld: &nld,
    gsf: &gsf,
    max_stages: 3,
    max_gamma_steps: 10,
    exit_particles: vec![Projectile::Neutron, Projectile::Proton, Projectile::Alpha],
    separation_energies: Box::new(|nuclide| /* mass-table lookup */ sep_energies(nuclide)),
    transmission_lookup: Box::new(|proj, nuclide, e| /* T(E) lookup */ lookup_t(proj, nuclide, e)),
};

let result = cascade_calculation(initial_state, &ctx)?;
// result.channel_cross_sections contains (n,γ), (n,n'), (n,2n), etc.
```

## Discrete vs Continuum Levels

Below the complete level energy `e_complete`, known discrete levels are used. Above it, the NLD continuum model takes over. This is handled automatically when a `DiscreteLevels` value is supplied via `HfCalculation::discrete_levels` (for the compound nucleus) or `ExitChannelData::daughter_discrete` (for residual nuclei).
