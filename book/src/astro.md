# Astrophysical Reaction Rates

The `nucrust-astro` crate computes quantities relevant to stellar nucleosynthesis.

## Maxwellian-Averaged Cross Section (MACS)

MACS is the reaction cross section averaged over a Maxwell-Boltzmann energy distribution at temperature T:

⟨σ⟩(kT) = (2/√π) · (1/kT)² · ∫₀^∞ σ(E) · E · exp(−E/kT) dE

`compute_macs` takes a `CrossSection` (energy grid + σ arrays) and a
`MacsConfig` (quadrature order and temperature grid), and returns one MACS
value (mb) per temperature:

```rust,ignore
use nucrust::nucrust_astro::{compute_macs, MacsConfig};

let config = MacsConfig::default(); // 20-point Gauss-Laguerre, standard T9 grid
let macs = compute_macs(&cross_section, &config)?; // Vec<f64>, mb
```

Uses Gauss-Laguerre quadrature with cubic spline interpolation of the cross section.

## Reaction Rate

The thermonuclear reaction rate NA⟨σv⟩ (cm³/mol/s) is computed from the MACS
values, the temperature grid (GK), and the reduced mass (amu):

```rust,ignore
use nucrust::nucrust_astro::compute_reaction_rate;
use nucrust::nucrust_core::units::reduced_mass;
use nucrust::nucrust_core::Projectile;

let mu = reduced_mass(Projectile::Neutron.mass_amu(), 56.0);
let rate = compute_reaction_rate(&macs, &config.temperature_grid, mu)?;
// rate.temperatures (GK), rate.na_sigma_v (cm³/mol/s), rate.macs (mb)
```

## S-factor

The astrophysical S-factor removes the Coulomb barrier penetration factor for charged-particle reactions:

S(E) = σ(E) · E · exp(2πη)

The charge numbers and reduced mass are taken from the reaction `Channel`;
the function errors for neutral projectiles:

```rust,ignore
use nucrust::nucrust_astro::compute_s_factor;

let s_factors = compute_s_factor(&cross_section, &channel)?; // MeV·b per energy point
```

## Stellar Enhancement Factor (SEF)

Accounts for contributions from thermally excited target states:

SEF(T) = [Σᵢ (2Jᵢ+1) σᵢ(T) exp(−Eᵢ/kT)] / [(2J₀+1) σ₀(T) · G(T)]

```rust,ignore
use nucrust::nucrust_astro::compute_sef;

// (excitation energy MeV, spin as 2J, cross section) per excited state
let excited: Vec<(f64, i32, CrossSection)> = vec![(0.847, 4, xs_first_excited)];

let sef = compute_sef(&gs_cross_section, &excited, &temperatures, /* gs 2J = */ 0)?;
```

## REACLIB Output

Reaction rates can be fitted to the 7-parameter REACLIB format for use in reaction network codes (SkyNet, WinNet, XNet). The fit is a linear least-squares in log space using faer QR decomposition:

```rust,ignore
use nucrust::nucrust_data::reaclib::fit_reaclib_params;

let params: [f64; 7] = fit_reaclib_params(&rate.temperatures, &rate.na_sigma_v)?;
// params[0] = a0, ..., params[6] = a6
```
